"""Joint multi-dataset segmentation with partial labels.

One 6-class segmentation model is needed for the end-to-end CVS pipeline, but
the two available datasets are complementary and each only *partially* labels
the shared class space (see :mod:`src.losses.partial_label` for the full
rationale):

    CholecSeg8k      labels  background, liver, gallbladder, tool
                     (cystic_duct effectively unlabelled; cystic_artery absent)
    Endoscapes-Seg   labels  background, gallbladder, cystic_duct,
                     cystic_artery, tool   (no liver label)

This module unions the two into a single training set. Each item carries a
per-sample ``labeled`` boolean mask over the 6 classes recording which classes
its source annotates, so the marginal loss can supervise only those and fold the
rest into background. The wrappers preserve the ``{image, mask}`` /
``_load_raw`` / ``load_mask`` contract shared by the CholecSeg8k and
Endoscapes-Seg datasets, so the class-stats pass, the weighted sampler and
copy-paste all keep working unchanged.
"""
from __future__ import annotations

import bisect

import torch
from torch.utils.data import Dataset

from src.data.cholecseg8k import CLASS_NAMES, NUM_CLASSES

# Which of the 6 shared classes each source dataset reliably annotates. The
# classes NOT listed are present in the frames but folded into that source's
# background label, so the marginal loss leaves them unsupervised there rather
# than pushing them to background (which would fight the other dataset).
CHOLECSEG8K_LABELED_CLASSES: tuple[int, ...] = tuple(
    CLASS_NAMES.index(name)
    for name in ("background", "liver", "gallbladder", "tool"))
ENDOSCAPES_LABELED_CLASSES: tuple[int, ...] = tuple(
    CLASS_NAMES.index(name)
    for name in ("background", "gallbladder", "cystic_duct",
                 "cystic_artery", "tool"))


def labeled_class_mask(class_indices, num_classes: int = NUM_CLASSES
                       ) -> torch.Tensor:
    """Boolean ``(num_classes,)`` mask; background (index 0) is always True."""
    mask = torch.zeros(num_classes, dtype=torch.bool)
    for index in class_indices:
        mask[int(index)] = True
    mask[0] = True
    return mask


class LabeledClassDataset(Dataset):
    """Tag every item of a segmentation dataset with a ``labeled`` class mask.

    Wraps a CholecSeg8k / Endoscapes-Seg dataset and adds a ``labeled`` field --
    the ``(num_classes,)`` boolean mask of the classes the wrapped dataset
    annotates -- to each returned item. ``load_mask`` / ``_load_raw`` / ``len``
    delegate to the base so the class-stats fast path and copy-paste keep
    working. Use this for the val/test splits too (a single source), so every
    batch carries ``labeled`` and the training loop always takes the marginal
    path.
    """

    def __init__(self, base: Dataset, labeled_classes,
                 num_classes: int = NUM_CLASSES):
        self.base = base
        self.labeled = labeled_class_mask(labeled_classes, num_classes)

    def __len__(self) -> int:
        return len(self.base)

    def load_mask(self, index: int):
        return self.base.load_mask(index)

    def _load_raw(self, index: int):
        return self.base._load_raw(index)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        item["labeled"] = self.labeled.clone()
        return item


class JointSegDataset(Dataset):
    """Concatenate several :class:`LabeledClassDataset` into one training set.

    Behaves like ``torch.utils.data.ConcatDataset`` but forwards ``load_mask``
    and ``_load_raw`` to the owning sub-dataset, so the class-stats pass (which
    counts mask pixels via ``load_mask``) and copy-paste harvesting (which reads
    raw pairs via ``_load_raw``) run over the union transparently. Item order is
    ``[sub-dataset 0 ..., sub-dataset 1 ...]`` and stable, so per-frame sampler
    weights computed in this order stay aligned with the sampler.
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("JointSegDataset needs at least one sub-dataset.")
        sizes, running = [], 0
        for dataset in self.datasets:
            running += len(dataset)
            sizes.append(running)
        self._cumulative = sizes

    def __len__(self) -> int:
        return self._cumulative[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(sub-dataset, local index)``."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self._cumulative, index)
        offset = self._cumulative[dataset_index - 1] if dataset_index else 0
        return dataset_index, index - offset

    def load_mask(self, index: int):
        dataset_index, local = self._locate(index)
        return self.datasets[dataset_index].load_mask(local)

    def _load_raw(self, index: int):
        dataset_index, local = self._locate(index)
        return self.datasets[dataset_index]._load_raw(local)

    def __getitem__(self, index: int) -> dict:
        dataset_index, local = self._locate(index)
        return self.datasets[dataset_index][local]
