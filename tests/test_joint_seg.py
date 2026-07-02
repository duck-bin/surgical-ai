"""Tests for the joint multi-dataset segmentation wrappers.

Exercises the concatenation/indexing, the per-sample ``labeled`` mask and the
``load_mask`` / ``_load_raw`` delegation (used by the class-stats pass and
copy-paste) against tiny in-memory stub datasets -- no data download.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cholecseg8k import CLASS_NAMES, NUM_CLASSES
from src.data.joint_seg import (
    CHOLECSEG8K_LABELED_CLASSES,
    ENDOSCAPES_LABELED_CLASSES,
    JointSegDataset,
    LabeledClassDataset,
    labeled_class_mask,
)

DUCT = CLASS_NAMES.index("cystic_duct")
ARTERY = CLASS_NAMES.index("cystic_artery")
LIVER = CLASS_NAMES.index("liver")


class _StubSeg(Dataset):
    """Minimal seg dataset: item id encoded into image/mask so we can trace it."""

    def __init__(self, n, tag):
        self.n = n
        self.tag = tag

    def __len__(self):
        return self.n

    def _load_raw(self, index):
        image = np.full((2, 2, 3), self.tag * 100 + index, dtype=np.uint8)
        mask = np.full((2, 2), index % NUM_CLASSES, dtype=np.uint8)
        return image, mask

    def load_mask(self, index):
        return np.full((2, 2), index % NUM_CLASSES, dtype=np.uint8)

    def __getitem__(self, index):
        image, mask = self._load_raw(index)
        return {"image": torch.from_numpy(image).permute(2, 0, 1).float(),
                "mask": torch.from_numpy(mask).long()}


def test_labeled_class_mask_marks_background_and_given_classes():
    mask = labeled_class_mask(CHOLECSEG8K_LABELED_CLASSES)
    assert mask.dtype == torch.bool and mask.shape == (NUM_CLASSES,)
    assert mask[0]                        # background always labelled
    assert mask[LIVER]                    # CholecSeg8k labels liver
    assert not mask[DUCT] and not mask[ARTERY]   # ...but not duct/artery


def test_endoscapes_labels_duct_and_artery_but_not_liver():
    mask = labeled_class_mask(ENDOSCAPES_LABELED_CLASSES)
    assert mask[DUCT] and mask[ARTERY]
    assert not mask[LIVER]


def test_labeled_dataset_adds_mask_to_every_item():
    base = _StubSeg(3, tag=1)
    wrapped = LabeledClassDataset(base, ENDOSCAPES_LABELED_CLASSES)
    item = wrapped[0]
    assert "labeled" in item
    assert torch.equal(item["labeled"], labeled_class_mask(ENDOSCAPES_LABELED_CLASSES))
    # delegation stays intact for the stats/copy-paste fast paths
    assert len(wrapped) == 3
    assert np.array_equal(wrapped.load_mask(1), base.load_mask(1))
    assert np.array_equal(wrapped._load_raw(2)[0], base._load_raw(2)[0])


def test_joint_dataset_concatenates_and_routes_indices():
    endo = LabeledClassDataset(_StubSeg(2, tag=1), ENDOSCAPES_LABELED_CLASSES)
    cholec = LabeledClassDataset(_StubSeg(3, tag=2), CHOLECSEG8K_LABELED_CLASSES)
    joint = JointSegDataset([endo, cholec])

    assert len(joint) == 5
    # First two items come from Endoscapes, carry the Endoscapes mask.
    assert torch.equal(joint[0]["labeled"],
                       labeled_class_mask(ENDOSCAPES_LABELED_CLASSES))
    assert bool(joint[1]["labeled"][DUCT])
    # Remaining three come from CholecSeg8k, carry the CholecSeg8k mask.
    assert torch.equal(joint[2]["labeled"],
                       labeled_class_mask(CHOLECSEG8K_LABELED_CLASSES))
    assert not bool(joint[4]["labeled"][DUCT])


def test_joint_load_mask_and_raw_delegate_to_owning_subdataset():
    endo = LabeledClassDataset(_StubSeg(2, tag=1), ENDOSCAPES_LABELED_CLASSES)
    cholec = LabeledClassDataset(_StubSeg(3, tag=2), CHOLECSEG8K_LABELED_CLASSES)
    joint = JointSegDataset([endo, cholec])

    # Global index 3 -> CholecSeg8k local index 1 (tag 2 -> image value 201).
    image, _ = joint._load_raw(3)
    assert int(image.flat[0]) == 2 * 100 + 1
    assert np.array_equal(joint.load_mask(3), cholec.load_mask(1))
    # Negative indexing resolves against the total length.
    assert np.array_equal(joint.load_mask(-1), cholec.load_mask(2))


def test_joint_default_collate_stacks_labeled_masks():
    from torch.utils.data import DataLoader

    endo = LabeledClassDataset(_StubSeg(2, tag=1), ENDOSCAPES_LABELED_CLASSES)
    cholec = LabeledClassDataset(_StubSeg(2, tag=2), CHOLECSEG8K_LABELED_CLASSES)
    joint = JointSegDataset([endo, cholec])
    batch = next(iter(DataLoader(joint, batch_size=4, shuffle=False)))
    assert batch["labeled"].shape == (4, NUM_CLASSES)
    assert batch["labeled"].dtype == torch.bool
    assert bool(batch["labeled"][0, DUCT])          # endoscapes row
    assert not bool(batch["labeled"][2, DUCT])      # cholecseg8k row
