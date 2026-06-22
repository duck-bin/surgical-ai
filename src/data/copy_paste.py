"""Copy-paste augmentation for rare anatomical classes.

SOTA context. The single biggest obstacle to segmenting ``cystic_duct``
(~0.03% of pixels) -- and, from Endoscapes, ``cystic_artery`` -- is not the loss
or the sampler but *exposure*: most training frames contain no rare-class pixels
at all, so there is almost no boundary signal to learn from. Loss reweighting
and a weighted sampler can only re-emphasise the few frames that already carry
the class; they cannot create new ones.

Copy-paste (Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation
Method for Instance Segmentation", CVPR 2021) directly manufactures rare-class
instances: harvest patches of the rare structure from the frames that *do* label
it, then paste them -- with light scale/flip jitter -- onto other training
frames, updating both the image and the mask. It is the canonical strong
augmentation for rare/small-object segmentation and is widely used in surgical
and medical scene segmentation, where rare critical structures are exactly the
ones that matter. Here it complements the existing fixes: the loss penalises
misses harder, the sampler re-selects rare frames, and copy-paste *multiplies*
the rare frames (and, crucially, their boundaries) the model ever sees.

Design. :func:`build_bank` harvests one crop per (frame, rare-class) with a
context margin; :class:`RareClassCopyPaste` pastes from that bank. Pasting is
split into :meth:`~RareClassCopyPaste.sample_plan` (decide what/where) and
:meth:`~RareClassCopyPaste.apply` (stamp it), so a temporal clip can sample one
plan and apply it identically to every frame -- keeping the paste spatially
consistent across the window the ConvGRU sees. Operates on raw ``(H, W, 3)``
uint8 images and ``(H, W)`` uint8 masks, *before* the albumentations pipeline,
so the pasted region also goes through the geometric/photometric augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Instance:
    """A harvested rare-class patch: an image crop, its mask, and the class id."""
    image: np.ndarray  # (h, w, 3) uint8
    mask: np.ndarray   # (h, w) uint8, == class on the structure, 0 elsewhere
    label: int


def build_bank(raw_pairs, classes, *, margin: float = 0.25,
               min_pixels: int = 20, max_per_class: int = 300) -> list[_Instance]:
    """Harvest rare-class instance crops from ``(image, mask)`` raw pairs.

    For each frame and each requested class, the class's bounding box is cropped
    with a ``margin`` of surrounding context (so the paste carries a little
    natural tissue boundary), and the crop's mask keeps *only* that class (other
    labels zeroed) so pasting stamps the structure, not a rectangle.

    Args:
        raw_pairs: iterable of ``(image_HWC_uint8, mask_HW_int)``.
        classes: rare class indices to harvest (e.g. ``[3]`` for cystic_duct).
        margin: context added around the bounding box, as a fraction of its size.
        min_pixels: skip instances smaller than this (noise / sliver labels).
        max_per_class: stop harvesting a class once this many crops are banked.
    """
    classes = list(classes)
    bank: list[_Instance] = []
    counts = {c: 0 for c in classes}
    for image, mask in raw_pairs:
        if all(counts[c] >= max_per_class for c in classes):
            break
        image = np.asarray(image)
        mask = np.asarray(mask)
        for c in classes:
            if counts[c] >= max_per_class:
                continue
            ys, xs = np.where(mask == c)
            if ys.size < min_pixels:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            height, width = y1 - y0 + 1, x1 - x0 + 1
            my, mx = int(round(height * margin)), int(round(width * margin))
            yy0, yy1 = max(0, y0 - my), min(mask.shape[0], y1 + 1 + my)
            xx0, xx1 = max(0, x0 - mx), min(mask.shape[1], x1 + 1 + mx)
            crop_mask = np.where(mask[yy0:yy1, xx0:xx1] == c, c, 0).astype(np.uint8)
            bank.append(_Instance(
                image=image[yy0:yy1, xx0:xx1].copy(),
                mask=crop_mask,
                label=int(c),
            ))
            counts[c] += 1
    return bank


def instances_from_dataset(dataset, classes, **kwargs) -> list[_Instance]:
    """:func:`build_bank` over a CholecSeg8k-style dataset (uses ``_load_raw``)."""
    def raw_pairs():
        for i in range(len(dataset)):
            yield dataset._load_raw(i)
    return build_bank(raw_pairs(), classes, **kwargs)


class RareClassCopyPaste:
    """Paste harvested rare-class instances onto a frame (or a whole clip).

    Args:
        bank: instances from :func:`build_bank` / :func:`instances_from_dataset`.
        paste_prob: probability that a given call pastes anything at all.
        max_instances: up to this many instances are pasted per call.
        scale_range: uniform scale jitter applied to each pasted instance.
        p_flip: probability of horizontally flipping a pasted instance.
        seed: seeds the internal RNG used when no ``rng`` is passed.
    """

    def __init__(self, bank, *, paste_prob: float = 0.5, max_instances: int = 2,
                 scale_range: tuple[float, float] = (0.7, 1.3),
                 p_flip: float = 0.5, seed: int | None = None):
        self.bank = list(bank)
        self.paste_prob = paste_prob
        self.max_instances = max_instances
        self.scale_range = scale_range
        self.p_flip = p_flip
        self._rng = np.random.default_rng(seed)

    def sample_plan(self, height: int, width: int, rng=None) -> list[dict]:
        """Decide what to paste and where, for a target of size ``(height, width)``.

        Returns a list of paste ops (instance index, top-left, size, flip). An
        empty list means "paste nothing" (drawn with prob ``1 - paste_prob``).
        The plan is independent of pixel content, so the same plan can be applied
        to every frame of a temporal clip for a spatially consistent paste.
        """
        rng = self._rng if rng is None else rng
        if not self.bank or rng.random() > self.paste_prob:
            return []
        plan: list[dict] = []
        for _ in range(int(rng.integers(1, self.max_instances + 1))):
            index = int(rng.integers(len(self.bank)))
            inst = self.bank[index]
            scale = float(rng.uniform(*self.scale_range))
            patch_h = max(1, int(round(inst.image.shape[0] * scale)))
            patch_w = max(1, int(round(inst.image.shape[1] * scale)))
            if patch_h > height or patch_w > width:
                continue  # instance bigger than the frame -> skip this one
            plan.append({
                "index": index,
                "scale": scale, "flip": bool(rng.random() < self.p_flip),
                "y0": int(rng.integers(0, height - patch_h + 1)),
                "x0": int(rng.integers(0, width - patch_w + 1)),
                "height": patch_h, "width": patch_w,
            })
        return plan

    def apply(self, image: np.ndarray, mask: np.ndarray, plan) -> tuple:
        """Stamp a ``sample_plan`` onto ``(image, mask)``; returns new arrays.

        Only the instance's class pixels are written (image and mask), so the
        surrounding context crop is *not* stamped -- the paste follows the
        structure's shape, not its bounding rectangle.
        """
        if not plan:
            return image, mask
        image = np.array(image, copy=True)
        mask = np.array(mask, copy=True)
        for op in plan:
            inst = self.bank[op["index"]]
            patch_img, patch_mask = inst.image, inst.mask
            if (patch_img.shape[0], patch_img.shape[1]) != (op["height"], op["width"]):
                patch_img, patch_mask = _resize(patch_img, patch_mask,
                                                 op["height"], op["width"])
            if op["flip"]:
                patch_img = patch_img[:, ::-1]
                patch_mask = patch_mask[:, ::-1]
            region = patch_mask == inst.label
            ys = slice(op["y0"], op["y0"] + op["height"])
            xs = slice(op["x0"], op["x0"] + op["width"])
            image[ys, xs][region] = patch_img[region]
            mask[ys, xs][region] = inst.label
        return image, mask

    def __call__(self, image, mask, rng=None) -> tuple:
        """Sample a plan for this frame and apply it (the per-frame path)."""
        plan = self.sample_plan(image.shape[0], image.shape[1], rng)
        return self.apply(image, mask, plan)


def _resize(image: np.ndarray, mask: np.ndarray, height: int, width: int):
    """Resize a patch (bilinear image, nearest mask). cv2 is imported lazily."""
    import cv2

    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image, mask
