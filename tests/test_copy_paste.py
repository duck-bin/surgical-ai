"""Tests for rare-class copy-paste augmentation.

Synthetic raw frames only (no dataset, no cv2: scale is left at 1.0 so the
resize path -- the only cv2 user -- is not exercised). The properties checked
are the ones that make copy-paste a valid rare-class augmentation: it increases
rare-class prevalence, writes the correct label and pixels, preserves other
labels, is a no-op at paste_prob=0, is reproducible, and applies one plan
identically across a clip (temporal consistency).
"""
import numpy as np

from src.data.copy_paste import (
    RareClassCopyPaste,
    build_bank,
    instances_from_dataset,
)

DUCT = 3
SIZE = 32


def _source_frame(side=8, value=200, label=DUCT):
    """A frame whose centre holds a labelled square of constant colour."""
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    a, b = SIZE // 2 - side // 2, SIZE // 2 + side // 2
    image[a:b, a:b] = value
    mask[a:b, a:b] = label
    return image, mask, side * side


def _bank(side=8, value=200):
    image, mask, _ = _source_frame(side, value)
    return build_bank([(image, mask)], [DUCT], margin=0.0, min_pixels=4)


def test_build_bank_harvests_only_rare_class():
    """A harvested instance carries the class pixels and nothing else."""
    image, mask, area = _source_frame()
    # Add an unrelated class -- it must NOT appear in the duct instance.
    mask[0:4, 0:4] = 1
    bank = build_bank([(image, mask)], [DUCT], margin=0.0, min_pixels=4)

    assert len(bank) == 1
    inst = bank[0]
    assert inst.label == DUCT
    assert set(np.unique(inst.mask)) <= {0, DUCT}
    assert int((inst.mask == DUCT).sum()) == area


def test_build_bank_respects_min_pixels_and_cap():
    """Sliver instances are skipped; max_per_class caps the harvest."""
    small_img, small_mask, _ = _source_frame(side=2)   # 4 px
    big_img, big_mask, _ = _source_frame(side=8)        # 64 px
    pairs = [(small_img, small_mask), (big_img, big_mask), (big_img, big_mask)]

    assert len(build_bank(pairs, [DUCT], min_pixels=16)) == 2   # small skipped
    assert len(build_bank(pairs, [DUCT], min_pixels=1, max_per_class=1)) == 1


def test_copy_paste_increases_rare_prevalence_and_writes_pixels():
    """Pasting onto a duct-free frame adds exactly the instance's class pixels."""
    bank = _bank(side=8, value=200)
    cp = RareClassCopyPaste(bank, paste_prob=1.0, max_instances=1,
                            scale_range=(1.0, 1.0), p_flip=0.0, seed=0)

    target_img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    target_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    assert int((target_mask == DUCT).sum()) == 0

    out_img, out_mask = cp(target_img, target_mask)

    pasted = out_mask == DUCT
    assert int(pasted.sum()) == 64                 # the 8x8 square
    assert np.all(out_img[pasted] == 200)          # the instance's pixels
    # The originals are untouched (apply copies).
    assert int((target_mask == DUCT).sum()) == 0


def test_copy_paste_preserves_other_labels():
    """Pixels outside the pasted structure keep their original label."""
    bank = _bank()
    cp = RareClassCopyPaste(bank, paste_prob=1.0, max_instances=1,
                            scale_range=(1.0, 1.0), p_flip=0.0, seed=1)
    target_img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    target_mask = np.full((SIZE, SIZE), 2, dtype=np.uint8)   # all gallbladder

    _, out_mask = cp(target_img, target_mask)

    assert np.all(out_mask[out_mask != DUCT] == 2)            # rest still class 2
    assert int((out_mask == DUCT).sum()) == 64


def test_paste_prob_zero_is_identity():
    """paste_prob=0 never pastes -> the frame is returned unchanged."""
    bank = _bank()
    cp = RareClassCopyPaste(bank, paste_prob=0.0, seed=0)
    img = np.full((SIZE, SIZE, 3), 7, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

    out_img, out_mask = cp(img, mask)

    assert np.array_equal(out_img, img)
    assert np.array_equal(out_mask, mask)


def test_copy_paste_is_reproducible_with_seed():
    """Same seed -> same plan -> identical paste."""
    bank = _bank()
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

    a = RareClassCopyPaste(bank, paste_prob=1.0, scale_range=(1.0, 1.0), seed=42)(img, mask)
    b = RareClassCopyPaste(bank, paste_prob=1.0, scale_range=(1.0, 1.0), seed=42)(img, mask)

    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_one_plan_applies_consistently_across_a_clip():
    """A single sampled plan stamps the same coordinates on every frame.

    This is what keeps a temporal clip's paste spatially consistent (the
    CholecSeg8kWindowDataset path samples one plan per clip).
    """
    bank = _bank()
    cp = RareClassCopyPaste(bank, paste_prob=1.0, max_instances=1,
                            scale_range=(1.0, 1.0), p_flip=0.0, seed=3)
    plan = cp.sample_plan(SIZE, SIZE)
    assert plan  # non-empty

    frame_a = (np.zeros((SIZE, SIZE, 3), np.uint8), np.zeros((SIZE, SIZE), np.uint8))
    frame_b = (np.full((SIZE, SIZE, 3), 5, np.uint8), np.ones((SIZE, SIZE), np.uint8))
    _, mask_a = cp.apply(*frame_a, plan)
    _, mask_b = cp.apply(*frame_b, plan)

    # Same duct footprint in both frames regardless of their content.
    assert np.array_equal(mask_a == DUCT, mask_b == DUCT)


def test_instances_from_dataset_uses_load_raw():
    """instances_from_dataset harvests via a dataset's _load_raw(i)."""
    image, mask, area = _source_frame()

    class _FakeDataset:
        def __len__(self):
            return 3
        def _load_raw(self, i):
            return image, mask

    bank = instances_from_dataset(_FakeDataset(), [DUCT], margin=0.0, min_pixels=4)
    assert len(bank) == 3
    assert all(int((inst.mask == DUCT).sum()) == area for inst in bank)
