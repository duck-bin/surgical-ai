"""Endoscapes2023 CVS frame-level loader.

Endoscapes2023 (201 videos, 11,090 CVS-annotated frames) is the primary CVS
benchmark. It requires PhysioNet credentialed access (czwq-jh81); data is
expected to be downloaded manually to ``./data/endoscapes2023/``.

Each annotated frame carries 3 binary Strasberg criteria; the loader also
exposes the official video-level train/val/test splits. Implemented in Step 6.
"""
from __future__ import annotations

from torch.utils.data import Dataset

CVS_CRITERIA: tuple[str, ...] = (
    "c1_two_structures",
    "c2_triangle_cleared",
    "c3_cystic_plate_exposed",
)


class Endoscapes2023Dataset(Dataset):
    """torch Dataset over CVS-annotated Endoscapes2023 frames.

    Implemented in Step 6. Expects the official directory layout described in
    Mascagni et al., Scientific Data 2024.  # TODO: verify layout.
    """

    def __init__(self, root: str = "./data/endoscapes2023", split: str = "train",
                 image_size: int = 512, transform=None):
        raise NotImplementedError("Endoscapes2023 loader is implemented in Step 6.")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in Step 6.")

    def __getitem__(self, index: int):
        raise NotImplementedError("Implemented in Step 6.")
