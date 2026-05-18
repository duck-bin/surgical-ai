"""CholecSeg8k dataset loader and the 13 -> 6 class mapping.

CholecSeg8k contains 8,080 frames drawn from 17 Cholec80 video clips, each
annotated with 13 semantic classes. We merge them into 6 CVS-relevant target
classes.

Original 13 CholecSeg8k classes:
     1  Black Background          8  Blood
     2  Abdominal Wall            9  Cystic Duct
     3  Liver                    10  L-hook Electrocautery
     4  Gastrointestinal Tract   11  Gallbladder
     5  Fat                      12  Hepatic Vein
     6  Grasper                  13  Liver Ligament
     7  Connective Tissue

6-class target schema (index -> name):
     0  background      3  cystic_duct
     1  liver           4  cystic_artery
     2  gallbladder     5  tool

13 -> 6 mapping (decision: strict 6 classes, no fabricated labels):
    background    <- Black Background, Abdominal Wall, Gastrointestinal Tract,
                     Fat, Connective Tissue, Blood
    liver         <- Liver, Hepatic Vein, Liver Ligament
    gallbladder   <- Gallbladder
    cystic_duct   <- Cystic Duct
    cystic_artery <- (NONE) CholecSeg8k has no cystic-artery label. This class
                     receives 0 pixels from CholecSeg8k and is learned instead
                     from Endoscapes2023, which does annotate the cystic artery.
                     # TODO: verify mapping against the CholecSeg8k paper.
    tool          <- Grasper, L-hook Electrocautery

Splits are video-level (12 train / 2 val / 3 test of 17 videos) to prevent
frame leakage. Loader logic is implemented in Step 2.
"""
from __future__ import annotations

from torch.utils.data import Dataset

CLASS_NAMES: tuple[str, ...] = (
    "background",
    "liver",
    "gallbladder",
    "cystic_duct",
    "cystic_artery",
    "tool",
)

# Original CholecSeg8k label name -> target class index (see module docstring).
# Populated and verified against the dataset encoding in Step 2.
CHOLECSEG8K_13_TO_6: dict[str, int] = {}


class CholecSeg8kDataset(Dataset):
    """torch Dataset over CholecSeg8k with the 13 -> 6 class remapping.

    Implemented in Step 2.
    """

    def __init__(self, split: str = "train", image_size: int = 512, transform=None):
        raise NotImplementedError("CholecSeg8k loader is implemented in Step 2.")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in Step 2.")

    def __getitem__(self, index: int):
        raise NotImplementedError("Implemented in Step 2.")


def load_cholecseg8k(hf_repo: str = "minwoosun/CholecSeg8k"):
    """Load CholecSeg8k via the HuggingFace ``datasets`` loader. Step 2."""
    raise NotImplementedError("Implemented in Step 2.")
