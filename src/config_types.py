import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    images_dir: str
    masks_dir: str
    mask_suffix: str = "_mask"
    image_exts: List[str] = None
    mask_exts: List[str] = None

    def __post_init__(self):
        if self.image_exts is None:
            self.image_exts = [".png", ".jpg", ".jpeg"]
        if self.mask_exts is None:
            self.mask_exts = [".png"]

    def match_mask(self, image_path: str) -> str:
        stem, _ = os.path.splitext(os.path.basename(image_path))
        # Prefer <stem>_mask.* in masks_dir
        for me in self.mask_exts:
            m = os.path.join(self.masks_dir, f"{stem}{self.mask_suffix}{me}")
            if os.path.exists(m):
                return m
        # Fallback: identical filename under masks_dir
        for ie in self.image_exts:
            for me in self.mask_exts:
                candidate = os.path.join(self.masks_dir, f"{stem}{me}")
                if os.path.exists(candidate):
                    return candidate
        return ""
