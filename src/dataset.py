import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config_types import Config


def _load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # HWC -> CHW, normalize to [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _load_mask(path: str) -> torch.Tensor:
    m = Image.open(path).convert("L")
    arr = np.array(m)
    # binarize: >0 -> 1
    binm = (arr > 0).astype(np.uint8)
    t = torch.from_numpy(binm).unsqueeze(0).float()
    return t


class MapMaskDataset(Dataset):
    def __init__(self, cfg: Config, paths: list[str]):
        self.cfg = cfg
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        mask_path = self.cfg.match_mask(img_path)
        if not mask_path or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {img_path}")
        image = _load_image(img_path)
        mask = _load_mask(mask_path)
        sample = {
            "image_path": img_path,
            "mask_path": mask_path,
            "image": image,
            "mask": mask,
        }
        return sample
