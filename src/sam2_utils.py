import os
from pathlib import Path
from typing import Tuple

SAM2_21_CKPT = "sam2.1_hiera_large.pt"
SAM2_21_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

SAM2_21_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/checkpoints/sam2.1_hiera_large.pt"
SAM2_21_CFG_URL = "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_l.yaml"


def ensure_sam2_assets(weights_dir: str = "weights") -> Tuple[str, str]:
    """
    Ensure SAM2.1 checkpoint and config exist locally under repo. Downloads from official URLs if missing.
    Returns tuple of (ckpt_path, cfg_path).
    """
    wdir = Path(weights_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = wdir / SAM2_21_CKPT
    cfg_path = Path(SAM2_21_CFG)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        import urllib.request
        urllib.request.urlretrieve(SAM2_21_CKPT_URL, str(ckpt_path))
    if not cfg_path.exists():
        import urllib.request
        urllib.request.urlretrieve(SAM2_21_CFG_URL, str(cfg_path))

    return str(ckpt_path), str(cfg_path)
