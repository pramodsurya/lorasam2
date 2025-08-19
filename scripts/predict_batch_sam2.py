#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import random

from src.config_types import Config
from src.fs_utils import list_images
from infer_sam2 import infer_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./Maps")
    ap.add_argument("--masks_dir", default="./mask")
    ap.add_argument("--out_dir", default="./outputs/preds")
    ap.add_argument("--lora", default="./outputs/sam2_lora/best.pt")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    cfg = Config(images_dir=args.images_dir, masks_dir=args.masks_dir, mask_suffix="_mask",
                 image_exts=[".png", ".jpg", ".jpeg"], mask_exts=[".png"])
    imgs = list_images(cfg.images_dir, cfg.image_exts)
    unlabeled = []
    for p in imgs:
        m = cfg.match_mask(p)
        if not (m and os.path.exists(m)):
            unlabeled.append(p)
    random.shuffle(unlabeled)
    pick = unlabeled[: max(0, args.n)]
    os.makedirs(args.out_dir, exist_ok=True)
    for ip in pick:
        stem = Path(ip).stem
        outp = os.path.join(args.out_dir, stem + "_mask.png")
        infer_image(ip, outp, args.lora)
        print("pred:", outp)


if __name__ == "__main__":
    main()
