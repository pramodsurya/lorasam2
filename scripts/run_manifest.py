#!/usr/bin/env python3
import argparse
import sys, os as _os
_repo_root = _os.path.dirname(_os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import random
import json
from pathlib import Path

from src.config_types import Config
from src.fs_utils import list_images


def build_manifest(images_dir: str, masks_dir: str, train_k: int, predict_k: int,
                   model_name: str, config_path: str, epochs: int, batch: int, lr: float,
                   gpu_filter: str, min_vram: int, max_price: float, will_auto_fetch_ckpt: bool):
    cfg = Config(images_dir=images_dir, masks_dir=masks_dir, mask_suffix="_mask",
                 image_exts=[".png", ".jpg", ".jpeg"], mask_exts=[".png"])
    imgs = list_images(cfg.images_dir, cfg.image_exts)
    labeled, unlabeled = [], []
    for p in imgs:
        m = cfg.match_mask(p)
        if m and _os.path.exists(m):
            labeled.append(Path(p).stem)
        else:
            unlabeled.append(Path(p).stem)
    random.seed(42)
    random.shuffle(labeled)
    random.shuffle(unlabeled)
    train_stems = labeled[:max(0, train_k)]
    predict_stems = unlabeled[:max(0, predict_k)]
    mani = {
        "dataset": {
            "images": len(imgs),
            "labeled": len(labeled),
            "unlabeled": len(unlabeled),
            "train_stems": train_stems,
            "predict_stems": predict_stems,
        },
        "model": {
            "name": model_name,
            "config": config_path,
            "auto_fetch_checkpoint": will_auto_fetch_ckpt,
        },
        "hyperparams": {
            "epochs": epochs,
            "batch": batch,
            "lr": lr,
        },
        "gpu_filters": {
            "allowed": gpu_filter,
            "min_vram_gb": min_vram,
            "max_price_per_hour": max_price,
        }
    }
    return mani


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./Maps")
    ap.add_argument("--masks_dir", default="./mask")
    ap.add_argument("--train_k", type=int, default=5)
    ap.add_argument("--predict_k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", default="A100|A10|L40S")
    ap.add_argument("--min_vram", type=int, default=24)
    ap.add_argument("--max_price", type=float, default=1.20)
    ap.add_argument("--model_name", default="SAM 2.1 (hiera-large)")
    ap.add_argument("--config_path", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--manifest_out", default="./logs/run_manifest.json")
    args = ap.parse_args()

    mani = build_manifest(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        train_k=args.train_k,
        predict_k=args.predict_k,
        model_name=args.model_name,
        config_path=args.config_path,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        gpu_filter=args.gpu,
        min_vram=args.min_vram,
        max_price=args.max_price,
        will_auto_fetch_ckpt=True,
    )
    Path(_os.path.dirname(args.manifest_out)).mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w") as f:
        json.dump(mani, f, indent=2)
    print(json.dumps(mani, indent=2))


if __name__ == "__main__":
    main()
