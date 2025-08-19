#!/usr/bin/env python3
import os
import random
import argparse

from infer import run_inference


ess = """
Usage:
  python scripts/predict_batch.py --images_dir ./Maps --config ./configs/config.yaml --checkpoint weights/sam_vit_b_01ec64.pth --lora outputs/sam_lora/best.pt --n 5
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./Maps")
    ap.add_argument("--config", default="./configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="base SAM checkpoint (pth)")
    ap.add_argument("--lora", default="./outputs/sam_lora/best.pt")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    exts = (".png", ".jpg", ".jpeg")
    paths = []
    for root, _, files in os.walk(args.images_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    random.shuffle(paths)
    sel = paths[: args.n]
    print(f"Predicting {len(sel)} images…")
    for p in sel:
        out = run_inference(p, args.config, args.checkpoint, out_path=None, threshold=0.5, lora_path=args.lora)
        print(" -", p, "->", out)

if __name__ == "__main__":
    main()
