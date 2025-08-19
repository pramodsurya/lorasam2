#!/usr/bin/env python3
"""
Sanity check Maps/mask dataset and print image-mask pairs, missing masks, and a small sample.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config_types import Config
from src.fs_utils import list_images

CFG = Config(images_dir="./Maps", masks_dir="./mask")

imgs = list_images(CFG.images_dir, CFG.image_exts)
print(f"Found {len(imgs)} images.")

missing = []
paired = []
for p in imgs:
    m = CFG.match_mask(p)
    if not m or not os.path.exists(m):
        missing.append(p)
    else:
        paired.append((p, m))

print(f"Paired: {len(paired)} | Missing masks: {len(missing)}")
if missing:
    print("Example missing:")
    for x in missing[:10]:
        print(" -", x)
else:
    print("All images have masks.")

print("Sample pairs:")
for a, b in paired[:5]:
    print(" -", a, "->", b)
