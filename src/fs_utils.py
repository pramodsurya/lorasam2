import os
from typing import List, Tuple


def list_images(root: str, exts: List[str]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            lf = f.lower()
            if any(lf.endswith(e) for e in exts):
                out.append(os.path.join(dirpath, f))
    out.sort()
    return out


def split_train_val(paths: List[str], val_ratio: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    import random

    random.Random(seed).shuffle(paths)
    n_val = int(len(paths) * val_ratio)
    val = paths[:n_val]
    train = paths[n_val:]
    return train, val
