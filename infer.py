import os
import yaml
import torch
import numpy as np
from PIL import Image
from typing import Optional


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_sam_model(variant: str, checkpoint: str):
    from segment_anything import sam_model_registry

    sam = sam_model_registry[f"{variant}"](checkpoint=checkpoint)
    return sam


def preprocess_image(img: Image.Image, size: int = 1024):
    import torch.nn.functional as F

    img = img.convert("RGB")
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    B, C = 1, 3
    _, H, W = t.shape
    scale = size / max(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    imgs_resized = torch.nn.functional.interpolate(t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = size - new_h
    pad_w = size - new_w
    imgs_padded = torch.nn.functional.pad(imgs_resized, (0, pad_w, 0, pad_h), value=0)

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255.0
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1) / 255.0
    x = (imgs_padded - pixel_mean) / pixel_std
    return x, (H, W), (new_h, new_w)


def run_inference(image_path: str, cfg_path: str, checkpoint_path: Optional[str], out_path: Optional[str] = None, threshold: float = 0.5, lora_path: Optional[str] = None):
    cfg = load_cfg(cfg_path)
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})
    if checkpoint_path is None:
        checkpoint_path = model_cfg["checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = build_sam_model(model_cfg["sam_variant"], checkpoint_path)
    sam.to(device).eval()
    # Optionally load LoRA
    if lora_path and os.path.isfile(lora_path):
        state = torch.load(lora_path, map_location="cpu")
        lora_state = state.get("lora", state)
        # Lazy import to avoid circular
        from src.lora_inject import LoRAInjector
        LoRAInjector.load_lora_state_dict(sam, lora_state)

    img = Image.open(image_path)
    x, orig_hw, new_hw = preprocess_image(img, size=1024)
    x = x.to(device)

    with torch.no_grad():
        emb = sam.image_encoder(x)
        sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
        pe = sam.prompt_encoder.get_dense_pe()
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        masks = torch.nn.functional.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks = masks[..., : new_hw[0], : new_hw[1]]
        masks = torch.nn.functional.interpolate(masks, size=orig_hw, mode="bilinear", align_corners=False)
        prob = masks.sigmoid()[0, 0].cpu().numpy()
    bin_mask = (prob >= float(threshold)).astype(np.uint8) * 255

    if out_path is None:
        os.makedirs(cfg.get("inference", {}).get("out_dir", "./outputs/preds"), exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(cfg.get("inference", {}).get("out_dir", "./outputs/preds"), f"{stem}_pred.png")

    Image.fromarray(bin_mask).save(out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="./configs/config.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    path = run_inference(args.image, args.config, args.checkpoint, args.out, args.threshold, args.lora)
    print(f"Saved: {path}")
