import os
import argparse
import cv2
import torch
import numpy as np

from src.lora_inject import LoRAInjector
from src.sam2_utils import ensure_sam2_assets


def build_sam2(ckpt: str, cfg: str):
    try:
        from sam2.build_sam import build_sam2
    except Exception:
        os.system("pip install 'git+https://github.com/facebookresearch/sam2.git' --quiet")
        from sam2.build_sam import build_sam2
    return build_sam2(config=cfg, ckpt=ckpt, device="cuda" if torch.cuda.is_available() else "cpu")


def preprocess(img: np.ndarray, device):
    from sam2.utils.transforms import ResizeLongestSide
    resizer = ResizeLongestSide(1024)
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    imr = resizer.apply_image(img)
    t = torch.from_numpy(imr).permute(2, 0, 1).float().to(device)
    nh, nw = t.shape[-2:]
    ph, pw = 1024 - nh, 1024 - nw
    t = torch.nn.functional.pad(t, (0, pw, 0, ph), value=0)
    t = (t * 255.0 - pixel_mean) / pixel_std
    return t.unsqueeze(0), (img.shape[:2], (nh, nw))


def infer_image(img_path: str, out_path: str, lora_path: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, cfg = ensure_sam2_assets()
    model = build_sam2(ckpt, cfg)
    LoRAInjector.mark_only_lora_trainable(model)
    if lora_path and os.path.isfile(lora_path):
        state = torch.load(lora_path, map_location="cpu")
        lora_sd = state.get("lora", state)
        LoRAInjector.load_lora_state_dict(model, lora_sd)
    model.to(device).eval()

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x, info = preprocess(rgb, device)

    with torch.no_grad(), torch.cuda.amp.autocast(True):
        emb = model.image_encoder(x)
        image_pe = model.prompt_encoder.get_dense_pe()
        sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
        low, _ = model.mask_decoder(image_embeddings=emb, image_pe=image_pe, sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        low = torch.nn.functional.interpolate(low, size=(1024, 1024), mode="bilinear", align_corners=False)
        (H0, W0), (nh, nw) = info
        low = low[..., :nh, :nw]
        logits = torch.nn.functional.interpolate(low, size=(H0, W0), mode="bilinear", align_corners=False)

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, mask)

    # Also save overlay visualization
    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0].replace("_mask", "")
    overlay_path = os.path.join(overlay_dir, f"{stem}_overlay.png")
    color = (0, 255, 255)  # yellow/cyan blend
    alpha = 0.4
    m3 = np.repeat((mask > 0).astype(np.uint8)[..., None], 3, axis=2)
    blended = bgr.copy()
    blended[m3.astype(bool)] = (blended[m3.astype(bool)] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cv2.imwrite(overlay_path, blended)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()
    infer_image(args.image, args.out, args.lora)


if __name__ == "__main__":
    main()
