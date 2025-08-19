import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config_types import Config
from src.dataset import MapMaskDataset
from src.fs_utils import list_images, split_train_val
from src.lora_inject import LoRAInjector
from src.sam2_utils import ensure_sam2_assets


def build_sam2(ckpt: str, cfg: str):
    # Install/import SAM2 at runtime
    try:
        from sam2.build_sam import build_sam2
    except Exception:
        os.system("pip install 'git+https://github.com/facebookresearch/sam2.git' --quiet")
        from sam2.build_sam import build_sam2

    model = build_sam2(config=cfg, ckpt=ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    return model


def forward_prompt_free_sam2(model, imgs: torch.Tensor):
    """Prompt-free forward for SAM2 mask decoder. Returns mask logits resized to input size."""
    device = imgs.device
    B, C, H, W = imgs.shape
    # SAM2 expects uint8 [H,W,3] in its encode_image convenience, but we can mirror preprocessing
    # Use model.preprocess from sam2.utils (available in repo)
    from sam2.utils.transforms import ResizeLongestSide

    resizer = ResizeLongestSide(model.image_encoder.img_size if hasattr(model, 'image_encoder') else 1024)
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

    target = 1024
    ims = []
    infos = []
    for i in range(B):
        im = imgs[i].detach().cpu().permute(1, 2, 0).numpy()
        imr = resizer.apply_image(im)
        t = torch.from_numpy(imr).permute(2, 0, 1).float().to(device)
        nh, nw = t.shape[-2:]
        ph, pw = target - nh, target - nw
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), value=0)
        t = (t * 255.0 - pixel_mean) / pixel_std
        ims.append(t)
        infos.append(((H, W), (nh, nw)))
    x = torch.stack(ims, 0)

    with torch.no_grad():
        image_embeddings = model.image_encoder(x)
        image_pe = model.prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    masks = torch.nn.functional.interpolate(low_res_masks, size=(target, target), mode="bilinear", align_corners=False)
    outs = []
    for i, m in enumerate(masks):
        (H0, W0), (nh, nw) = infos[i]
        m = m[..., :nh, :nw]
        m = torch.nn.functional.interpolate(m.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False).squeeze(0)
        outs.append(m)
    return torch.cat(outs, dim=0)


def main(images_dir: str = "./Maps", masks_dir: str = "./mask", out_dir: str = "./outputs/sam2_lora",
         epochs: int = 12, batch_size: int = 2, lr: float = 1e-4, resume: str | None = None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    c = Config(images_dir=images_dir, masks_dir=masks_dir, mask_suffix="_mask",
               image_exts=[".png", ".jpg", ".jpeg"], mask_exts=[".png"])
    all_imgs = list_images(c.images_dir, c.image_exts)
    imgs_with_masks: List[str] = []
    for p in all_imgs:
        m = c.match_mask(p)
        if m and os.path.exists(m):
            imgs_with_masks.append(p)
    if len(imgs_with_masks) == 0:
        raise RuntimeError("No images with matching masks found under repo.")

    train_paths, val_paths = split_train_val(imgs_with_masks, 0.1)
    train_ds = MapMaskDataset(c, train_paths)
    val_ds = MapMaskDataset(c, val_paths) if val_paths else None
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2) if val_ds else None

    # Model + LoRA
    ckpt, cfg = ensure_sam2_assets()
    model = build_sam2(ckpt=ckpt, cfg=cfg)
    model.to(device)
    injector = LoRAInjector(rank=8, alpha=16)
    injector.apply(model)
    injector.mark_only_lora_trainable(model)

    if resume and os.path.isfile(resume):
        state = torch.load(resume, map_location="cpu")
        lora_sd = state.get("lora", state)
        LoRAInjector.load_lora_state_dict(model, lora_sd)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    bce = torch.nn.BCEWithLogitsLoss()

    best = 1e9
    best_written = False
    for epoch in range(1, epochs + 1):
        model.train()
        tbar = tqdm(dl_train, desc=f"E{epoch} train", ncols=100)
        running = 0.0
        for batch in tbar:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):
                logits = forward_prompt_free_sam2(model, imgs)
                loss = bce(logits, gts)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * imgs.size(0)
            tbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg = running / max(1, len(train_ds))
        torch.save({
            "epoch": epoch,
            "lora": LoRAInjector.lora_state_dict(model),
            "train_loss": avg,
        }, os.path.join(out_dir, f"epoch_{epoch:03d}.pt"))

        if dl_val is not None:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for vb in dl_val:
                    vi = vb["image"].to(device)
                    vg = vb["mask"].to(device)
                    with torch.cuda.amp.autocast(True):
                        vlogits = forward_prompt_free_sam2(model, vi)
                        l = bce(vlogits, vg)
                    vl += l.item() * vi.size(0)
            vavg = vl / max(1, len(val_ds))
            if vavg < best:
                best = vavg
                torch.save({
                    "epoch": epoch,
                    "lora": LoRAInjector.lora_state_dict(model),
                    "val_loss": vavg,
                }, os.path.join(out_dir, "best.pt"))
                best_written = True

    # If no validation set, ensure best.pt exists by copying the last checkpoint
    if not best_written:
        last_ckpt = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
        if os.path.isfile(last_ckpt):
            import shutil
            shutil.copy(last_ckpt, os.path.join(out_dir, "best.pt"))

    print("Training complete (SAM2).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="./Maps")
    ap.add_argument("--masks_dir", default="./mask")
    ap.add_argument("--out_dir", default="./outputs/sam2_lora")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    main(images_dir=args.images_dir, masks_dir=args.masks_dir, out_dir=args.out_dir,
         epochs=args.epochs, batch_size=args.batch, lr=args.lr, resume=args.resume)
