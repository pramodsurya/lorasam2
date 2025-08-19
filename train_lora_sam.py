import os
import yaml
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config_types import Config
from src.dataset import MapMaskDataset
from src.fs_utils import list_images, split_train_val
from src.lora_inject import LoRAInjector


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_sam_model(variant: str, checkpoint: str):
    from segment_anything import sam_model_registry

    sam = sam_model_registry[f"{variant}"](checkpoint=checkpoint)
    return sam


def main(cfg_path: str = "./configs/config.yaml", resume: str | None = None):
    cfg_raw = load_cfg(cfg_path)
    data_cfg = cfg_raw["train"]
    model_cfg = cfg_raw["model"]
    trn_cfg = cfg_raw["train_params"]

    os.makedirs(trn_cfg["out_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    c = Config(
        images_dir=data_cfg["images_dir"],
        masks_dir=data_cfg["masks_dir"],
        mask_suffix=data_cfg.get("mask_suffix", "_mask"),
        image_exts=data_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]),
        mask_exts=data_cfg.get("mask_exts", [".png"]),
    )
    all_imgs = list_images(c.images_dir, c.image_exts)
    if not all_imgs:
        raise RuntimeError(f"No images found in {c.images_dir}")
    # Filter to images that have masks
    imgs_with_masks = []
    for p in all_imgs:
        m = c.match_mask(p)
        if m and os.path.exists(m):
            imgs_with_masks.append(p)
    if not imgs_with_masks:
        raise RuntimeError("No images with matching masks were found. Ensure masks are named <stem>_mask.png in ./mask.")
    train_paths, val_paths = split_train_val(imgs_with_masks, data_cfg.get("val_split", 0.1))

    train_ds = MapMaskDataset(c, train_paths)
    val_ds = MapMaskDataset(c, val_paths) if val_paths else None

    dl_train = DataLoader(train_ds, batch_size=trn_cfg.get("batch_size", 1), shuffle=True, num_workers=trn_cfg.get("num_workers", 2))
    dl_val = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=trn_cfg.get("num_workers", 2)) if val_ds else None

    # Model
    sam = build_sam_model(model_cfg["sam_variant"], model_cfg["checkpoint"])  # type: ignore
    sam.to(device)

    # Inject LoRA
    injector = LoRAInjector(rank=model_cfg["lora"]["rank"], alpha=model_cfg["lora"]["alpha"], target_modules=model_cfg["lora"].get("target_modules"))
    injector.apply(sam)
    injector.mark_only_lora_trainable(sam)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=trn_cfg.get("lr", 5e-5), weight_decay=trn_cfg.get("weight_decay", 0.0))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(trn_cfg.get("amp", True)))

    # Optionally resume LoRA weights
    if resume and os.path.isfile(resume):
        state = torch.load(resume, map_location="cpu")
        lora_sd = state.get("lora", state)
        LoRAInjector.load_lora_state_dict(sam, lora_sd)
        print(f"Resumed LoRA from {resume}")

    # Simple BCE loss on predicted mask logits. We use SAM mask decoder; for supervised training we can pass image embeddings and forego prompts.
    from segment_anything.modeling import Sam
    from segment_anything.utils.transforms import ResizeLongestSide

    assert isinstance(sam, Sam)

    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    resizer = ResizeLongestSide(1024)

    def preprocess(imgs):
        # imgs in [0,1]
        B, C, H, W = imgs.shape
        target_size = 1024
        outs = []
        infos = []
        for i in range(B):
            im = imgs[i].detach().cpu().permute(1, 2, 0).numpy()
            im_resized = resizer.apply_image(im)
            t = torch.from_numpy(im_resized).permute(2, 0, 1).float().to(device)
            new_h, new_w = t.shape[-2:]
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0)
            t = (t * 255.0 - pixel_mean) / pixel_std
            outs.append(t)
            infos.append(((H, W), (new_h, new_w)))
        x = torch.stack(outs, dim=0)
        return x, infos

    def postprocess(masks, infos):
        outs = []
        for i, m in enumerate(masks):
            orig_hw, new_hw = infos[i]
            m = m[..., : new_hw[0], : new_hw[1]]
            m = torch.nn.functional.interpolate(m, size=orig_hw, mode="bilinear", align_corners=False)
            outs.append(m)
        return torch.cat(outs, dim=0)

    def forward_mask(model: Sam, imgs):
        B, C, H, W = imgs.shape
        x, infos = preprocess(imgs)
        image_embeddings = model.image_encoder(x)
        # prompts: none
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=None, masks=None)
        image_pe = model.prompt_encoder.get_dense_pe()
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = torch.nn.functional.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks = postprocess(masks, infos)
        return masks

    bce = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    for epoch in range(1, trn_cfg.get("epochs", 20) + 1):
        sam.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch} train", ncols=100)
        total_loss = 0.0
        for batch in pbar:
            imgs = batch["image"].to(device)
            gts = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = forward_mask(sam, imgs)
                loss = bce(logits, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train = total_loss / max(1, len(train_ds))

        # save checkpoint each epoch
        save_path = os.path.join(trn_cfg["out_dir"], f"epoch_{epoch:03d}.pt")
        lora_only = LoRAInjector.lora_state_dict(sam)
        torch.save({
            "epoch": epoch,
            "lora": lora_only,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_loss": avg_train,
        }, save_path)

        if dl_val is not None:
            sam.eval()
            vloss = 0.0
            with torch.no_grad():
                for vb in tqdm(dl_val, desc="val", ncols=100):
                    vi = vb["image"].to(device)
                    vg = vb["mask"].to(device)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        vlogits = forward_mask(sam, vi)
                        vl = bce(vlogits, vg)
                    vloss += vl.item() * vi.size(0)
            vavg = vloss / max(1, len(val_ds))
            if vavg < best_val:
                best_val = vavg
                torch.save({
                    "epoch": epoch,
                    "lora": LoRAInjector.lora_state_dict(sam),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_loss": avg_train,
                    "val_loss": vavg,
                }, os.path.join(trn_cfg["out_dir"], "best.pt"))

    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.resume)
