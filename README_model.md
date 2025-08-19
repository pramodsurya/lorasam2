# lorasam2: Fine-tune SAM with LoRA for Historical Map Features

This repo fine-tunes Segment Anything (SAM) with LoRA on your historical maps.

Data
- Images: `./Maps`
- Masks: `./mask` with filenames like `<stem>_mask.png` or same `<stem>.png` under `mask/`.

Quick start (local GPU)
- Put SAM ViT-B weight at `weights/sam_vit_b_01ec64.pth`.
- Install deps: `pip install -r requirements.txt`.
- Train: `python train_lora_sam.py --config configs/config.yaml`.
- Infer: `python infer.py --image Maps/E44G15_56J15.png --config configs/config.yaml --checkpoint outputs/sam_lora/best.pt`.

Vast.ai
There are two flows: prebuilt image or upload/install.

Prebuilt image (recommended)
- Build and push image: see `docker/Dockerfile` and `docker/build_and_push.sh` (requires docker login to your registry).
- Launch with one command using the Vast launcher (no web UI):
	- Set envs (no secrets in files):
		- VASTAI_API_KEY: your Vast key (env only)
		- RCLONE_CONFIG_B64: base64 of your rclone config (optional)
	- Run:
		- `./scripts/vast_launcher.sh --image <your-registry>/lorasam2:latest --gpu "A100|A10|L40S" --min-vram 24 --max-price 1.50 --data-remote <remote>/datasets/topo --out-remote <remote>/outputs/lorasam2 --sam-url <url-to-sam-weight> --mode round1 --epochs 20 --batch 1 --infer-n 5`
- The instance will pull data, train, sync outputs, run 5-image inference for QA, and exit. Use `scripts/vast_stop.sh <id> [--destroy]` to stop/destroy.

Upload/install flow
- `python scripts/vast_automate.py --gpu A100 --weights weights/sam_vit_b_01ec64.pth`
- This uploads the repo, installs deps, trains, predicts 5, and downloads outputs.

Active learning loop
1) Train on current masks.
2) Run inference on 5 new maps, get predictions in `outputs/preds`.
3) Manually correct masks and place into `mask/` with `<stem>_mask.png`.
4) Re-run training; the script will pick up the new masks automatically.

Dataset layout and tiling
- Use rclone-style remotes for data sync. Expected remote structure:
	- `<remote>/datasets/topo/images/`
	- `<remote>/datasets/topo/masks/`
- Masks are binary and name-matched to images using `<stem>_mask.png`.
- Large sheets should be tiled to ~1024 tiles with stride ~768 (set via env TILE_SIZE/TILE_STRIDE). Keep 1–2 whole sheets as holdout.

Label policy (single class: Water bodies)
- Include: perennial blue water (fills, lines, canals), dry/seasonal tanks (white polygons with black dots), streams/rivers.
- Exclude: non-water symbols and unrelated map artifacts.

Rounds and hyperparameters
- Round 1: 5 labeled maps → train from scratch. Save LoRA (round1).
- Round 2: Add 5 corrected predictions (total 10). Resume from round1, lower LR, optionally oversample new 5.

Environment variables (startup)
- DATA_REMOTE, OUTPUT_REMOTE: rclone remotes for data in/out.
- RCLONE_CONFIG_B64: optional base64 rclone config (avoid secrets in files).
- SAM_CHECKPOINT_URL: HTTP(S) URL for base SAM checkpoint.
- RESUME_LORA_URL: optional URL for round1 LoRA when running round2.
- LORA_MODE: round1 or round2.
- EPOCHS, BATCH_SIZE, TILE_SIZE, TILE_STRIDE, LR, LR_RESUME, INFER_N.

Stopping / cleanup
- Stop or destroy the instance to avoid cost:
	- `./scripts/vast_stop.sh <instance_id>` (stop)
	- `./scripts/vast_stop.sh <instance_id> --destroy` (destroy)

Common failures
- No offers found: adjust `--gpu`, `--min-vram`, or `--max-price`.
- Remote sync failed: verify DATA_REMOTE/OUTPUT_REMOTE and rclone config.
- SAM missing: provide `--sam-url` or bake the weight in your image.
