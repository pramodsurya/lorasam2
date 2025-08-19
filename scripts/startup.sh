#!/usr/bin/env bash
set -euo pipefail

# Startup script for Vast container: reads env, syncs data via rclone, runs training/inference, syncs outputs, and exits.
# Expects rclone config provided via RCLONE_CONFIG_B64 or pre-mounted config.

log() { echo "[startup] $*"; }
err() { echo "[startup][ERROR] $*" >&2; }

# Required env
: "${DATA_REMOTE:?DATA_REMOTE is required (e.g., s3:mybucket/datasets/topo)}"
: "${OUTPUT_REMOTE:?OUTPUT_REMOTE is required (e.g., s3:mybucket/outputs/lorasam2)}"

# Optional secrets via base64 rclone config (avoid printing)
if [[ -n "${RCLONE_CONFIG_B64:-}" ]]; then
  mkdir -p /root/.config/rclone
  echo "$RCLONE_CONFIG_B64" | base64 -d > /root/.config/rclone/rclone.conf
  chmod 600 /root/.config/rclone/rclone.conf
  log "rclone config written"
fi

WORKDIR=/workspace/lorasam2
cd "$WORKDIR"

# Fail-fast checks
if ! command -v rclone >/dev/null 2>&1; then
  err "rclone not found in image"
  exit 2
fi

# Prepare data dirs
mkdir -p data/images data/masks outputs/sam_lora outputs/preds weights logs

# Sync dataset in (images + masks) — assume remotes have subfolders images/, masks/
log "Syncing dataset from $DATA_REMOTE ..."
if ! rclone sync "$DATA_REMOTE/images" data/images --fast-list; then
  err "Failed to sync images from $DATA_REMOTE/images"
  exit 3
fi
if ! rclone sync "$DATA_REMOTE/masks" data/masks --fast-list; then
  err "Failed to sync masks from $DATA_REMOTE/masks"
  exit 3
fi

# SAM checkpoint: either provided via URL or already present in weights/
if [[ -n "${SAM_CHECKPOINT_URL:-}" ]]; then
  log "Fetching SAM checkpoint from URL"
  fname="weights/sam_vit_b_01ec64.pth"
  if ! curl -fsSL "$SAM_CHECKPOINT_URL" -o "$fname"; then
    err "Failed to download SAM checkpoint from $SAM_CHECKPOINT_URL"
    exit 4
  fi
fi
if [[ ! -f weights/sam_vit_b_01ec64.pth ]]; then
  err "SAM checkpoint weights/sam_vit_b_01ec64.pth missing"
  exit 4
fi

# Compose a runtime config file from env
cat > configs/runtime.yaml <<YML
train:
  images_dir: ./data/images
  masks_dir: ./data/masks
  mask_suffix: _mask
  image_exts: [".png", ".jpg", ".jpeg"]
  mask_exts: [".png"]
  shuffle: true
  val_split: ${HOLDOUT_FRACTION:-0.1}

model:
  sam_variant: vit_b
  checkpoint: ./weights/sam_vit_b_01ec64.pth
  lora:
    rank: 8
    alpha: 16

train_params:
  epochs: ${EPOCHS:-20}
  batch_size: ${BATCH_SIZE:-1}
  lr: ${LR:-5e-5}
  weight_decay: 0.0
  save_every: 1
  out_dir: ./outputs/sam_lora
  num_workers: 4
  amp: true

inference:
  threshold: 0.5
  out_dir: ./outputs/preds
YML

# Round handling
RESUME_ARG=""
if [[ "${LORA_MODE:-round1}" == "round2" ]]; then
  if [[ -n "${RESUME_LORA_URL:-}" ]]; then
    log "Downloading resume LoRA from $RESUME_LORA_URL"
    if ! curl -fsSL "$RESUME_LORA_URL" -o outputs/sam_lora/round1.pt; then
      err "Failed to download resume LoRA from $RESUME_LORA_URL"
      exit 5
    fi
    RESUME_ARG="--resume outputs/sam_lora/round1.pt"
  fi
  # Lower LR for resume; override in runtime config
  python - <<'PY'
import yaml
p = 'configs/runtime.yaml'
cfg = yaml.safe_load(open(p))
cfg['train_params']['lr'] = float("${LR_RESUME:-2e-5}")
open(p,'w').write(yaml.dump(cfg))
print('Adjusted LR for round2 resume')
PY
fi

# Oversample newly corrected stems if provided (comma-separated stems)
if [[ -n "${OVERSAMPLE_STEMS:-}" ]]; then
  echo "$OVERSAMPLE_STEMS" > logs/oversample.txt
  log "Oversample list written: $OVERSAMPLE_STEMS (informational; integrate into sampler if needed)"
fi

# Train
log "Starting training..."
set +e
python train_lora_sam.py --config configs/runtime.yaml $RESUME_ARG | tee logs/train.log
STATUS=$?
set -e

# Always sync logs/checkpoints even on failure
log "Syncing outputs to $OUTPUT_REMOTE ..."
rclone copy outputs "$OUTPUT_REMOTE/outputs" --fast-list || true
rclone copy logs "$OUTPUT_REMOTE/logs" --fast-list || true

if [[ $STATUS -ne 0 ]]; then
  err "Training failed with status $STATUS (outputs/logs were synced)"
  exit $STATUS
fi

# Inference batch if requested (INFER_N>0)
if [[ ${INFER_N:-0} -gt 0 ]]; then
  log "Running inference on ${INFER_N} sheets"
  python scripts/predict_batch.py --images_dir ./data/images --config configs/runtime.yaml --checkpoint ./weights/sam_vit_b_01ec64.pth --lora outputs/sam_lora/best.pt --n ${INFER_N}
  rclone copy outputs/preds "$OUTPUT_REMOTE/preds" --fast-list || true
fi

log "All done."
