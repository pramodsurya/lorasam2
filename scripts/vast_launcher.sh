#!/usr/bin/env bash
set -euo pipefail

# Vast non-interactive launcher: searches offers with constraints and creates an instance
# with your prebuilt Docker image and a non-interactive on-start that runs startup.sh.

usage() {
  cat <<USAGE
Usage: 
  VASTAI_API_KEY=*** ./scripts/vast_launcher.sh \
    --image ghcr.io/youruser/lorasam2:latest \
    --gpu "A100|A10|L40S" \
    --min-vram 24 \
    --max-price 1.50 \
    --data-remote s3:bucket/datasets/topo \
    --out-remote s3:bucket/outputs/lorasam2 \
    [--rclone-config-b64 <base64>] \
    [--sam-url https://.../sam_vit_b_01ec64.pth] \
    [--resume-url https://.../round1.pt] \
    [--mode round1|round2] \
    [--epochs 20] [--batch 1] [--tile 1024] [--stride 768] [--lr 5e-5] [--lr-resume 2e-5] \
    [--infer-n 5]
USAGE
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

VASTAI_API_KEY=${VASTAI_API_KEY:-}
[[ -z "$VASTAI_API_KEY" ]] && { echo "Set VASTAI_API_KEY in env" >&2; exit 2; }

IMAGE=""
GPU_FILTER="A100|A10|L40S"
MIN_VRAM=24
MAX_PRICE=1.50
DATA_REMOTE=""
OUT_REMOTE=""
RCLONE_CONFIG_B64=""
SAM_URL=""
RESUME_URL=""
MODE="round1"
EPOCHS=20
BATCH=1
TILE=1024
STRIDE=768
LR=5e-5
LR_RESUME=2e-5
INFER_N=5

while [[ $# -gt 0 ]]; do
  case $1 in
    --image) IMAGE=$2; shift 2;;
    --gpu) GPU_FILTER=$2; shift 2;;
    --min-vram) MIN_VRAM=$2; shift 2;;
    --max-price) MAX_PRICE=$2; shift 2;;
    --data-remote) DATA_REMOTE=$2; shift 2;;
    --out-remote) OUT_REMOTE=$2; shift 2;;
    --rclone-config-b64) RCLONE_CONFIG_B64=$2; shift 2;;
    --sam-url) SAM_URL=$2; shift 2;;
    --resume-url) RESUME_URL=$2; shift 2;;
    --mode) MODE=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --tile) TILE=$2; shift 2;;
    --stride) STRIDE=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    --lr-resume) LR_RESUME=$2; shift 2;;
    --infer-n) INFER_N=$2; shift 2;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

[[ -z "$IMAGE" ]] && { echo "--image is required" >&2; exit 2; }
[[ -z "$DATA_REMOTE" ]] && { echo "--data-remote is required" >&2; exit 2; }
[[ -z "$OUT_REMOTE" ]] && { echo "--out-remote is required" >&2; exit 2; }

# Build Vast offer query per docs
QUERY="reliability>0.95 rentable=true verified=true gpu_ram>=$MIN_VRAM dph<=$MAX_PRICE"
QUERY="$QUERY gpu_name~$GPU_FILTER"

echo "Searching offers with: $QUERY" >&2
OFFERS=$(vastai --api-key "$VASTAI_API_KEY" search offers "$QUERY" --raw || true)

if [[ -z "$OFFERS" || "$OFFERS" == "[]" || "$OFFERS" == "{}" ]]; then
  echo "No offers found. Active filters: $QUERY" >&2
  exit 3
fi

# Pick cheapest by dph
OFFER_ID=$(echo "$OFFERS" | python - <<'PY'
import json,sys
data=json.load(sys.stdin)
offers=data.get('offers', data)
offers=[o for o in offers if 'dph' in o]
offers.sort(key=lambda o: float(o['dph']))
print(offers[0]['id'])
PY
)

[[ -z "$OFFER_ID" ]] && { echo "Failed to select offer" >&2; exit 3; }

echo "Selected offer: $OFFER_ID" >&2

# Build onstart script: export envs and run startup.sh non-interactively
ONSTART_CMD=$(cat <<EOS
set -e
mkdir -p /workspace/lorasam2
cd /workspace/lorasam2 || true
export LORA_MODE="$MODE"
export DATA_REMOTE="$DATA_REMOTE"
export OUTPUT_REMOTE="$OUT_REMOTE"
export RCLONE_CONFIG_B64="$RCLONE_CONFIG_B64"
export SAM_CHECKPOINT_URL="$SAM_URL"
export RESUME_LORA_URL="$RESUME_URL"
export EPOCHS="$EPOCHS"
export BATCH_SIZE="$BATCH"
export TILE_SIZE="$TILE"
export TILE_STRIDE="$STRIDE"
export LR="$LR"
export LR_RESUME="$LR_RESUME"
export INFER_N="$INFER_N"
bash scripts/startup.sh || true
EOS
)

# Create instance with image and onstart
RESP=$(vastai --api-key "$VASTAI_API_KEY" create instance "$OFFER_ID" --image "$IMAGE" --disk 60 --ssh --direct --label lorasam2 --onstart-cmd "$ONSTART_CMD")
echo "$RESP"

# Extract instance id, print SSH and logs help
INSTANCE_ID=$(echo "$RESP" | python - <<'PY'
import json,sys
try:
  data=json.loads(sys.stdin.read())
  print(data.get('new_contract') or data.get('id') or '')
except Exception:
  print('')
PY
)

if [[ -z "$INSTANCE_ID" ]]; then
  echo "Could not extract instance id from response" >&2
  exit 4
fi

echo "Instance ID: $INSTANCE_ID"
echo "SSH: $(vastai --api-key "$VASTAI_API_KEY" ssh-url $INSTANCE_ID)"
echo "Logs: vastai --api-key $VASTAI_API_KEY logs $INSTANCE_ID --tail 200"
echo "To stop: vastai --api-key $VASTAI_API_KEY stop instance $INSTANCE_ID"
echo "To destroy: vastai --api-key $VASTAI_API_KEY destroy instance $INSTANCE_ID"
