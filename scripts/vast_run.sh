#!/usr/bin/env bash
# Vast.ai setup + training script. Usage: bash scripts/vast_run.sh <remote_dir>
set -euo pipefail

REMOTE_DIR=${1:-/workspace/lorasam2}

# Install deps
python3 -m pip install --upgrade pip
pip install -r requirements.txt
# Install SAM (GitHub) if not provided
if ! python -c "import segment_anything" 2>/dev/null; then
  pip install git+https://github.com/facebookresearch/segment-anything.git
fi

mkdir -p weights outputs/sam_lora outputs/preds
# Download SAM ViT-B checkpoint if missing (requires hosted URL; user should upload or mount)
if [ ! -f weights/sam_vit_b_01ec64.pth ]; then
  echo "Place SAM ViT-B checkpoint at weights/sam_vit_b_01ec64.pth" >&2
fi

python train_lora_sam.py --config configs/config.yaml
