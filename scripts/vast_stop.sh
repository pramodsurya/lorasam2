#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: VASTAI_API_KEY=*** ./scripts/vast_stop.sh <instance_id> [--destroy]" >&2
  exit 2
fi

VASTAI_API_KEY=${VASTAI_API_KEY:-}
[[ -z "$VASTAI_API_KEY" ]] && { echo "Set VASTAI_API_KEY in env" >&2; exit 2; }

ID=$1
DESTROY=${2:-}

vastai --api-key "$VASTAI_API_KEY" stop instance "$ID" || true
if [[ "$DESTROY" == "--destroy" ]]; then
  vastai --api-key "$VASTAI_API_KEY" destroy instance "$ID" || true
fi
echo "Done."
