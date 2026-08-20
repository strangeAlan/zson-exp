#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WEIGHTS_DIR="${ROOT}/model_weights"

mkdir -p "${WEIGHTS_DIR}"
gdown --id 11SugqEg3LR2voKdLvq9Xe_zch10ek006 \
  --output "${WEIGHTS_DIR}/rgbd_11cls.pth"
curl -fL --retry 3 \
  https://raw.githubusercontent.com/rai-opensource/vlfm/refs/heads/main/data/pointnav_weights.pth \
  --output "${WEIGHTS_DIR}/pointnav_weights.pth"

echo "Weights downloaded to ${WEIGHTS_DIR}"
