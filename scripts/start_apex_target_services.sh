#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
VLFM_ROOT="${VLFM_ROOT:-${ROOT}/.local/vlfm}"
VLFM_PYTHON="${VLFM_PYTHON:-${ROOT}/.local/envs/vlfm/bin/python}"
SESSION="${ZSON3_APEX_TARGET_SESSION:-zson3_apex_target_services}"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
MOBILE_SAM_CHECKPOINT="${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}"
if [[ "${MOBILE_SAM_CHECKPOINT}" != /* ]]; then
  MOBILE_SAM_CHECKPOINT="${VLFM_ROOT}/${MOBILE_SAM_CHECKPOINT}"
fi
export MOBILE_SAM_CHECKPOINT

# Keep the model runtime outside zson3. These are read-only references to the
# already validated VLFM service environment, so no checkpoint is duplicated.
declare -a REQUIRED_PATHS=(
  "${VLFM_PYTHON}"
  "${MOBILE_SAM_CHECKPOINT}"
  "${VLFM_ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
  "${VLFM_ROOT}/data/groundingdino_swint_ogc.pth"
  "${VLFM_ROOT}/data/yolov7-e6e.pt"
  "${VLFM_ROOT}/yolov7"
)
for required in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing Apex target service dependency: ${required}" >&2
    exit 1
  fi
done

declare -a SPECS=(
  "dino:12181:${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port 12181"
  "mobile_sam:12183:${VLFM_PYTHON} -m vlfm.vlm.sam --port 12183"
  "yolov7:12184:${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port 12184"
)

missing=()
for spec in "${SPECS[@]}"; do
  IFS=: read -r name port command <<<"${spec}"
  if bash -c "</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
    echo "${name} already listening on ${port}"
  else
    missing+=("${spec}")
  fi
done

if (( ${#missing[@]} == 0 )); then
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} exists but required ports are missing." >&2
  echo "Inspect it with: tmux attach-session -t ${SESSION}" >&2
  exit 1
fi

first=1
for spec in "${missing[@]}"; do
  IFS=: read -r name port command <<<"${spec}"
  launch="cd ${VLFM_ROOT} && ${command}"
  if (( first )); then
    tmux new-session -d -s "${SESSION}" -n "${name}" "${launch}"
    first=0
  else
    tmux new-window -t "${SESSION}" -n "${name}" "${launch}"
  fi
done

for _ in $(seq 1 180); do
  all_ready=1
  for spec in "${SPECS[@]}"; do
    IFS=: read -r name port command <<<"${spec}"
    if ! bash -c "</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
      all_ready=0
    fi
  done
  if (( all_ready )); then
    echo "Apex target services ready in tmux session ${SESSION}."
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for Apex target services; inspect ${SESSION}." >&2
exit 1
