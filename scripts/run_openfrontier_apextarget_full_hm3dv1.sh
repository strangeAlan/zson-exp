#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${ZSON3_PYTHON:-${ROOT}/.local/envs/zson3/bin/python}"
SEED="${ZSON3_EVAL_SEED:-20260727}"
RUN_ID="${ZSON3_RUN_ID:-openfrontier_apextarget_v1_deterministic_full_hm3dv1_2000_seed${SEED}}"
OUT_DIR="${ROOT}/results/${RUN_ID}"
FLOOR_THRESHOLD="${ZSON3_FLOOR_HEIGHT_THRESHOLD:-1.5}"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="${ZSON3_NAV_GPU:-0}"
export PYTHONUNBUFFERED=1
export ZSON3_REQUIRE_SAM3=0
export ZSON3_REQUIRE_APEX_TARGET=1
export ZSON3_QWEN_BACKEND="${ZSON3_QWEN_BACKEND:-vllm}"
export ZSON3_QWEN_API_STYLE="${ZSON3_QWEN_API_STYLE:-openai}"
export ZSON3_QWEN_MODEL="${ZSON3_QWEN_MODEL:-qwen3-vl-8b}"

mkdir -p "${OUT_DIR}"
touch "${OUT_DIR}/raw.log" "${OUT_DIR}/progress.log"
cd "${ROOT}"

printf '[ZSON3 LAUNCH] OF-ApexTarget-v1 deterministic full_hm3dv1 run_id=%s\n' "${RUN_ID}" \
  | tee -a "${OUT_DIR}/progress.log"

"${PYTHON}" scripts/analyze_hm3dv1_floor_requirements.py \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --floor-height-threshold "${FLOOR_THRESHOLD}" \
  --expected-episodes 2000 \
  2>&1 | tee -a "${OUT_DIR}/progress.log"

set +e
"${ROOT}/scripts/launch_model_services.sh" 2>&1 \
  | tee -a "${OUT_DIR}/raw.log" \
  | tee -a "${OUT_DIR}/progress.log"
SERVICE_STATUS=${PIPESTATUS[0]}
set -e
if (( SERVICE_STATUS != 0 )); then
  printf '[ZSON3 ABORT] service_preflight_failed exit_code=%s\n' "${SERVICE_STATUS}" \
    | tee -a "${OUT_DIR}/progress.log"
  exit "${SERVICE_STATUS}"
fi

set +e
"${PYTHON}" -u scripts/run_hm3dv1_random100.py \
  --seed "${SEED}" \
  --all-episodes \
  --max-steps 500 \
  --max-time 3600 \
  --config config/zson3/navigation_hm3dv1_qwen_apextarget.yaml \
  --output-dir "${OUT_DIR}" \
  --resume \
  2>&1 | tee -a "${OUT_DIR}/raw.log" | stdbuf -oL grep --line-buffered -E \
    '^\[ZSON3 EVAL\]|^\[ZSON3 ABORT\]|^\[ZSON3 SUMMARY\]|Traceback|Exception|CUDA out of memory|Killed' \
  | tee -a "${OUT_DIR}/progress.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

set +e
"${PYTHON}" scripts/analyze_hm3dv1_floor_requirements.py \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --floor-height-threshold "${FLOOR_THRESHOLD}" \
  --expected-episodes 2000 \
  2>&1 | tee -a "${OUT_DIR}/progress.log"
FLOOR_STATUS=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\nfloor_audit_exit_code=%s\n' "${RUN_STATUS}" "${FLOOR_STATUS}" \
  >> "${OUT_DIR}/summary.txt"
if (( RUN_STATUS != 0 )); then
  exit "${RUN_STATUS}"
fi
exit "${FLOOR_STATUS}"
