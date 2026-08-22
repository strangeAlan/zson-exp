#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
SERVICE_ROOT="${ZSON3_SERVICE_ROOT:-${ROOT}}"
PYTHON="${ZSON3_PYTHON:-${SERVICE_ROOT}/.local/envs/zson3/bin/python}"
SEED="${ZSON3_EVAL_SEED:-20260727}"
RUN_ID="${ZSON3_RUN_ID:-openfrontier_base_sam3_full_hm3dv2_repro_1000_seed${SEED}}"
OUT_DIR="${ZSON3_OUTPUT_DIR:-${ROOT}/results/${RUN_ID}}"
EPISODE_MANIFEST="${ZSON3_EPISODE_MANIFEST:-${ROOT}/results/openfrontier_base_sam3_full_hm3dv2_1000_seed20260727/manifest.json}"
EXPECTED_MANIFEST_SHA256="${ZSON3_EXPECTED_MANIFEST_SHA256:-c7ef8f4bcc42a54d29932c71ff6371e46bccb8e720ad9afaa9f89df2e6271374}"

actual_manifest_sha256="$(sha256sum "${EPISODE_MANIFEST}" | awk '{print $1}')"
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "HM3Dv2 source manifest hash mismatch: ${actual_manifest_sha256}" >&2
  exit 1
fi

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="${ZSON3_NAV_GPU:-0}"
export PYTHONUNBUFFERED=1
export ZSON3_REQUIRE_SAM3=1
export ZSON3_REQUIRE_APEX_TARGET=0
export ZSON3_QWEN_BACKEND="${ZSON3_QWEN_BACKEND:-vllm}"
export ZSON3_QWEN_API_STYLE="${ZSON3_QWEN_API_STYLE:-openai}"
export ZSON3_QWEN_MODEL="${ZSON3_QWEN_MODEL:-qwen3-vl-8b}"

mkdir -p "${OUT_DIR}"
touch "${OUT_DIR}/raw.log" "${OUT_DIR}/progress.log"
cd "${ROOT}"

printf '[ZSON3 LAUNCH] openfrontier_sam3 full_hm3dv2 run_id=%s manifest_sha256=%s\n' \
  "${RUN_ID}" "${actual_manifest_sha256}" | tee -a "${OUT_DIR}/progress.log"
env ZSON3_ROOT="${SERVICE_ROOT}" ZSON3_REQUIRE_SAM3=1 \
  ZSON3_REQUIRE_APEX_TARGET=0 "${SERVICE_ROOT}/scripts/launch_model_services.sh"

set +e
"${PYTHON}" -u scripts/run_hm3dv1_random100.py \
  --dataset hm3dv2 \
  --seed "${SEED}" \
  --episode-manifest "${EPISODE_MANIFEST}" \
  --max-steps 500 \
  --max-time 3600 \
  --config config/zson3/navigation_hm3dv1_qwen.yaml \
  --output-dir "${OUT_DIR}" \
  --resume \
  2>&1 | tee -a "${OUT_DIR}/raw.log" | stdbuf -oL grep --line-buffered -E \
    '^\[ZSON3 EVAL\]|^\[ZSON3 ABORT\]|^\[ZSON3 SUMMARY\]|Traceback|Exception|CUDA out of memory|Killed' \
  | tee -a "${OUT_DIR}/progress.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\n' "${RUN_STATUS}" >> "${OUT_DIR}/summary.txt"
exit "${RUN_STATUS}"
