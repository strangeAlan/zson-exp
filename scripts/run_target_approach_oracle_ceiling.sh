#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${ZSON3_PYTHON:-${ROOT}/.local/envs/zson3/bin/python}"
SEED="${ZSON3_EVAL_SEED:-20260727}"
MANIFEST="${ZSON3_ORACLE_MANIFEST:-${ROOT}/config/evaluation/hm3dv2_approach_oracle_probe23.json}"
RUN_ROOT="${ZSON3_OUTPUT_DIR:-${ROOT}/results/target_approach_oracle_ceiling_seed${SEED}}"
PROTECTION="${RUN_ROOT}/protection_ofbase"
ORACLE_A="${RUN_ROOT}/oracle_a"
ORACLE_B="${RUN_ROOT}/oracle_b"
ORACLE_B_MANIFEST="${RUN_ROOT}/oracle_b_manifest.json"
SUMMARY="${RUN_ROOT}/oracle_summary_v1.json"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="${ZSON3_NAV_GPU:-0}"
export PYTHONUNBUFFERED=1
export ZSON3_REQUIRE_SAM3=1
export ZSON3_REQUIRE_APEX_TARGET=0
export ZSON3_QWEN_BACKEND="${ZSON3_QWEN_BACKEND:-vllm}"
export ZSON3_QWEN_API_STYLE="${ZSON3_QWEN_API_STYLE:-openai}"
export ZSON3_QWEN_MODEL="${ZSON3_QWEN_MODEL:-qwen3-vl-8b}"

mkdir -p "${RUN_ROOT}"
cd "${ROOT}"
env ZSON3_ROOT="${ROOT}" ZSON3_REQUIRE_SAM3=1 \
  ZSON3_REQUIRE_APEX_TARGET=0 "${ROOT}/scripts/launch_model_services.sh"

run_eval() {
  local name="$1"
  shift
  local out="${RUN_ROOT}/${name}"
  mkdir -p "${out}"
  touch "${out}/raw.log" "${out}/progress.log"
  "${PYTHON}" -u scripts/run_hm3dv1_random100.py \
    --dataset hm3dv2 \
    --seed "${SEED}" \
    --episode-manifest "${MANIFEST}" \
    --max-time 3600 \
    --config config/zson3/navigation_hm3dv1_qwen.yaml \
    --output-dir "${out}" \
    --resume \
    "$@" \
    2>&1 | tee -a "${out}/raw.log" | stdbuf -oL grep --line-buffered -E \
      '^\[ZSON3 EVAL\]|^\[ZSON3 ABORT\]|^\[ZSON3 SUMMARY\]|Traceback|Exception|CUDA out of memory|Killed' \
      | tee -a "${out}/progress.log"
}

# The protection replay uses the ordinary PointnavAgent at the official 500
# step budget. Oracle code is neither imported nor instantiated here.
run_eval protection_ofbase --manifest-cohort protection --max-steps 500

# Ceiling executions receive one fixed, uniform 300-step post-acceptance
# allowance. 800 is exactly the original 500-step horizon plus that allowance.
run_eval oracle_a --manifest-cohort diagnostic --target-approach-oracle a \
  --oracle-post-accept-steps 300 --max-steps 800

"${PYTHON}" scripts/audit_target_approach_oracle.py \
  --manifest "${MANIFEST}" \
  --oracle-a "${ORACLE_A}" \
  --output "${ORACLE_B_MANIFEST}" \
  --prepare-b

b_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["selection"]))' "${ORACLE_B_MANIFEST}")"
if [[ "${b_count}" -gt 0 ]]; then
  MANIFEST="${ORACLE_B_MANIFEST}"
  run_eval oracle_b --target-approach-oracle b \
    --oracle-post-accept-steps 300 --max-steps 800
else
  mkdir -p "${ORACLE_B}/episodes"
fi

"${PYTHON}" scripts/audit_target_approach_oracle.py \
  --manifest "${ZSON3_ORACLE_MANIFEST:-${ROOT}/config/evaluation/hm3dv2_approach_oracle_probe23.json}" \
  --oracle-a "${ORACLE_A}" \
  --oracle-b "${ORACLE_B}" \
  --protection "${PROTECTION}" \
  --output "${SUMMARY}" | tee "${RUN_ROOT}/audit.log"

