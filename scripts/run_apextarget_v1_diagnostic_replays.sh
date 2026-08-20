#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${ZSON3_PYTHON:-${ROOT}/.local/envs/zson3/bin/python}"
GROUP="${ZSON3_DIAGNOSTIC_GROUP:-quick}"
OUTPUT_ROOT="${ZSON3_DIAGNOSTIC_ROOT:-${ROOT}/artifacts/diagnostics/apextarget_v1}"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="${ZSON3_NAV_GPU:-0}"
export PYTHONUNBUFFERED=1
export ZSON3_REQUIRE_SAM3=0
export ZSON3_REQUIRE_APEX_TARGET=1
export ZSON3_QWEN_BACKEND="${ZSON3_QWEN_BACKEND:-vllm}"
export ZSON3_QWEN_API_STYLE="${ZSON3_QWEN_API_STYLE:-openai}"
export ZSON3_QWEN_MODEL="${ZSON3_QWEN_MODEL:-qwen3-vl-8b}"

case "${GROUP}" in
  quick)
    # Two GT-unmatched reliable targets plus the shortest safe-endpoint failure.
    CASES=(
      "5cdEh9F2hJL:97:gt_unmatched_bed"
      "wcojb4TFT35:24:gt_unmatched_tv"
      "mv2HUxq3B53:63:safe_endpoint_chair"
    )
    ;;
  visible_fn)
    # Detector miss, geometry rejection, and fusion/association failure candidates.
    CASES=(
      "6s7QHgap2fW:7:visible_yolo_miss_tv"
      "mv2HUxq3B53:39:visible_geometry_reject_toilet"
      "bxsVRursffK:89:visible_fusion_no_reliable_chair"
    )
    ;;
  *)
    echo "Unsupported ZSON3_DIAGNOSTIC_GROUP=${GROUP}; use quick or visible_fn" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT}"
"${ROOT}/scripts/launch_model_services.sh"

progress="${OUTPUT_ROOT}/${GROUP}.progress.log"
printf '[ZSON3 DIAGNOSTIC] group=%s cases=%s\n' "${GROUP}" "${#CASES[@]}" \
  | tee -a "${progress}"

for spec in "${CASES[@]}"; do
  IFS=: read -r scene episode label <<<"${spec}"
  output_dir="${OUTPUT_ROOT}/${label}_${scene}_ep${episode}"
  if [[ -f "${output_dir}/result.json" ]]; then
    echo "[ZSON3 DIAGNOSTIC] skip completed ${label} ${scene}/${episode}" \
      | tee -a "${progress}"
    continue
  fi
  mkdir -p "${output_dir}"
  echo "[ZSON3 DIAGNOSTIC] start ${label} ${scene}/${episode}" \
    | tee -a "${progress}"
  set +e
  "${PYTHON}" -u scripts/run_fixed_hm3dv1_episode.py \
    --scene "${scene}" \
    --episode-id "${episode}" \
    --seed 20260727 \
    --max-steps 500 \
    --max-time 3600 \
    --config config/zson3/navigation_hm3dv1_qwen_apextarget.yaml \
    --output-dir "${output_dir}" \
    --save-images \
    --apex-target-diagnostics \
    >"${output_dir}/raw.log" 2>&1
  status=$?
  set -e
  echo "[ZSON3 DIAGNOSTIC] done ${label} ${scene}/${episode} exit=${status}" \
    | tee -a "${progress}"
  if (( status != 0 )); then
    tail -n 80 "${output_dir}/raw.log" >&2
    exit "${status}"
  fi
  summary_line="$(grep -F '[ZSON3 FIXED SUMMARY]' "${output_dir}/raw.log" | tail -1 || true)"
  if [[ -n "${summary_line}" ]]; then
    echo "${summary_line}" | tee -a "${progress}"
  fi
done
