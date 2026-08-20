#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
case "${ZSON3_QWEN_BACKEND:-vllm}" in
  vllm) "${ROOT}/scripts/start_qwen_vllm.sh" ;;
  transformers) "${ROOT}/scripts/start_local_qwen.sh" ;;
  *) echo "Unsupported ZSON3_QWEN_BACKEND=${ZSON3_QWEN_BACKEND}" >&2; exit 1 ;;
esac
if [[ "${ZSON3_REQUIRE_SAM3:-0}" == "1" ]]; then
  "${ROOT}/scripts/start_sam3.sh"
fi
if [[ "${ZSON3_REQUIRE_APEX_TARGET:-0}" == "1" ]]; then
  "${ROOT}/scripts/start_apex_target_services.sh"
fi
"${ROOT}/scripts/check_model_services.sh"
