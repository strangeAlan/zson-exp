#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
QWEN_ENV="${ZSON3_QWEN_ENV:-${ROOT}/.local/envs/qwen-transformers}"
QWEN_SERVER_SCRIPT="${ZSON3_QWEN_SERVER_SCRIPT:-${ROOT}/zson3/services/qwen_server.py}"
QWEN_MODEL_PATH="${ZSON3_QWEN_MODEL_PATH:-${ROOT}/.local/models/qwen3-vl-8b}"
QWEN_GPU="${ZSON3_QWEN_GPU:-1}"
QWEN_DEVICE_MAP="${ZSON3_QWEN_DEVICE_MAP:-cuda:0}"
QWEN_ATTENTION="${ZSON3_QWEN_ATTENTION:-flash_attention_2}"
QWEN_PORT="${ZSON3_QWEN_PORT:-18080}"
QWEN_MAX_NEW_TOKENS="${ZSON3_QWEN_SERVER_MAX_NEW_TOKENS:-512}"
RUNTIME_DIR="${ZSON3_RUNTIME_DIR:-${ROOT}/.runtime}"
LOG_DIR="${ZSON3_QWEN_LOG_DIR:-${ROOT}/logs/qwen}"

health_url="http://127.0.0.1:${QWEN_PORT}/health"
if curl --noproxy '*' -fsS --max-time 3 "${health_url}" >/dev/null 2>&1; then
  echo "[zson3:qwen] existing service is healthy: ${health_url}"
  exit 0
fi

for required in "${QWEN_ENV}/bin/python" "${QWEN_SERVER_SCRIPT}" "${QWEN_MODEL_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[zson3:qwen] missing resource: ${required}" >&2
    exit 1
  fi
done

if [[ "${QWEN_ATTENTION}" == "flash_attention_2" ]] \
  && ! "${QWEN_ENV}/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
  echo "[zson3:qwen] flash_attention_2 requested but flash_attn is unavailable in ${QWEN_ENV}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
pid_file="${RUNTIME_DIR}/qwen-${QWEN_PORT}.pid"
log_file="${LOG_DIR}/server-${QWEN_PORT}.log"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

nohup env CUDA_VISIBLE_DEVICES="${QWEN_GPU}" \
  "${QWEN_ENV}/bin/python" "${QWEN_SERVER_SCRIPT}" \
  --host 127.0.0.1 \
  --port "${QWEN_PORT}" \
  --model-path "${QWEN_MODEL_PATH}" \
  --dtype bfloat16 \
  --device-map "${QWEN_DEVICE_MAP}" \
  --attention-implementation "${QWEN_ATTENTION}" \
  --max-new-tokens "${QWEN_MAX_NEW_TOKENS}" \
  --temperature 0.0 \
  --log-dir "${LOG_DIR}" </dev/null >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${pid_file}"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS --max-time 3 "${health_url}" >/dev/null 2>&1; then
    echo "[zson3:qwen] started pid=${pid}: ${health_url}"
    exit 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[zson3:qwen] process exited; see ${log_file}" >&2
    exit 1
  fi
  sleep 5
done

echo "[zson3:qwen] health timeout; see ${log_file}" >&2
exit 1
