#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
VLLM_ENV="${ZSON3_QWEN_VLLM_ENV:-${ROOT}/.local/envs/qwen-vllm}"
MODEL_PATH="${ZSON3_QWEN_MODEL_PATH:-${ROOT}/.local/models/qwen3-vl-8b}"
SERVED_MODEL="${ZSON3_QWEN_MODEL:-qwen3-vl-8b}"
GPU="${ZSON3_QWEN_GPU:-1}"
PORT="${ZSON3_QWEN_PORT:-18080}"
GPU_UTIL="${ZSON3_QWEN_VLLM_GPU_UTIL:-0.70}"
MAX_MODEL_LEN="${ZSON3_QWEN_MAX_MODEL_LEN:-4096}"
BATCH_INVARIANT="${ZSON3_QWEN_BATCH_INVARIANT:-1}"
RUNTIME_DIR="${ZSON3_RUNTIME_DIR:-${ROOT}/.runtime}"
LOG_DIR="${ZSON3_QWEN_LOG_DIR:-${ROOT}/logs/qwen-vllm}"
pid_file="${RUNTIME_DIR}/qwen-${PORT}.pid"
log_file="${LOG_DIR}/server-${PORT}.log"

models_url="http://127.0.0.1:${PORT}/v1/models"
if payload="$(curl --noproxy '*' -fsS --max-time 3 "${models_url}" 2>/dev/null)" \
  && [[ "${payload}" == *"\"${SERVED_MODEL}\""* ]]; then
  owned_pid=""
  if [[ -f "${pid_file}" ]]; then
    owned_pid="$(<"${pid_file}")"
  fi
  owned_cmdline="$(tr '\0' ' ' <"/proc/${owned_pid}/cmdline" 2>/dev/null || true)"
  if [[ -n "${owned_pid}" && "${owned_cmdline}" == *"vllm serve"* ]]; then
    owned_env="$(tr '\0' '\n' <"/proc/${owned_pid}/environ" 2>/dev/null || true)"
    if [[ "${owned_env}" != *"VLLM_BATCH_INVARIANT=${BATCH_INVARIANT}"* \
      || "${owned_cmdline}" != *"--attention-backend FLASH_ATTN"* \
      || "${owned_cmdline}" != *"--no-enable-prefix-caching"* ]]; then
      echo "[zson3:qwen-vllm] healthy owned service does not satisfy the requested reproducibility contract" >&2
      echo "[zson3:qwen-vllm] stop it before restarting with the current launcher" >&2
      exit 1
    fi
    echo "[zson3:qwen-vllm] owned service is healthy pid=${owned_pid} batch_invariant=${BATCH_INVARIANT} prefix_cache=off: ${models_url}"
    exit 0
  fi
  if [[ "${ZSON3_ALLOW_EXTERNAL_QWEN:-0}" == "1" ]]; then
    echo "[zson3:qwen-vllm] external service accepted by explicit override: ${models_url}"
    exit 0
  fi
  echo "[zson3:qwen-vllm] healthy but unowned service on port ${PORT}; refusing an unstable evaluation dependency" >&2
  echo "[zson3:qwen-vllm] stop it first, or set ZSON3_ALLOW_EXTERNAL_QWEN=1 explicitly" >&2
  exit 1
fi

for required in "${VLLM_ENV}/bin/vllm" "${MODEL_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[zson3:qwen-vllm] missing resource: ${required}" >&2
    exit 1
  fi
done
if bash -c "</dev/tcp/127.0.0.1/${PORT}" >/dev/null 2>&1; then
  echo "[zson3:qwen-vllm] port ${PORT} is occupied by an incompatible service" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
rm -f "${pid_file}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

nohup setsid env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  VLLM_BATCH_INVARIANT="${BATCH_INVARIANT}" \
  "${VLLM_ENV}/bin/vllm" serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs 1 \
  --attention-backend FLASH_ATTN \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --generation-config vllm \
  --trust-remote-code \
  </dev/null >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${pid_file}"

for _ in $(seq 1 180); do
  if payload="$(curl --noproxy '*' -fsS --max-time 3 "${models_url}" 2>/dev/null)" \
    && [[ "${payload}" == *"\"${SERVED_MODEL}\""* ]]; then
    echo "[zson3:qwen-vllm] started pid=${pid}: ${models_url}"
    exit 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[zson3:qwen-vllm] process exited; see ${log_file}" >&2
    tail -n 80 "${log_file}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "[zson3:qwen-vllm] health timeout; see ${log_file}" >&2
exit 1
