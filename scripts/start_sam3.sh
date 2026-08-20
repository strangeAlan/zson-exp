#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
SAM3_ENV="${ZSON3_SAM3_ENV:-${ROOT}/.local/envs/sam3}"
SAM3_GPU="${ZSON3_SAM3_GPU:-1}"
SAM3_PORT="${ZSON3_SAM3_PORT:-12186}"
SAM3_CHECKPOINT="${ZSON3_SAM3_CHECKPOINT:-${ROOT}/.local/models/sam3.pt}"
RUNTIME_DIR="${ZSON3_RUNTIME_DIR:-${ROOT}/.runtime}"
LOG_DIR="${ZSON3_SAM3_LOG_DIR:-${ROOT}/logs/sam3}"
server_script="${ROOT}/sam3_server.py"
health_url="http://127.0.0.1:${SAM3_PORT}/health"
pid_file="${RUNTIME_DIR}/sam3-${SAM3_PORT}.pid"
log_file="${LOG_DIR}/server-${SAM3_PORT}.log"

if payload="$(curl --noproxy '*' -fsS --max-time 3 "${health_url}" 2>/dev/null)" \
  && [[ "${payload}" == *'"service":"sam3"'* || "${payload}" == *'"service": "sam3"'* ]]; then
  owned_pid=""
  if [[ -f "${pid_file}" ]]; then
    owned_pid="$(<"${pid_file}")"
  fi
  owned_cmdline="$(tr '\0' ' ' <"/proc/${owned_pid}/cmdline" 2>/dev/null || true)"
  if [[ -n "${owned_pid}" && "${owned_cmdline}" == *"sam3_server.py"* ]]; then
    echo "[zson3:sam3] owned service is healthy pid=${owned_pid}: ${health_url}"
    exit 0
  fi
  if [[ "${ZSON3_ALLOW_EXTERNAL_SAM3:-0}" == "1" ]]; then
    echo "[zson3:sam3] external service accepted by explicit override: ${health_url}"
    exit 0
  fi
  echo "[zson3:sam3] healthy but unowned service on port ${SAM3_PORT}; refusing an unstable evaluation dependency" >&2
  echo "[zson3:sam3] stop it first, or set ZSON3_ALLOW_EXTERNAL_SAM3=1 explicitly" >&2
  exit 1
fi

for required in "${SAM3_ENV}/bin/python" "${server_script}" "${ROOT}/third_party/sam3/sam3"; do
  if [[ ! -e "${required}" ]]; then
    echo "[zson3:sam3] missing resource: ${required}" >&2
    exit 1
  fi
done

"${SAM3_ENV}/bin/python" -c 'import einops, flask, sam3' || {
  echo "[zson3:sam3] environment import closure failed" >&2
  exit 1
}

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
rm -f "${pid_file}"
if [[ ! -f "${SAM3_CHECKPOINT}" ]]; then
  echo "[zson3:sam3] checkpoint does not exist: ${SAM3_CHECKPOINT}" >&2
  exit 1
fi
cmd=("${SAM3_ENV}/bin/python" "${server_script}" --port "${SAM3_PORT}" --checkpoint "${SAM3_CHECKPOINT}")

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
nohup setsid env CUDA_VISIBLE_DEVICES="${SAM3_GPU}" \
  "${cmd[@]}" </dev/null >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${pid_file}"

for _ in $(seq 1 180); do
  if payload="$(curl --noproxy '*' -fsS --max-time 3 "${health_url}" 2>/dev/null)" \
    && [[ "${payload}" == *'"service":"sam3"'* || "${payload}" == *'"service": "sam3"'* ]]; then
    echo "[zson3:sam3] started pid=${pid}: ${health_url}"
    exit 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[zson3:sam3] process exited; see ${log_file}" >&2
    tail -80 "${log_file}" >&2 || true
    rm -f "${pid_file}"
    exit 1
  fi
  sleep 5
done

echo "[zson3:sam3] health timeout; see ${log_file}" >&2
exit 1
