#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZSON3_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUNTIME_DIR="${ZSON3_RUNTIME_DIR:-${ROOT}/.runtime}"

stop_owned_service() {
  local name="$1"
  local port="$2"
  local marker="$3"
  local alternate_marker="${4:-}"
  local pid_file="${RUNTIME_DIR}/${name}-${port}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "[zson3:${name}] no ZSON3-owned PID record; nothing stopped"
    return
  fi
  local pid cmdline
  pid="$(<"${pid_file}")"
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${cmdline}" != *"${marker}"* ]] \
    && { [[ -z "${alternate_marker}" ]] || [[ "${cmdline}" != *"${alternate_marker}"* ]]; }; then
    echo "[zson3:${name}] PID ${pid} does not match an owned service; refusing to stop" >&2
    return 1
  fi
  kill "${pid}"
  rm -f "${pid_file}"
  echo "[zson3:${name}] stopped ZSON3-owned pid=${pid}"
}

stop_owned_service qwen "${ZSON3_QWEN_PORT:-18080}" qwen_server.py "vllm serve"
stop_owned_service sam3 "${ZSON3_SAM3_PORT:-12186}" sam3_server.py
