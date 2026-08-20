#!/usr/bin/env bash
set -euo pipefail

QWEN_PORT="${ZSON3_QWEN_PORT:-18080}"
SAM3_PORT="${ZSON3_SAM3_PORT:-12186}"
if payload="$(curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${QWEN_PORT}/v1/models" 2>/dev/null)" \
  && [[ "${payload}" == *'"qwen3-vl-8b"'* ]]; then
  echo "qwen3-vl-8b  healthy  backend=vllm  http://127.0.0.1:${QWEN_PORT}"
else
  curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${QWEN_PORT}/health" >/dev/null
  echo "qwen3-vl-8b  healthy  backend=transformers  http://127.0.0.1:${QWEN_PORT}"
fi

if payload="$(curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${SAM3_PORT}/health" 2>/dev/null)" \
  && [[ "${payload}" == *'"service":"sam3"'* || "${payload}" == *'"service": "sam3"'* ]]; then
  echo "sam3  healthy  http://127.0.0.1:${SAM3_PORT}"
else
  echo "sam3  unavailable  127.0.0.1:${SAM3_PORT}"
fi

for entry in "groundingdino:12181" "blip2-itm:12182" "mobile-sam:12183" "yolov7:12184"; do
  name="${entry%%:*}"
  port="${entry##*:}"
  if bash -c "</dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
    echo "${name}  listening  127.0.0.1:${port}"
  else
    echo "${name}  unavailable  127.0.0.1:${port}"
  fi
done
