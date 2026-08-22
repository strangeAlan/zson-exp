# Qwen runtime

ZSON3 supports two out-of-process Qwen3-VL backends behind the same semantic
adapter. The default benchmark backend is vLLM; the Transformers server remains
available as a bounded fallback.

## Frozen vLLM environment

The validated server environment is independent from `zson3`, `vlfm`, and
`multi-agent-nav`:

```text
environment: zson3-qwen-vllm
Python:      3.12
vLLM:        0.15.1
Torch:       2.9.1+cu128
CUDA wheel:  12.8
model:       /home/hsy/model/qwen/Qwen3-VL-8B-Instruct
```

Do not upgrade this environment to vLLM 0.20 or newer without checking its
Torch wheel. vLLM 0.26 installed Torch `cu130`, which cannot run on the current
570.133.20 driver. The 0.15.1/`cu128` combination passed `pip check`, loaded
Qwen3-VL, and served the OpenAI-compatible multimodal endpoint.

The server is limited to one concurrent sequence, a 4096-token context, and
70% of GPU1. This leaves enough GPU1 memory for the isolated SAM3 service.

## Measured fixed-fixture latency

All measurements used the same HM3Dv1 turn-6 image and byte-identical
OpenFrontier A/B/C prompt:

```text
Transformers with CPU offload:       61.10 s
Transformers, full GPU + SDPA:        6.52 s
Transformers, full GPU + FlashAttn:   6.20 s (steady state)
vLLM 0.15.1 first request:            5.01 s
vLLM 0.15.1 steady state:             1.59 s
```

The prompt parser and probability contract passed unchanged. Generated wording
is not expected to be byte-identical across inference engines.

## Backends

Start the default vLLM backend:

```bash
ZSON3_QWEN_BACKEND=vllm scripts/launch_model_services.sh
```

Use the Transformers fallback explicitly:

```bash
ZSON3_QWEN_BACKEND=transformers \
ZSON3_QWEN_API_STYLE=native \
scripts/launch_model_services.sh
```

Navigation processes using vLLM must export:

```bash
export ZSON3_QWEN_API_STYLE=openai
export ZSON3_QWEN_MODEL=qwen3-vl-8b
```

`scripts/run_openfrontier_random100.sh` sets these variables itself.

## Evaluation service ownership

The default launcher only accepts a healthy vLLM process whose PID is recorded
in `.runtime/qwen-18080.pid` and whose command line matches `vllm serve`. Qwen
and SAM3 are detached into independent sessions and write dedicated server
logs. An external Qwen may be accepted explicitly with
`ZSON3_ALLOW_EXTERNAL_QWEN=1`; otherwise evaluation refuses an unmanaged
process instead of silently depending on its terminal lifetime.

The evaluator checks Qwen and SAM3 before every episode. A transport failure
during an episode aborts immediately and writes the interrupted record under
`failures/`, not `episodes/`, so `--resume` retries that manifest entry.
