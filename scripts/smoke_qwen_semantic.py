#!/usr/bin/env python3
"""Run the preserved OpenFrontier semantic scorer through local Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from vlm.models import VLMModel
from vlm.utils import build_frontier_probability_prompt, detect_frontier_probabilities
from zson3.services.qwen import QwenClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_turn6_som.png"),
    )
    parser.add_argument("--labels", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--target", default="chair")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    prompt = build_frontier_probability_prompt(args.labels, args.target)
    health = QwenClient().health()
    started = time.perf_counter()
    success, probabilities, raw_response = detect_frontier_probabilities(
        rgb_image=image,
        labels=args.labels,
        target_object=args.target,
        vlm_model=VLMModel.QWEN3_VL_8B_LOCAL,
    )
    latency = time.perf_counter() - started
    if not success:
        raise RuntimeError(f"Qwen response did not satisfy the upstream parser: {raw_response}")
    for label in args.labels:
        value = probabilities.get(label)
        if not isinstance(value, list) or len(value) < 2:
            raise RuntimeError(f"Missing probability/reason pair for frontier {label}: {value!r}")
        if not isinstance(value[0], (int, float)) or not 0 <= value[0] <= 1:
            raise RuntimeError(f"Invalid probability for frontier {label}: {value[0]!r}")
    print(json.dumps({
        "status": "ok",
        "health": health,
        "model": VLMModel.QWEN3_VL_8B_LOCAL.value,
        "target": args.target,
        "labels": args.labels,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_array_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        "latency_seconds": latency,
        "raw_response": raw_response,
        "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "parsed_probabilities": probabilities,
    }, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
