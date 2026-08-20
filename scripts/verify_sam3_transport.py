#!/usr/bin/env python3
"""One-shot semantic equivalence and latency check for SAM3 wire formats."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sam3_transport import decode_masks, encode_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_turn6.npz"),
    )
    parser.add_argument("--prompt", default="bed")
    parser.add_argument("--url", default="http://127.0.0.1:12186/sam3")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sam3_transport_equivalence.json")
    )
    args = parser.parse_args()

    with np.load(args.fixture) as fixture:
        image = np.asarray(fixture["rgb"], dtype=np.uint8)

    legacy_request = {"image": image.tolist(), "prompt": args.prompt}
    packed_request = {
        **encode_image(image),
        "prompt": args.prompt,
        "response_format": "packed-v1",
    }
    session = requests.Session()
    session.trust_env = False

    responses = []
    for name, payload in (("legacy", legacy_request), ("packed", packed_request)):
        started = time.perf_counter()
        response = session.post(args.url, json=payload, timeout=(3, 180))
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        body = response.json()
        if body.get("result") != "success":
            raise RuntimeError(f"{name} SAM3 call failed: {body}")
        responses.append((name, payload, body, elapsed, len(response.content)))

    legacy = responses[0][2]
    packed = responses[1][2]
    legacy_masks = decode_masks(legacy)
    packed_masks = decode_masks(packed)
    masks_equal = (
        legacy_masks.size == 0 and packed_masks.size == 0
    ) or np.array_equal(legacy_masks, packed_masks)
    boxes_equal = np.array_equal(
        np.asarray(legacy["boxes"], dtype=np.float32),
        np.asarray(packed["boxes"], dtype=np.float32),
    )
    scores_equal = np.array_equal(
        np.asarray(legacy["scores"], dtype=np.float32),
        np.asarray(packed["scores"], dtype=np.float32),
    )
    report = {
        "fixture": str(args.fixture),
        "prompt": args.prompt,
        "semantic_equivalence": {
            "masks_exact": masks_equal,
            "legacy_mask_shape": list(legacy_masks.shape),
            "packed_mask_shape": list(packed_masks.shape),
            "boxes_float32_exact": boxes_equal,
            "scores_float32_exact": scores_equal,
        },
        "legacy": {
            "request_json_bytes": len(json.dumps(responses[0][1])),
            "response_bytes": responses[0][4],
            "round_trip_seconds": responses[0][3],
            "server_timings": legacy.get("timings"),
        },
        "packed": {
            "request_json_bytes": len(json.dumps(responses[1][1])),
            "response_bytes": responses[1][4],
            "round_trip_seconds": responses[1][3],
            "server_timings": packed.get("timings"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not (masks_equal and boxes_equal and scores_equal):
        raise SystemExit("SAM3 packed transport changed model outputs")


if __name__ == "__main__":
    main()
