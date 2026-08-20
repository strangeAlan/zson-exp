#!/usr/bin/env python3
"""Run the unmodified upstream FrontierNet detector on a fixed fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from utils.frontier_utils import read_config_yaml


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_reset.npz"),
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("model_weights/rgbd_11cls.pth")
    )
    parser.add_argument("--config", type=Path, default=Path("config/navigation.yaml"))
    args = parser.parse_args()

    config = read_config_yaml(args.config)
    fixture = np.load(args.fixture)
    fixture_arrays = {
        key: array_hash(fixture[key]) for key in sorted(fixture.files)
    }
    model = load_model(
        args.weights, num_classes=config["num_classes"], use_depth=True
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FrontierDetector(
        model=model,
        camera_intrinsic=fixture["camera_intrinsic"].copy(),
        use_depth=True,
        img_size_model=tuple(config["input_img_size"]),
        device=device,
    )
    region, gain = detector.detect(
        fixture["rgb"],
        fixture["depth"],
        df_normalizer=config["df_normalizer"],
        df_thr=config["df_thr"],
    )
    frontiers = detector.anchor_fts(
        fixture["depth"], fixture["camera_extrinsic"]
    )
    frontier_trace = []
    for frontier in frontiers or []:
        frontier_trace.append(
            {
                "pixel_pos": [float(x) for x in frontier.pixel_pos],
                "pos3d": [float(x) for x in frontier.pos3d],
                "view_direction": [float(x) for x in frontier.view_direction],
                "gain": float(frontier.gain),
            }
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "device": device,
                "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                "fixture_arrays_sha256": fixture_arrays,
                "frontier_region_pixels": int(np.count_nonzero(region)),
                "frontier_region_sha256": array_hash(region),
                "information_gain_sha256": array_hash(gain),
                "anchored_frontiers": frontier_trace,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
