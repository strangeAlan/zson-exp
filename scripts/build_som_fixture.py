#!/usr/bin/env python3
"""Rebuild the fixed OpenFrontier set-of-marks image from the HM3Dv1 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from utils.frontier_utils import read_config_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_turn6.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_turn6_som.png"),
    )
    parser.add_argument("--weights", type=Path, default=Path("model_weights/rgbd_11cls.pth"))
    parser.add_argument("--config", type=Path, default=Path("config/navigation.yaml"))
    args = parser.parse_args()

    config = read_config_yaml(args.config)
    fixture = np.load(args.fixture)
    model = load_model(args.weights, num_classes=config["num_classes"], use_depth=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FrontierDetector(
        model=model,
        camera_intrinsic=fixture["camera_intrinsic"].copy(),
        use_depth=True,
        img_size_model=tuple(config["input_img_size"]),
        device=device,
    )
    detector.detect(
        fixture["rgb"], fixture["depth"],
        df_normalizer=config["df_normalizer"], df_thr=config["df_thr"],
    )
    frontiers = detector.anchor_fts(fixture["depth"], fixture["camera_extrinsic"])
    if not frontiers:
        raise RuntimeError("The fixed fixture produced no anchored frontiers")
    for index, frontier in enumerate(frontiers):
        frontier.label = chr(65 + index) if index < 26 else str(index)
    image, labels, _ = detector.get_SoM_img(frontiers, radius=20, alpha=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8, copy=False)).save(args.output)
    print(json.dumps({
        "status": "ok",
        "output": str(args.output.resolve()),
        "labels": labels,
        "image_array_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
