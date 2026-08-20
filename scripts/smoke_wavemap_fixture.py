#!/usr/bin/env python3
"""Exercise OpenFrontier's Wavemap integration on the fixed RGB-D fixture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pywavemap import InterpolationMode

from mapping.wavemap import WaveMapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_turn6.npz"),
    )
    args = parser.parse_args()
    fixture = np.load(args.fixture)
    intrinsic = fixture["camera_intrinsic"]
    depth = fixture["depth"]
    params = {
        "min_cell_width": 0.05,
        "width": int(depth.shape[1]),
        "height": int(depth.shape[0]),
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "min_range": 0.05,
        "max_range": 3.5,
        "resolution": 0.1,
    }
    mapper = WaveMapper(params=params)
    try:
        started = time.perf_counter()
        mapper.insert_depth_to_buffer(
            depth=depth, transform=np.linalg.inv(fixture["camera_extrinsic"])
        )
        mapper.integrate_from_buffer()
        integration_seconds = time.perf_counter() - started

        # Query a bounded local lattice for the component gate. The unchanged
        # full OpenFrontier occupancy interpolation remains an episode-gate test.
        camera_position = np.linalg.inv(fixture["camera_extrinsic"])[:3, 3]
        offsets = np.stack(
            np.meshgrid(
                np.linspace(-2.0, 2.0, 41),
                np.linspace(-2.0, 2.0, 41),
                np.linspace(-1.0, 1.0, 21),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        query = offsets + camera_position
        odds = np.asarray(
            mapper.map.interpolate(query, InterpolationMode.NEAREST)
        ).reshape(-1)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "integration_seconds": integration_seconds,
                    "query_points": int(query.shape[0]),
                    "known_points": int(np.count_nonzero(np.abs(odds) > 1e-5)),
                    "occupied_points": int(np.count_nonzero(odds > 0.6)),
                    "free_points": int(np.count_nonzero(odds < -1e-5)),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        mapper.close()


if __name__ == "__main__":
    main()
