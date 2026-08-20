#!/usr/bin/env python3
"""Capture the fixed HM3Dv1 RGB-D/pose input used by migration gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import habitat
import numpy as np

from zson3.runtime.habitat_sensors import camera_extrinsic, pinhole_intrinsic
from zson3.runtime.hm3d import build_hm3d_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="6s7QHgap2fW")
    parser.add_argument("--episode-id", default="0")
    parser.add_argument(
        "--turns", type=int, default=0, help="Deterministic 30-degree right turns after reset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_reset.npz"),
    )
    args = parser.parse_args()

    config = build_hm3d_config(scene=args.scene, top_down_map=False)
    env = habitat.Env(config=config)
    try:
        observation = env.reset()
        episode = env.current_episode
        if str(episode.episode_id) != args.episode_id:
            raise RuntimeError(
                f"Expected episode {args.episode_id}, got {episode.episode_id}"
            )
        for _ in range(args.turns):
            observation = env.step("turn_right")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            rgb=observation["rgb"],
            depth=observation["depth"].squeeze(-1),
            gps=observation["gps"],
            compass=observation["compass"],
            camera_intrinsic=pinhole_intrinsic(config),
            camera_extrinsic=camera_extrinsic(env),
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(args.output.resolve()),
                    "scene": args.scene,
                    "episode_id": str(episode.episode_id),
                    "target": episode.object_category,
                    "turns": args.turns,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
