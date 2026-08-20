#!/usr/bin/env python3
"""Instantiate the frozen HM3Dv1 runtime without loading algorithm models."""

from __future__ import annotations

import argparse
import json

import habitat

from zson3.runtime.hm3d import HM3DProtocol, build_hm3d_config, hm3d_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="6s7QHgap2fW")
    parser.add_argument("--episode-id", default="0")
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_hm3d_config(scene=args.scene, top_down_map=False)
    env = habitat.Env(config=config)
    try:
        observation = env.reset()
        if str(env.current_episode.episode_id) != args.episode_id:
            raise RuntimeError(
                "Fixed smoke episode drifted: expected "
                f"{args.episode_id}, got {env.current_episode.episode_id}"
            )
        for _ in range(args.steps):
            observation = env.step("turn_right")

        episode = env.current_episode
        payload = {
            "status": "ok",
            "protocol": HM3DProtocol().to_dict(),
            "paths": {key: str(value) for key, value in hm3d_paths().items()},
            "episode": {
                "id": str(episode.episode_id),
                "scene_id": episode.scene_id,
                "target": episode.object_category,
            },
            "observations": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in observation.items()
                if hasattr(value, "shape")
            },
            "metrics": sorted(env.get_metrics()),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
