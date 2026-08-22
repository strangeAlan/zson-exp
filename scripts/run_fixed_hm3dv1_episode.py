#!/usr/bin/env python3
"""Run one reproducible HM3Dv1 episode through the migrated OpenFrontier stack."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import habitat

from nav.pointnav_agent import PointnavAgent
from utils.frontier_utils import read_config_yaml
from zson3.runtime.hm3d import build_hm3d_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="6s7QHgap2fW")
    parser.add_argument("--episode-id", default="0")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-time", type=int, default=3600)
    parser.add_argument(
        "--config",
        default="config/zson3/navigation_hm3dv1_qwen.yaml",
    )
    parser.add_argument(
        "--unet-weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/hm3dv1_6s7QHgap2fW_ep0"),
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--log-level", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.write_path = str(args.output_dir / "navigation_state.json")

    config = build_hm3d_config(
        scene=args.scene,
        seed=0,
        top_down_map=args.render_video,
    )
    openfrontier_config = read_config_yaml(args.config)
    env = habitat.Env(config=config)
    agent = None
    started = time.perf_counter()
    reason = "not_started"
    error = None

    try:
        env.reset()
        episode = env.current_episode
        if str(episode.episode_id) != args.episode_id:
            raise RuntimeError(
                f"Fixed episode drifted: expected {args.episode_id}, "
                f"got {episode.episode_id}"
            )

        agent = PointnavAgent(
            env,
            args,
            save_dir=str(args.output_dir),
            openfrontier_config=openfrontier_config,
            habitat_config=config,
            scene=args.scene,
        )
        agent.setup_system()
        agent.initialize()

        reason = "continue_navigation"
        while not env.episode_over:
            if time.perf_counter() - started >= args.max_time:
                reason = "max_time_reached"
                env.step("stop")
                break
            navigate, reason = agent.navigation(save_images=args.save_images)
            if args.render_video:
                agent.update_video()
            if not navigate:
                env.step("stop")
                break

    except BaseException as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        reason = "exception"
    finally:
        elapsed = time.perf_counter() - started
        metrics = env.get_metrics() if env.current_episode is not None else {}
        episode = env.current_episode
        payload = {
            "status": "error" if error else "ok",
            "scene": args.scene,
            "episode_id": str(episode.episode_id) if episode else None,
            "target": episode.object_category if episode else None,
            "reason": reason,
            "navigation_steps": agent.navigation_steps if agent else None,
            "elapsed_seconds": elapsed,
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "metrics": metrics,
            "frontiers": len(agent.ft_manager.frontiers) if agent and agent.ft_manager else None,
            "valid_frontiers": (
                len(agent.ft_manager.valid_frontiers)
                if agent and agent.ft_manager
                else None
            ),
            "detected_objects": len(agent.detected_objects) if agent else None,
            "target_diagnostics": (
                agent.get_target_diagnostics() if agent else None
            ),
            "timings": agent.timings if agent else None,
            "error": error,
        }
        result_path = args.output_dir / "result.json"
        result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(json.dumps(payload, indent=2, default=str))

        if agent is not None:
            agent.close()
        env.close()

    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
