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
from habitat.datasets import make_dataset

from nav.pointnav_agent import PointnavAgent
from utils.frontier_utils import read_config_yaml
from zson3.runtime.hm3d import build_hm3d_config
from zson3.runtime.metrics import success_spl_at_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="6s7QHgap2fW")
    parser.add_argument("--episode-id", default="0")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-time", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260727)
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
    parser.add_argument(
        "--apex-target-diagnostics",
        action="store_true",
        help="Save per-step RGB, detector/SAM/semantic overlays, masks and pose/fusion metadata",
    )
    parser.add_argument("--log-level", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.write_path = str(args.output_dir / "navigation_state.json")

    config = build_hm3d_config(
        scene=args.scene,
        seed=args.seed,
        top_down_map=args.render_video,
    )
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    requested_scene = Path(args.scene).stem.removesuffix(".basis")
    selected = [
        episode
        for episode in dataset.episodes
        if Path(episode.scene_id).stem.removesuffix(".basis") == requested_scene
        and str(episode.episode_id) == args.episode_id
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one HM3Dv1 episode for scene={requested_scene} "
            f"episode_id={args.episode_id}, found {len(selected)}"
        )
    dataset.episodes = selected
    openfrontier_config = read_config_yaml(args.config)
    env = habitat.Env(config=config, dataset=dataset)
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
        if env.episode_over and reason == "continue_navigation":
            reason = "max_steps_reached"

    except BaseException as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        reason = "exception"
    finally:
        elapsed = time.perf_counter() - started
        metrics = dict(env.get_metrics()) if env.current_episode is not None else {}
        metrics.update(
            success_spl_at_distance(env, metrics, success_distance=1.0)
        )
        if not metrics.get("success") and reason == "object_found":
            reason = (
                "object_found_at_1m_only"
                if metrics.get("success_at_1m")
                else "false_positive"
            )
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
        print(
            "[ZSON3 FIXED SUMMARY] "
            f"success={float(metrics.get('success', 0.0)):.0f} "
            f"spl={float(metrics.get('spl', 0.0)):.4f} "
            f"SR@1m={float(metrics.get('success_at_1m', 0.0)) * 100:.2f}% "
            f"SPL@1m={float(metrics.get('spl_at_1m', 0.0)):.4f}",
            flush=True,
        )

        if agent is not None:
            if args.render_video and agent.video_frames:
                agent.save_trajectory(args.output_dir)
            agent.close()
        env.close()

        if args.apex_target_diagnostics:
            _write_apex_evidence_video(args.output_dir)

    if error is not None:
        raise SystemExit(1)


def _write_apex_evidence_video(output_dir: Path) -> None:
    """Stream saved evidence frames into a compact review video."""
    frames = sorted(
        (output_dir / "apex_target_replay" / "frames").glob("*_evidence.jpg")
    )
    if not frames:
        return
    import imageio.v2 as imageio

    video_path = output_dir / "apex_target_replay" / "evidence.mp4"
    with imageio.get_writer(video_path, fps=8) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))


if __name__ == "__main__":
    main()
