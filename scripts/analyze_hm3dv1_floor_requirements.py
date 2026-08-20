#!/usr/bin/env python3
"""Classify HM3Dv1 episodes by required vertical transition, then score results.

This is an offline oracle-side audit. It reads dataset goal viewpoints and
completed result JSON files; none of its outputs are exposed to the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from habitat.datasets import make_dataset

from zson3.runtime.hm3d import build_hm3d_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--floor-height-threshold", type=float, default=1.5)
    parser.add_argument("--expected-episodes", type=int, default=2000)
    return parser.parse_args()


def normalized_scene(scene_id: str) -> str:
    return Path(scene_id).stem.removesuffix(".basis")


def episode_floor_record(episode, *, threshold: float, index: int) -> dict:
    start_y = float(episode.start_position[1])
    viewpoint_ys = [
        float(viewpoint.agent_state.position[1])
        for goal in episode.goals
        for viewpoint in (goal.view_points or [])
    ]
    source = "goal_view_points"
    if not viewpoint_ys:
        viewpoint_ys = [float(goal.position[1]) for goal in episode.goals]
        source = "goal_positions_fallback"
    if not viewpoint_ys:
        raise RuntimeError(
            f"Episode has no goal heights: {episode.scene_id}/{episode.episode_id}"
        )
    min_vertical_gap = min(abs(goal_y - start_y) for goal_y in viewpoint_ys)
    requirement = (
        "cross_floor_required"
        if min_vertical_gap > threshold
        else "same_floor_available"
    )
    return {
        "index": index,
        "scene": normalized_scene(episode.scene_id),
        "episode_id": str(episode.episode_id),
        "target": episode.object_category,
        "requirement": requirement,
        "start_y_m": start_y,
        "min_goal_viewpoint_vertical_gap_m": min_vertical_gap,
        "goal_count": len(episode.goals),
        "goal_viewpoint_count": len(viewpoint_ys),
        "height_source": source,
    }


def aggregate(records: list[dict], results_by_identity: dict[tuple[str, str], dict]) -> dict:
    groups = {}
    for group in ("all", "same_floor_available", "cross_floor_required"):
        selected = [
            record
            for record in records
            if group == "all" or record["requirement"] == group
        ]
        results = [
            results_by_identity[(record["scene"], record["episode_id"])]
            for record in selected
            if (record["scene"], record["episode_id"]) in results_by_identity
        ]
        count = len(results)
        groups[group] = {
            "dataset_episodes": len(selected),
            "dataset_fraction": len(selected) / len(records),
            "evaluated_episodes": count,
            "successes": int(
                sum(float(item["metrics"].get("success", 0.0)) for item in results)
            ),
            "sr": (
                sum(float(item["metrics"].get("success", 0.0)) for item in results)
                / count
                if count
                else None
            ),
            "spl": (
                sum(float(item["metrics"].get("spl", 0.0)) for item in results)
                / count
                if count
                else None
            ),
            "successes_at_1m": int(
                sum(
                    float(item["metrics"].get("success_at_1m", 0.0))
                    for item in results
                )
            ),
            "sr_at_1m": (
                sum(
                    float(item["metrics"].get("success_at_1m", 0.0))
                    for item in results
                )
                / count
                if count
                else None
            ),
            "spl_at_1m": (
                sum(
                    float(item["metrics"].get("spl_at_1m", 0.0))
                    for item in results
                )
                / count
                if count
                else None
            ),
            "exceptions": sum(item.get("status") != "ok" for item in results),
        }
    return groups


def fmt_metric(value: float | None) -> str:
    return "pending" if value is None else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.floor_height_threshold) or args.floor_height_threshold <= 0:
        raise ValueError("--floor-height-threshold must be finite and positive")

    config = build_hm3d_config(seed=args.seed, top_down_map=False)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    episodes = list(dataset.episodes)
    if len(episodes) != args.expected_episodes:
        raise RuntimeError(
            f"Expected {args.expected_episodes} HM3Dv1 episodes, found {len(episodes)}"
        )
    records = [
        episode_floor_record(
            episode, threshold=args.floor_height_threshold, index=index
        )
        for index, episode in enumerate(episodes)
    ]
    identities = [(record["scene"], record["episode_id"]) for record in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("HM3Dv1 floor audit found duplicate episode identities")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "hm3dv1_goal_viewpoint_vertical_gap_v1",
        "policy_visible": False,
        "definition": (
            "cross_floor_required iff every annotated success viewpoint has an "
            "absolute vertical gap from the episode start greater than the threshold"
        ),
        "floor_height_threshold_m": args.floor_height_threshold,
        "episode_count": len(records),
        "records": records,
    }
    (args.output_dir / "floor_requirements.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    results_by_identity = {}
    for path in sorted((args.output_dir / "episodes").glob("*.json")):
        result = json.loads(path.read_text())
        identity = (normalized_scene(result["scene"]), str(result["episode_id"]))
        if identity in results_by_identity:
            raise RuntimeError(f"Duplicate completed result identity: {identity}")
        results_by_identity[identity] = result
    unknown = set(results_by_identity) - set(identities)
    if unknown:
        raise RuntimeError(f"Results contain identities outside HM3Dv1: {sorted(unknown)[:5]}")

    summary = {
        "protocol": manifest["protocol"],
        "policy_visible": False,
        "floor_height_threshold_m": args.floor_height_threshold,
        "groups": aggregate(records, results_by_identity),
    }
    (args.output_dir / "floor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        f"protocol={summary['protocol']}",
        "policy_visible=false",
        f"floor_height_threshold_m={args.floor_height_threshold:.3f}",
    ]
    for name, group in summary["groups"].items():
        lines.append(
            f"{name}: dataset={group['dataset_episodes']} "
            f"fraction={group['dataset_fraction']:.6f} "
            f"evaluated={group['evaluated_episodes']} "
            f"successes={group['successes']} sr={fmt_metric(group['sr'])} "
            f"spl={fmt_metric(group['spl'])} "
            f"successes_at_1m={group['successes_at_1m']} "
            f"sr_at_1m={fmt_metric(group['sr_at_1m'])} "
            f"spl_at_1m={fmt_metric(group['spl_at_1m'])} "
            f"exceptions={group['exceptions']}"
        )
    text = "\n".join(lines) + "\n"
    (args.output_dir / "floor_summary.txt").write_text(text)
    print("[ZSON3 FLOOR AUDIT] " + lines[3])
    print("[ZSON3 FLOOR AUDIT] " + lines[4])
    print("[ZSON3 FLOOR AUDIT] " + lines[5])


if __name__ == "__main__":
    main()
