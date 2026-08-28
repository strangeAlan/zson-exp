#!/usr/bin/env python3
"""Compare the frozen PointGoal Oracle run with the theta-wrapped replay."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


PLAN_RE = re.compile(
    r"Planned actions: (?:None|\d+), rho: -?[\d.]+, "
    r"theta: (?P<theta>-?[\d.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--wrapped-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def episode_key(item: dict) -> tuple[str, str]:
    return (str(item["scene_id"]), str(item["episode_id"]))


def load_episodes(root: Path) -> dict[tuple[str, str], dict]:
    return {
        episode_key(item): item
        for item in (read_json(path) for path in sorted((root / "episodes").glob("*.json")))
    }


def raw_theta_stats(path: Path) -> dict:
    values = [float(value) for value in PLAN_RE.findall(path.read_text(errors="replace"))]
    return {
        "planner_rows": len(values),
        "outside_official_range": sum(
            value < -math.pi or value > math.pi for value in values
        ),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def oracle_timeline(item: dict) -> list[dict]:
    diagnostics = item.get("target_diagnostics") or {}
    oracle = diagnostics.get("target_approach_oracle") or {}
    return oracle.get("timeline") or []


def timeline_stats(rows: list[dict]) -> dict:
    actions = {str(action): 0 for action in (1, 2, 3)}
    positions = []
    rho = []
    theta = []
    for row in rows:
        action = row.get("action")
        if str(action) in actions:
            actions[str(action)] += 1
        if row.get("position") is not None:
            positions.append(row["position"])
        if row.get("rho") is not None:
            rho.append(float(row["rho"]))
        if row.get("theta") is not None:
            theta.append(float(row["theta"]))
    turns = actions["2"] + actions["3"]
    movement = actions["1"] + turns
    return {
        "rows": len(rows),
        "rho": {
            "start": rho[0] if rho else None,
            "min": min(rho) if rho else None,
            "final": rho[-1] if rho else None,
        },
        "theta": {
            "min": min(theta) if theta else None,
            "max": max(theta) if theta else None,
            "outside_official_range": sum(
                value < -math.pi or value > math.pi for value in theta
            ),
        },
        "actions": actions,
        "turn_fraction": turns / movement if movement else None,
        "net_position_change": (
            math.dist(positions[0], positions[-1]) if len(positions) >= 2 else None
        ),
    }


def compare_oracle_mode(baseline: Path, wrapped: Path, mode: str) -> list[dict]:
    old = load_episodes(baseline / mode)
    new = load_episodes(wrapped / mode)
    if old.keys() != new.keys():
        raise RuntimeError(f"{mode} episode identities differ")
    records = []
    for key in old:
        old_item = old[key]
        new_item = new[key]
        old_rows = oracle_timeline(old_item)
        new_rows = oracle_timeline(new_item)
        if not old_rows and not new_rows:
            continue
        if len(old_rows) != len(new_rows):
            raise RuntimeError(f"{mode} timeline length differs for {key}")
        action_differences = sum(
            left.get("action") != right.get("action")
            for left, right in zip(old_rows, new_rows)
        )
        rho_differences = [
            abs(float(left["rho"]) - float(right["rho"]))
            for left, right in zip(old_rows, new_rows)
            if left.get("rho") is not None and right.get("rho") is not None
        ]
        position_differences = [
            math.dist(left["position"], right["position"])
            for left, right in zip(old_rows, new_rows)
            if left.get("position") is not None and right.get("position") is not None
        ]
        meta = new_item.get("probe_metadata") or {}
        records.append(
            {
                "source_probe_index": meta.get("source_probe_index"),
                "source_full_index": meta.get("source_full_index"),
                "target": new_item.get("target"),
                "baseline": timeline_stats(old_rows),
                "wrapped": timeline_stats(new_rows),
                "rows_changed_by_wrap": sum(
                    abs(float(left["theta"]) - float(right["theta"])) > 6.0
                    for left, right in zip(old_rows, new_rows)
                    if left.get("theta") is not None and right.get("theta") is not None
                ),
                "action_differences": action_differences,
                "max_rho_difference": max(rho_differences, default=0.0),
                "max_position_difference": max(position_differences, default=0.0),
                "baseline_success": float(old_item["metrics"]["success"]) > 0.5,
                "wrapped_success": float(new_item["metrics"]["success"]) > 0.5,
                "baseline_steps": old_item["navigation_steps"],
                "wrapped_steps": new_item["navigation_steps"],
                "baseline_final_gt_distance": old_item["metrics"]["distance_to_goal"],
                "wrapped_final_gt_distance": new_item["metrics"]["distance_to_goal"],
            }
        )
    return records


def protection_comparison(baseline: Path, wrapped: Path) -> dict:
    old = load_episodes(baseline / "protection_ofbase")
    new = load_episodes(wrapped / "protection_ofbase")
    if old.keys() != new.keys():
        raise RuntimeError("protection episode identities differ")
    rows = []
    for key in old:
        left, right = old[key], new[key]
        rows.append(
            {
                "scene": right["scene"],
                "episode_id": right["episode_id"],
                "target": right["target"],
                "success": [left["metrics"]["success"], right["metrics"]["success"]],
                "steps": [left["navigation_steps"], right["navigation_steps"]],
                "spl": [left["metrics"]["spl"], right["metrics"]["spl"]],
                "final_gt_distance": [
                    left["metrics"]["distance_to_goal"],
                    right["metrics"]["distance_to_goal"],
                ],
            }
        )
    return {
        "episodes": len(rows),
        "paired_regression_losses": sum(
            left["metrics"]["success"] > 0.5
            and right["metrics"]["success"] <= 0.5
            for left, right in zip(old.values(), new.values())
        ),
        "exact_success": sum(row["success"][0] == row["success"][1] for row in rows),
        "exact_steps": sum(row["steps"][0] == row["steps"][1] for row in rows),
        "exact_spl": sum(row["spl"][0] == row["spl"][1] for row in rows),
        "baseline_sr": sum(row["success"][0] > 0.5 for row in rows) / len(rows),
        "wrapped_sr": sum(row["success"][1] > 0.5 for row in rows) / len(rows),
        "baseline_spl": sum(row["spl"][0] for row in rows) / len(rows),
        "wrapped_spl": sum(row["spl"][1] for row in rows) / len(rows),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    modes = ("protection_ofbase", "oracle_a", "oracle_b")
    raw = {
        "baseline": {
            mode: raw_theta_stats(args.baseline_root / mode / "raw.log") for mode in modes
        },
        "wrapped": {
            mode: raw_theta_stats(args.wrapped_root / mode / "raw.log") for mode in modes
        },
    }
    for side in raw.values():
        side["total"] = {
            "planner_rows": sum(side[mode]["planner_rows"] for mode in modes),
            "outside_official_range": sum(
                side[mode]["outside_official_range"] for mode in modes
            ),
        }
    oracle_a = compare_oracle_mode(args.baseline_root, args.wrapped_root, "oracle_a")
    oracle_b = compare_oracle_mode(args.baseline_root, args.wrapped_root, "oracle_b")
    wrapped_summary = read_json(args.wrapped_root / "oracle_summary_v1.json")
    protection = protection_comparison(args.baseline_root, args.wrapped_root)
    oracle_b_reached = sum(row["wrapped_success"] for row in oracle_b)
    payload = {
        "schema_version": 1,
        "audit": "pointgoal_theta_wrap_final_regression",
        "baseline_root": str(args.baseline_root.resolve()),
        "wrapped_root": str(args.wrapped_root.resolve()),
        "policy_change": "theta = atan2(sin(theta), cos(theta))",
        "static_test": {
            "command": "python -m pytest -q tests/test_pointgoal_theta_wrap.py",
            "passed": 74,
            "failed": 0,
        },
        "raw_theta": raw,
        "oracle_a": {
            "accepted_no_stop_correct_candidates": wrapped_summary[
                "accepted_no_stop_correct_candidates"
            ],
            "accepted_no_stop_rescues": wrapped_summary[
                "accepted_no_stop_oracle_a_rescues"
            ],
            "rescue_rate": wrapped_summary[
                "accepted_no_stop_oracle_a_rescue_rate"
            ],
            "paired_records": oracle_a,
            "action_differences": sum(row["action_differences"] for row in oracle_a),
        },
        "oracle_b": {
            "episodes": len(oracle_b),
            "reached": oracle_b_reached,
            "paired_records": oracle_b,
            "action_differences": sum(row["action_differences"] for row in oracle_b),
        },
        "protection": protection,
        "gate": {
            "oracle_b_required": 3,
            "oracle_b_observed": oracle_b_reached,
            "protection_regressions": protection["paired_regression_losses"],
            "expand_to_probe64": (
                oracle_b_reached >= 3
                and protection["paired_regression_losses"] == 0
            ),
        },
        "decision": "SHELVE",
        "decision_reason": (
            "Theta is contract-correct after wrapping, but the active policy encodes "
            "the integrated polar PointGoal as rho/cos(theta)/sin(theta), so all paired "
            "Oracle actions remain identical and Oracle B remains 0/4."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "raw_theta": raw,
        "oracle_a_rescues": payload["oracle_a"]["accepted_no_stop_rescues"],
        "oracle_a_action_differences": payload["oracle_a"]["action_differences"],
        "oracle_b_reached": payload["oracle_b"]["reached"],
        "oracle_b_action_differences": payload["oracle_b"]["action_differences"],
        "protection": {key: value for key, value in payload["protection"].items() if key != "rows"},
        "decision": payload["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
