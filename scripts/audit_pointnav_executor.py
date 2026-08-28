#!/usr/bin/env python3
"""Read-only audit of OF-base PointNav traces.

The script never imports or executes the policy.  It parses the frozen HM3Dv2
full results and the already-completed Target Approach Oracle artifacts, then
writes a versioned evidence summary.  In particular, it keeps observed defect
exposure separate from counterfactual rescue/headroom.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


PLAN_RE = re.compile(
    r"Planned actions: (?P<action>None|\d+), rho: (?P<rho>-?[\d.]+), "
    r"theta: (?P<theta>-?[\d.]+), forward heat: (?P<fheat>\d+) "
    r"rotation heat: (?P<rheat>\d+)"
)
STEP_RE = re.compile(r"\[AGENT\]\s+\[(\d+)\]")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def accepted_step(item: dict) -> int | None:
    events = (item.get("target_diagnostics") or {}).get("verification_events") or []
    values = [int(event["step"]) for event in events if event.get("accepted")]
    return min(values) if values else None


def failure_stage(item: dict) -> str:
    diag = item.get("target_diagnostics") or {}
    accepted = accepted_step(item) is not None
    stopped = diag.get("termination_event") is not None
    if accepted:
        return "accepted_stop_fail" if stopped else "accepted_no_stop"
    visible = max(
        (int(event.get("target_pixels", 0)) for event in diag.get("visibility_events") or []),
        default=0,
    )
    if visible < 50:
        return "never_visible"
    has_candidate = any(
        int(event.get("mask_count", 0)) > 0
        for event in diag.get("segmentation_events") or []
    )
    return "candidate_no_accept" if has_candidate else "visible_no_candidate"


def parse_log(path: Path) -> list[dict]:
    rows = []
    step = None
    object_lock_active = False
    for line in path.read_text(errors="replace").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            step = int(step_match.group(1))
        if "Object found with probability" in line:
            object_lock_active = True
        match = PLAN_RE.search(line)
        if not match:
            continue
        action_text = match.group("action")
        rows.append(
            {
                "step": step,
                "action": None if action_text == "None" else int(action_text),
                "rho": float(match.group("rho")),
                "theta": float(match.group("theta")),
                "forward_heat": int(match.group("fheat")),
                "rotation_heat": int(match.group("rheat")),
                "object_lock_active": object_lock_active,
            }
        )
    return rows


def trace_stats(rows: list[dict]) -> dict:
    actions = Counter(row["action"] for row in rows)
    rho = [row["rho"] for row in rows]
    theta = [row["theta"] for row in rows]
    out_of_range = [value for value in theta if value < -math.pi or value > math.pi]
    branch_jumps = sum(
        abs(right - left) > 5.5
        for left, right in zip(theta, theta[1:])
    )
    turns = actions[2] + actions[3]
    movement_actions = actions[1] + turns
    return {
        "planner_rows": len(rows),
        "actions": {
            "stop_or_none": actions[None],
            "forward": actions[1],
            "turn_left": actions[2],
            "turn_right": actions[3],
        },
        "turn_fraction": turns / movement_actions if movement_actions else None,
        "rho": {
            "start": rho[0] if rho else None,
            "min": min(rho) if rho else None,
            "final": rho[-1] if rho else None,
            "range": max(rho) - min(rho) if rho else None,
        },
        "theta": {
            "min": min(theta) if theta else None,
            "max": max(theta) if theta else None,
            "outside_official_range": len(out_of_range),
            "branch_cut_jumps": branch_jumps,
        },
        "max_forward_heat": max((row["forward_heat"] for row in rows), default=0),
        "max_rotation_heat": max((row["rotation_heat"] for row in rows), default=0),
    }


def result_files(root: Path) -> list[Path]:
    return sorted((root / "episodes").glob("*.json"))


def full_audit(root: Path) -> dict:
    episodes = []
    exposure = Counter()
    failure_exposure = Counter()
    stage_exposure: dict[str, Counter] = {}
    accepted_failures = []
    strict_stagnation = []
    accepted_post_theta_exposed = 0
    for result_path in result_files(root):
        item = read_json(result_path)
        index = int(item["index"])
        log_matches = list((root / "episode_logs").glob(f"{index:03d}_*/navigation_log.txt"))
        if len(log_matches) != 1:
            raise RuntimeError(f"Expected one log for full index {index}: {log_matches}")
        rows = parse_log(log_matches[0])
        accept = accepted_step(item)
        post = [row for row in rows if row.get("object_lock_active")]
        all_stats = trace_stats(rows)
        post_stats = trace_stats(post)
        success = float(item.get("metrics", {}).get("success", 0.0)) > 0.5
        stage = "success" if success else failure_stage(item)
        exposed = all_stats["theta"]["outside_official_range"] > 0
        exposure["episodes"] += 1
        exposure["exposed"] += int(exposed)
        exposure["rows"] += all_stats["planner_rows"]
        exposure["out_of_range_rows"] += all_stats["theta"]["outside_official_range"]
        stage_exposure.setdefault(stage, Counter())["episodes"] += 1
        stage_exposure[stage]["exposed"] += int(exposed)
        if not success:
            failure_exposure["episodes"] += 1
            failure_exposure["exposed"] += int(exposed)

        record = {
            "index": index,
            "scene": item.get("scene"),
            "episode_id": item.get("episode_id"),
            "target": item.get("target"),
            "success": success,
            "reason": item.get("reason"),
            "final_gt_distance": item.get("metrics", {}).get("distance_to_goal"),
            "collisions_total": (item.get("metrics", {}).get("collisions") or {}).get("count"),
            "accepted_step": accept,
            "stage": stage,
            "all": all_stats,
            "post_accept": post_stats,
        }
        episodes.append(record)
        if stage == "accepted_no_stop":
            accepted_failures.append(record)
            accepted_post_theta_exposed += int(
                post_stats["theta"]["outside_official_range"] > 0
            )
        if (
            not success
            and item.get("reason") in {"max_steps_reached", "robot_stuck"}
            and len(rows) >= 30
        ):
            tail = trace_stats(rows[-50:])
            rho = tail["rho"]
            if (
                tail["turn_fraction"] is not None
                and tail["turn_fraction"] >= 0.4
                and rho["start"] is not None
                and rho["final"] >= rho["start"] - 0.2
            ):
                strict_stagnation.append(
                    {
                        "index": index,
                        "stage": stage,
                        "reason": item.get("reason"),
                        "tail": tail,
                        "endpoint_fixed_or_navmesh_verified": False,
                    }
                )

    return {
        "episodes": len(episodes),
        "successes": sum(item["success"] for item in episodes),
        "failures": sum(not item["success"] for item in episodes),
        "theta_range_exposure": dict(exposure),
        "failure_theta_range_exposure": dict(failure_exposure),
        "stage_theta_range_exposure": {
            key: dict(value) for key, value in sorted(stage_exposure.items())
        },
        "accepted_no_stop": accepted_failures,
        "accepted_no_stop_count": len(accepted_failures),
        "accepted_no_stop_post_accept_theta_exposed": accepted_post_theta_exposed,
        "stagnation_signature_upper_bound": strict_stagnation,
        "stagnation_signature_upper_bound_count": len(strict_stagnation),
    }


def oracle_audit(root: Path) -> dict:
    summary = read_json(root / "oracle_summary_v1.json")
    b_root = root / "oracle_b" / "episodes"
    records = []
    for path in sorted(b_root.glob("*.json")):
        item = read_json(path)
        meta = item.get("probe_metadata") or {}
        oracle = (item.get("target_diagnostics") or {}).get("target_approach_oracle") or {}
        timeline = oracle.get("timeline") or []
        stats = trace_stats(
            [
                {
                    "step": row.get("step"),
                    "action": row.get("action"),
                    "rho": float(row["rho"]),
                    "theta": float(row["theta"]),
                    "forward_heat": int(row.get("forward_heat", 0)),
                    "rotation_heat": int(row.get("rotation_heat", 0)),
                }
                for row in timeline
                if row.get("rho") is not None and row.get("theta") is not None
            ]
        )
        positions = [row.get("position") for row in timeline if row.get("position")]
        endpoint = oracle.get("selected_endpoint") or {}
        fixed = [row.get("fixed_endpoint") for row in timeline if row.get("fixed_endpoint")]
        effective = [row.get("effective_pointnav_goal") for row in timeline if row.get("effective_pointnav_goal")]
        fixed_xy = fixed[0][:2] if fixed else None
        effective_xy_drift = (
            [math.dist(fixed_xy, value[:2]) for value in effective]
            if fixed_xy is not None
            else []
        )
        records.append(
            {
                "source_probe_index": meta.get("source_probe_index"),
                "source_full_index": meta.get("source_full_index"),
                "target": item.get("target"),
                "reason": item.get("reason"),
                "final_gt_distance": item.get("metrics", {}).get("distance_to_goal"),
                "collisions_total": (item.get("metrics", {}).get("collisions") or {}).get("count"),
                "requested_endpoint_fixed": len({tuple(value) for value in fixed}) <= 1,
                "effective_endpoint_unique_xy_count_1mm": len(
                    {tuple(round(axis, 3) for axis in value[:2]) for value in effective}
                ),
                "max_effective_xy_drift_from_requested": (
                    max(effective_xy_drift) if effective_xy_drift else None
                ),
                "navmesh_path_distance": endpoint.get("current_geodesic"),
                "endpoint_to_gt_geodesic": endpoint.get("gt_geodesic"),
                "net_xy_motion": (
                    math.dist(positions[0][:2], positions[-1][:2]) if len(positions) >= 2 else None
                ),
                "trace": stats,
            }
        )
    return {
        "decision": summary.get("decision"),
        "taxonomy": summary.get("taxonomy"),
        "oracle_b_count": len(records),
        "oracle_b_rescues": sum(
            float(item.get("final_gt_distance") or math.inf) <= 1.0 for item in records
        ),
        "oracle_b": records,
    }


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": 1,
        "audit": "pointnav_executor_final_read_only",
        "full_root": str(args.full.resolve()),
        "oracle_root": str(args.oracle.resolve()),
        "full": full_audit(args.full),
        "oracle": oracle_audit(args.oracle),
        "interpretation_guardrails": {
            "theta_exposure_is_not_rescue_headroom": True,
            "full_logs_do_not_record_raw_and_final_actions_separately": True,
            "normal_mode_final_action_equals_raw_policy_action_by_code": True,
            "frontier_endpoint_reachability_not_reconstructable_from_full_logs": True,
            "stagnation_signature_is_an_upper_bound_not_executor_attribution": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote={args.output}")
    print(json.dumps({
        "full": {key: value for key, value in payload["full"].items() if key not in {"accepted_no_stop", "stagnation_signature_upper_bound"}},
        "oracle": payload["oracle"],
    }, indent=2))


if __name__ == "__main__":
    main()
