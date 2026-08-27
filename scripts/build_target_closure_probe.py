#!/usr/bin/env python3
"""Freeze the HM3Dv2 TargetClosure paired Failure/Regression ProbeSet.

Selection is derived only from the frozen OF-base 1000-episode run.  The
resulting manifest carries the original episode index and cohort so every
probe entry can be located exactly in the baseline logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "results/openfrontier_base_sam3_full_hm3dv2_1000_seed20260727"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "config/evaluation/hm3dv2_target_closure_probe64.json"

COHORT_QUOTAS = {
    "failure_false_commit_no_gt": 12,
    "failure_false_commit_gt_visible": 12,
    "failure_accepted_no_stop": 8,
    "regression_path_exhausted_far_centroid": 12,
    "regression_path_exhausted_near_centroid": 8,
    "regression_distance_stop": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_pixels_near_verification(diagnostics: dict, step: int) -> int:
    by_step = {
        int(event["step"]): int(event.get("target_pixels", 0))
        for event in diagnostics.get("visibility_events", [])
    }
    # OF-base verifies the six latest RGB frames.  The evaluator-only semantic
    # trace is sampled on the same navigation steps.
    return max((by_step.get(index, 0) for index in range(step - 5, step + 1)), default=0)


def classify(result: dict) -> str | None:
    diagnostics = result.get("target_diagnostics") or {}
    accepted = [
        event
        for event in diagnostics.get("verification_events", [])
        if event.get("accepted")
    ]
    termination = diagnostics.get("termination_event")
    success = bool(result["metrics"].get("success", 0.0))

    if not success and termination is not None and accepted:
        pixels = target_pixels_near_verification(
            diagnostics, int(accepted[-1]["step"])
        )
        return (
            "failure_false_commit_gt_visible"
            if pixels >= 100
            else "failure_false_commit_no_gt"
        )
    if not success and accepted and termination is None:
        return "failure_accepted_no_stop"
    if success and termination is not None and termination.get("path_exhausted"):
        return (
            "regression_path_exhausted_far_centroid"
            if float(termination.get("distance_to_object", 0.0)) >= 1.0
            else "regression_path_exhausted_near_centroid"
        )
    if success and termination is not None:
        return "regression_distance_stop"
    return None


def stratified_pick(rows: list[dict], count: int) -> list[dict]:
    """Round-robin targets, preferring a new scene and alternating lengths."""
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)
    for candidates in by_target.values():
        candidates.sort(key=lambda row: (row["navigation_steps"], row["index"]))
        # Interleave short and long trajectories.
        ordered = []
        while candidates:
            ordered.append(candidates.pop(0))
            if candidates:
                ordered.append(candidates.pop(-1))
        candidates[:] = ordered

    picked: list[dict] = []
    used_scenes: set[str] = set()
    target_order = sorted(by_target, key=lambda target: (len(by_target[target]), target))
    while len(picked) < count:
        progressed = False
        for target in target_order:
            candidates = by_target[target]
            if not candidates:
                continue
            fresh_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate["scene"] not in used_scenes
                ),
                0,
            )
            candidate = candidates.pop(fresh_index)
            picked.append(candidate)
            used_scenes.add(candidate["scene"])
            progressed = True
            if len(picked) == count:
                break
        if not progressed:
            raise RuntimeError(f"Only found {len(picked)} of {count} requested rows")
    return picked


def main() -> None:
    args = parse_args()
    manifest_path = args.baseline / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    source_selection = source_manifest["selection"]

    cohorts: dict[str, list[dict]] = defaultdict(list)
    for result_path in sorted((args.baseline / "episodes").glob("*.json")):
        result = json.loads(result_path.read_text())
        cohort = classify(result)
        if cohort is None:
            continue
        diagnostics = result.get("target_diagnostics") or {}
        accepted = [
            event
            for event in diagnostics.get("verification_events", [])
            if event.get("accepted")
        ]
        termination = diagnostics.get("termination_event")
        row = {
            "index": int(result["index"]),
            "scene": result["scene"],
            "episode_id": str(result["episode_id"]),
            "target": result["target"],
            "navigation_steps": int(result["navigation_steps"]),
            "distance_to_goal": float(result["metrics"]["distance_to_goal"]),
            "target_pixels_in_verification_window": (
                target_pixels_near_verification(
                    diagnostics, int(accepted[-1]["step"])
                )
                if accepted
                else 0
            ),
            "centroid_distance_at_stop": (
                float(termination["distance_to_object"])
                if termination is not None
                else None
            ),
            "cohort": cohort,
        }
        cohorts[cohort].append(row)

    selected = []
    cohort_counts = {}
    for cohort, quota in COHORT_QUOTAS.items():
        picked = stratified_pick(cohorts[cohort], quota)
        cohort_counts[cohort] = len(picked)
        for row in picked:
            original = dict(source_selection[row["index"]])
            original["source_target"] = original.get(
                "source_target", original.get("target")
            )
            original.update(
                {
                    "source_index": row["index"],
                    "probe_cohort": cohort,
                    "baseline_success": cohort.startswith("regression_"),
                    "baseline_reason": (
                        "object_found"
                        if cohort.startswith("regression_")
                        else "target_closure_failure"
                    ),
                    "baseline_navigation_steps": row["navigation_steps"],
                    "baseline_distance_to_goal": row["distance_to_goal"],
                    "baseline_target_pixels_in_verification_window": row[
                        "target_pixels_in_verification_window"
                    ],
                    "baseline_centroid_distance_at_stop": row[
                        "centroid_distance_at_stop"
                    ],
                }
            )
            selected.append(original)

    payload = {
        "metadata": {
            "name": "hm3dv2_target_closure_probe64",
            "source_run": str(args.baseline.resolve()),
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": sha256(manifest_path),
            "identity_guarantee": (
                "exact OF-base full-1000 (scene_id, episode_id, source_index); "
                "32 target-closure failures followed by 32 baseline successes"
            ),
            "selection_policy": (
                "fixed cohort quotas; target round-robin; prefer distinct scenes; "
                "alternate short/long baseline trajectories"
            ),
            "gt_visibility_note": (
                "target pixel counts are evaluator-only annotations and never policy input"
            ),
            "cohort_counts": cohort_counts,
            "success_criteria": {
                "failure_rescues_at_least": 6,
                "regression_losses_at_most": 3,
                "paired_net_gain_positive": True,
                "common_success_median_steps_increase_at_most": 0.15,
            },
        },
        "selection": selected,
    }
    if len(selected) != 64:
        raise RuntimeError(f"Expected 64 episodes, got {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(selected)} episodes to {args.output}")
    print(json.dumps(cohort_counts, indent=2))


if __name__ == "__main__":
    main()
