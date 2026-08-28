#!/usr/bin/env python3
"""Build Oracle-B replay set and finalize the Target Approach ceiling audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-a", type=Path, required=True)
    parser.add_argument("--oracle-b", type=Path)
    parser.add_argument("--protection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-b", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def episode_results(run: Path) -> dict[int, dict]:
    results = {}
    for path in sorted((run / "episodes").glob("*.json")):
        item = read_json(path)
        metadata = item.get("probe_metadata") or {}
        source_index = int(metadata.get("source_probe_index", -1))
        if source_index < 0:
            raise RuntimeError(f"Missing source_probe_index: {path}")
        results[source_index] = item
    return results


def oracle(item: dict) -> dict:
    target = item.get("target_diagnostics") or {}
    return target.get("target_approach_oracle") or {}


def candidate_class(item: dict) -> str:
    return str((oracle(item).get("candidate_audit") or {}).get("classification", "missing"))


def metric_success(item: dict | None) -> bool:
    return bool(item and float(item.get("metrics", {}).get("success", 0.0)) > 0.5)


def prepare_b(args, manifest):
    a_results = episode_results(args.oracle_a)
    selection = []
    for entry in manifest["selection"]:
        source_index = int(entry["source_probe_index"])
        item = a_results.get(source_index)
        if item is None:
            continue
        if candidate_class(item) == "correct" and not metric_success(item):
            selection.append(entry)
    payload = {
        "metadata": {
            "name": "hm3dv2_approach_oracle_b_failures",
            "derived_from": str(args.oracle_a.resolve()),
            "derivation_rule": "Oracle-A metric failure with candidate classification=correct",
            "result_dependent_parameter_tuning": False,
        },
        "selection": selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"oracle_b_episodes={len(selection)} path={args.output}")


def compact_execution(item: dict | None) -> dict | None:
    if item is None:
        return None
    diag = oracle(item)
    timeline = diag.get("timeline") or []
    rho = [float(row["rho"]) for row in timeline if row.get("rho") is not None]
    endpoint_dist = [
        float(row["endpoint_distance"])
        for row in timeline
        if row.get("endpoint_distance") is not None
    ]
    actions = [row.get("action") for row in timeline]
    counts = {str(code): actions.count(code) for code in (1, 2, 3, None)}
    selected = diag.get("selected_endpoint")
    return {
        "success": metric_success(item),
        "reason": item.get("reason"),
        "navigation_steps": item.get("navigation_steps"),
        "final_gt_distance": item.get("metrics", {}).get("distance_to_goal"),
        "candidate_classification": candidate_class(item),
        "selected_endpoint": selected,
        "candidate_viewpoint_count": len(diag.get("candidate_viewpoints") or []),
        "rho": {
            "start": rho[0] if rho else None,
            "min": min(rho) if rho else None,
            "final": rho[-1] if rho else None,
        },
        "endpoint_distance": {
            "start": endpoint_dist[0] if endpoint_dist else None,
            "min": min(endpoint_dist) if endpoint_dist else None,
            "final": endpoint_dist[-1] if endpoint_dist else None,
        },
        "actions": counts,
        "post_accept_steps": len(timeline),
        "oracle_outcome": diag.get("outcome"),
        "artifacts": (diag.get("candidate_audit") or {}).get("artifacts"),
        "pursuit_video": diag.get("pursuit_video"),
    }


def first_acceptance(item: dict):
    events = (item.get("target_diagnostics") or {}).get("verification_events") or []
    for event in events:
        if event.get("accepted"):
            return {
                "step": event.get("step"),
                "centroid": event.get("object_centroid"),
                "probability": event.get("probability"),
            }
    return None


def source_full_result(source_root: Path, source_index: int) -> dict:
    matches = list((source_root / "episodes").glob(f"{source_index:03d}_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected source result {source_index}, found {matches}")
    return read_json(matches[0])


def finalize(args, manifest):
    if args.oracle_b is None or args.protection is None:
        raise ValueError("--oracle-b and --protection are required for finalization")
    a_results = episode_results(args.oracle_a)
    b_results = episode_results(args.oracle_b) if (args.oracle_b / "episodes").exists() else {}
    p_results = episode_results(args.protection)
    entries = {int(item["source_probe_index"]): item for item in manifest["selection"]}

    source_root = Path(
        "/home/hsy/zson-exp-ofbase-hm3dv2/results/"
        "openfrontier_base_sam3_full_hm3dv2_1000_seed20260727"
    )
    episodes = []
    taxonomy = {letter: 0 for letter in "ABCDE"}
    correct_failures = 0
    a_rescued_failures = 0
    accepted_correct = 0
    accepted_a_rescued = 0
    for source_index, entry in entries.items():
        if entry["oracle_cohort"] != "diagnostic":
            continue
        a_item = a_results.get(source_index)
        if a_item is None:
            raise RuntimeError(f"Missing Oracle-A result for source probe {source_index}")
        b_item = b_results.get(source_index)
        classification = candidate_class(a_item)
        base_failure = not bool(entry["baseline_success"])
        primary = None
        if base_failure:
            if classification in {"wrong", "ambiguous", "missing"}:
                primary = "A"
            elif metric_success(a_item):
                primary = "B"
            elif b_item is None:
                primary = "E"
            elif candidate_class(b_item) != "correct":
                primary = "E"
            elif metric_success(b_item):
                primary = "C"
            elif str(b_item.get("reason", "")).startswith("exception") or "no_reachable" in str(oracle(b_item).get("outcome")):
                primary = "E"
            else:
                primary = "D"
            taxonomy[primary] += 1
            if classification == "correct":
                correct_failures += 1
                a_rescued_failures += int(metric_success(a_item))
        is_accepted_no_stop = "accepted_no_stop" in entry["evidence_role"]
        if is_accepted_no_stop and classification == "correct":
            accepted_correct += 1
            accepted_a_rescued += int(metric_success(a_item))
        episodes.append(
            {
                "source_probe_index": source_index,
                "scene": Path(entry["scene_id"]).parent.name,
                "episode_id": entry["episode_id"],
                "target": entry["target"],
                "evidence_role": entry["evidence_role"],
                "baseline_success": entry["baseline_success"],
                "candidate_audit": oracle(a_item).get("candidate_audit"),
                "oracle_a": compact_execution(a_item),
                "oracle_b": compact_execution(b_item),
                "primary_attribution": primary,
            }
        )

    protection = []
    protection_losses = 0
    exact_success = 0
    exact_reason = 0
    exact_steps = 0
    exact_acceptance = 0
    exact_spl = 0
    for source_index, entry in entries.items():
        if entry["oracle_cohort"] != "protection":
            continue
        replay = p_results.get(source_index)
        if replay is None:
            raise RuntimeError(f"Missing protection result {source_index}")
        source = source_full_result(source_root, int(entry["source_full_index"]))
        replay_success = metric_success(replay)
        source_success = metric_success(source)
        protection_losses += int(source_success and not replay_success)
        exact_success += int(replay_success == source_success)
        exact_reason += int(replay.get("reason") == source.get("reason"))
        exact_steps += int(replay.get("navigation_steps") == source.get("navigation_steps"))
        exact_acceptance += int(first_acceptance(replay) == first_acceptance(source))
        exact_spl += int(
            replay.get("metrics", {}).get("spl")
            == source.get("metrics", {}).get("spl")
        )
        protection.append(
            {
                "source_probe_index": source_index,
                "target": entry["target"],
                "source": {
                    "success": source_success,
                    "reason": source.get("reason"),
                    "steps": source.get("navigation_steps"),
                    "first_acceptance": first_acceptance(source),
                },
                "replay": {
                    "success": replay_success,
                    "reason": replay.get("reason"),
                    "steps": replay.get("navigation_steps"),
                    "first_acceptance": first_acceptance(replay),
                },
            }
        )

    accepted_rate = accepted_a_rescued / accepted_correct if accepted_correct else 0.0
    go = accepted_correct > 0 and accepted_rate >= 0.5 and protection_losses == 0
    summary = {
        "schema_version": 1,
        "experiment": "Target Approach Oracle Ceiling",
        "frozen_baseline_commit": manifest["metadata"]["frozen_baseline_commit"],
        "decision": "GO" if go else "NO-GO",
        "decision_rule": "Oracle A rescues >=50% of correct-candidate accepted-no-STOP failures and protection loss=0",
        "candidate_correct_failures": correct_failures,
        "oracle_a_rescued_failures": a_rescued_failures,
        "oracle_a_rescue_rate_all_correct_failures": (
            a_rescued_failures / correct_failures if correct_failures else 0.0
        ),
        "accepted_no_stop_correct_candidates": accepted_correct,
        "accepted_no_stop_oracle_a_rescues": accepted_a_rescued,
        "accepted_no_stop_oracle_a_rescue_rate": accepted_rate,
        "oracle_b_additional_rescues": taxonomy["C"],
        "pointnav_failures": taxonomy["D"],
        "taxonomy": taxonomy,
        "taxonomy_labels": {
            "A": "wrong / ambiguous target candidate",
            "B": "candidate correct, Oracle A success",
            "C": "Oracle A failed, GT-viewpoint Oracle B success",
            "D": "GT-viewpoint Oracle B still failed",
            "E": "data, replay, or navmesh anomaly",
        },
        "protection": {
            "episodes": len(protection),
            "losses": protection_losses,
            "exact_success": exact_success,
            "exact_reason": exact_reason,
            "exact_steps": exact_steps,
            "exact_first_acceptance": exact_acceptance,
            "exact_spl": exact_spl,
        },
        "episodes": episodes,
        "protection_episodes": protection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "decision", "accepted_no_stop_correct_candidates",
        "accepted_no_stop_oracle_a_rescues", "accepted_no_stop_oracle_a_rescue_rate",
        "oracle_b_additional_rescues", "pointnav_failures", "taxonomy", "protection"
    )}, indent=2))


def main():
    args = parse_args()
    manifest = read_json(args.manifest)
    if args.prepare_b:
        prepare_b(args, manifest)
    else:
        finalize(args, manifest)


if __name__ == "__main__":
    main()
