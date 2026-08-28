#!/usr/bin/env python3
"""Action-level GT follower control for the four frozen Oracle-B endpoints.

This is an analysis utility, not a policy implementation.  It restores the
recorded Oracle-B acceptance pose in Habitat, asks Habitat-Sim's official
GreedyGeodesicFollower for discrete actions to the exact same legal endpoint,
and executes only those actions.  No result is reported as ObjectNav policy
performance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import habitat
import habitat_sim
import numpy as np
from habitat_sim.agent import AgentState
from habitat_sim.nav import GreedyGeodesicFollower

from scripts.run_hm3dv1_random100 import select_manifest_episodes
from utils.transform import to_habitat_position, to_habitat_rotation
from zson3.runtime.datasets import NavigationProtocol, build_objectnav_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_results(root: Path) -> dict[int, dict]:
    values = {}
    for path in sorted((root / "episodes").glob("*.json")):
        item = read_json(path)
        index = int((item.get("probe_metadata") or {})["source_probe_index"])
        values[index] = item
    return values


def camera_rotation_from_pointgoal(position, goal, theta) -> np.ndarray:
    """Recover the recorded level camera yaw from a point-goal observation."""

    delta = np.asarray(goal, dtype=float)[:2] - np.asarray(position, dtype=float)[:2]
    world_angle = math.atan2(float(delta[1]), float(delta[0]))
    right_axis_yaw = world_angle - (float(theta) + math.pi / 2.0)
    cosine, sine = math.cos(right_axis_yaw), math.sin(right_axis_yaw)
    # OpenFrontier camera convention: +X right, +Y down, +Z forward.
    return np.array(
        [
            [cosine, 0.0, -sine],
            [sine, 0.0, cosine],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )


def geodesic(pathfinder, start, goal) -> float:
    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32)
    query.requested_end = np.asarray(goal, dtype=np.float32)
    return float(query.geodesic_distance) if pathfinder.find_path(query) else math.inf


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    source = source_results(args.oracle_b)
    protocol = NavigationProtocol(success_distance=1.0, max_episode_steps=500)
    config = build_objectnav_config(
        "hm3dv2", seed=20260727, protocol=protocol, top_down_map=False
    )
    dataset, _, _, requested = select_manifest_episodes(
        config, args.manifest, "HM3Dv2"
    )
    rows = []
    with habitat.Env(config=config, dataset=dataset) as env:
        for request in requested:
            env.reset()
            probe = int(request["source_probe_index"])
            item = source[probe]
            oracle = (item.get("target_diagnostics") or {})["target_approach_oracle"]
            timeline = oracle["timeline"]
            first = timeline[0]
            endpoint_hab = np.asarray(oracle["selected_endpoint"]["position_hab"], dtype=float)
            start_hab = env.sim.pathfinder.snap_point(
                to_habitat_position(np.asarray(first["position"], dtype=float))
            )
            rotation = camera_rotation_from_pointgoal(
                first["position"], first["effective_pointnav_goal"], first["theta"]
            )
            state = AgentState(
                position=np.asarray(start_hab, dtype=float),
                rotation=to_habitat_rotation(rotation),
            )
            env.sim.get_agent(0).set_state(state)
            follower = GreedyGeodesicFollower(
                env.sim.pathfinder,
                env.sim.get_agent(0),
                goal_radius=0.2,
                stop_key=None,
                forward_key=1,
                left_key=2,
                right_key=3,
            )
            before = copy.deepcopy(env.sim.get_agent_state())
            initial_geodesic = geodesic(env.sim.pathfinder, before.position, endpoint_hab)
            actions = follower.find_path(endpoint_hab)
            # find_path is diagnostic planning and may simulate internally. Restore
            # the exact start before action execution.
            env.sim.get_agent(0).set_state(state)
            executed = []
            for action in actions:
                if action is None or action == 0:
                    break
                env.step(int(action))
                executed.append(int(action))
            after = env.sim.get_agent_state()
            final_geodesic = geodesic(env.sim.pathfinder, after.position, endpoint_hab)
            collision_metrics = (env.get_metrics().get("collisions") or {})
            counts = {str(code): executed.count(code) for code in (1, 2, 3)}
            rows.append(
                {
                    "source_probe_index": probe,
                    "source_full_index": int(request["source_full_index"]),
                    "target": request["target"],
                    "start_hab": np.asarray(start_hab, dtype=float).tolist(),
                    "endpoint_hab": endpoint_hab.tolist(),
                    "initial_geodesic": initial_geodesic,
                    "action_count": len(executed),
                    "actions": counts,
                    "final_geodesic": final_geodesic,
                    "reached_0_2m": bool(final_geodesic <= 0.2),
                    "net_motion": float(
                        np.linalg.norm(
                            np.asarray(after.position, dtype=float)
                            - np.asarray(before.position, dtype=float)
                        )
                    ),
                    "collisions": collision_metrics,
                }
            )
    payload = {
        "schema_version": 1,
        "analysis_only": True,
        "control": "Habitat-Sim GreedyGeodesicFollower to Oracle-B endpoint",
        "episodes": rows,
        "successes": sum(row["reached_0_2m"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
