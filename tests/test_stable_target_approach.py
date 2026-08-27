from __future__ import annotations

import numpy as np

from nav.stable_target_approach import (
    StableApproachState,
    build_approach_candidates,
)


class FakeNavmesh:
    def __init__(self, unreachable_x_below: float = -100.0):
        self.unreachable_x_below = unreachable_x_below

    def snap_point(self, point):
        point = np.asarray(point, dtype=float).copy()
        point[:2] = np.round(point[:2], 2)
        return point

    def geodesic_distance(self, start, goal):
        goal = np.asarray(goal)
        if goal[0] < self.unreachable_x_below:
            return float("inf")
        return float(np.linalg.norm(np.asarray(start)[:2] - goal[:2]))


def pose_at(x: float, y: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [x, y, 1.0]
    return pose


def test_candidate_builder_rejects_unreachable_legacy_and_keeps_reachable_views():
    candidates, rejected = build_approach_candidates(
        legacy_pose=pose_at(-2.0, 0.0),
        target_position=np.array([0.0, 0.0, 0.5]),
        current_pose=pose_at(2.0, 0.0),
        navmesh=FakeNavmesh(unreachable_x_below=-1.0),
    )
    assert candidates
    assert candidates[0].source.startswith("radial_")
    assert all(np.isfinite(candidate.path_distance) for candidate in candidates)
    assert any(
        item["source"] == "legacy" and item["reason"] == "unreachable"
        for item in rejected
    )


def test_valid_legacy_endpoint_is_kept_first():
    candidates, _ = build_approach_candidates(
        legacy_pose=pose_at(0.6, 0.0),
        target_position=np.array([0.0, 0.0, 0.5]),
        current_pose=pose_at(2.0, 0.0),
        navmesh=FakeNavmesh(),
    )
    assert candidates[0].source == "legacy"
    np.testing.assert_allclose(candidates[0].pose[:3, 3], [0.6, 0.0, 1.0])


def test_progress_window_detects_rotation_stagnation():
    candidate = build_approach_candidates(
        legacy_pose=pose_at(0.6, 0.0),
        target_position=np.array([0.0, 0.0, 0.5]),
        current_pose=pose_at(2.0, 0.0),
        navmesh=FakeNavmesh(),
    )[0][0]
    state = StableApproachState(1, 10, np.zeros(3), [candidate])
    for step in range(20):
        summary = state.record(
            step=step,
            position=np.array([2.0, 0.0, 1.0]),
            rho=1.4,
            endpoint_distance=1.4,
            action=2 if step % 2 else 3,
        )
    assert summary["stagnant"] is True
    assert summary["turn_ratio"] == 1.0


def test_progress_window_does_not_interrupt_normal_approach():
    candidate = build_approach_candidates(
        legacy_pose=pose_at(0.6, 0.0),
        target_position=np.array([0.0, 0.0, 0.5]),
        current_pose=pose_at(2.0, 0.0),
        navmesh=FakeNavmesh(),
    )[0][0]
    state = StableApproachState(1, 10, np.zeros(3), [candidate])
    for step in range(20):
        distance = 2.0 - 0.05 * step
        summary = state.record(
            step=step,
            position=np.array([distance, 0.0, 1.0]),
            rho=distance,
            endpoint_distance=distance,
            action=1,
        )
    assert summary["stagnant"] is False
    assert summary["rho_improvement"] > 0.9
