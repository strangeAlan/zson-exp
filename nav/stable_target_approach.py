"""Stable, navmesh-valid approach goals used only after OF target acceptance."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def pose_facing(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build an OpenFrontier camera pose at ``position`` facing ``target``."""
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0])
    forward /= np.linalg.norm(forward)
    down_hint = np.array([0.0, 0.0, -1.0])
    right = np.cross(down_hint, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-6)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


@dataclass
class ApproachCandidate:
    pose: np.ndarray
    source: str
    requested_position: np.ndarray
    snapped_position: np.ndarray
    snap_displacement: float
    path_distance: float
    target_distance: float
    score: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "requested_position": self.requested_position.tolist(),
            "snapped_position": self.snapped_position.tolist(),
            "snap_displacement": float(self.snap_displacement),
            "path_distance": float(self.path_distance),
            "target_distance": float(self.target_distance),
            "score": float(self.score),
        }


@dataclass
class PursuitSample:
    step: int
    position: np.ndarray
    rho: float
    endpoint_distance: float
    action: Optional[int]


@dataclass
class StableApproachState:
    object_id: int
    accepted_step: int
    target_position: np.ndarray
    candidates: list[ApproachCandidate]
    active_index: int = 0
    recovery_count: int = 0
    endpoint_changes: int = 0
    release_reason: Optional[str] = None
    history: deque[PursuitSample] = field(default_factory=lambda: deque(maxlen=20))

    @property
    def active(self) -> Optional[ApproachCandidate]:
        if not self.candidates or self.active_index >= len(self.candidates):
            return None
        return self.candidates[self.active_index]

    def record(
        self,
        *,
        step: int,
        position: np.ndarray,
        rho: float,
        endpoint_distance: float,
        action: Optional[int],
    ) -> dict:
        self.history.append(
            PursuitSample(
                step=int(step),
                position=np.asarray(position, dtype=float).copy(),
                rho=float(rho),
                endpoint_distance=float(endpoint_distance),
                action=None if action is None else int(action),
            )
        )
        return self.progress_summary()

    def progress_summary(self) -> dict:
        samples = list(self.history)
        if not samples:
            return {"window": 0, "stagnant": False}
        turns = sum(sample.action in (2, 3) for sample in samples)
        forward = sum(sample.action == 1 for sample in samples)
        path_length = sum(
            float(np.linalg.norm(b.position[:2] - a.position[:2]))
            for a, b in zip(samples, samples[1:])
        )
        net_translation = float(
            np.linalg.norm(samples[-1].position[:2] - samples[0].position[:2])
        )
        rho_improvement = float(samples[0].rho - samples[-1].rho)
        endpoint_improvement = float(
            samples[0].endpoint_distance - samples[-1].endpoint_distance
        )
        turn_ratio = float(turns / len(samples))
        full = len(samples) == self.history.maxlen
        stagnant = bool(
            full
            and rho_improvement < 0.12
            and endpoint_improvement < 0.12
            and (turn_ratio >= 0.55 or net_translation < 0.15)
        )
        return {
            "window": len(samples),
            "start_step": int(samples[0].step),
            "end_step": int(samples[-1].step),
            "rho_start": float(samples[0].rho),
            "rho_end": float(samples[-1].rho),
            "rho_improvement": rho_improvement,
            "endpoint_distance_start": float(samples[0].endpoint_distance),
            "endpoint_distance_end": float(samples[-1].endpoint_distance),
            "endpoint_improvement": endpoint_improvement,
            "translation_path_length": path_length,
            "net_translation": net_translation,
            "turn_count": int(turns),
            "forward_count": int(forward),
            "turn_ratio": turn_ratio,
            "stagnant": stagnant,
        }

    def advance_candidate(self) -> Optional[ApproachCandidate]:
        self.active_index += 1
        self.recovery_count += 1
        self.endpoint_changes += 1
        self.history.clear()
        return self.active


def _unit_xy(vector: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=float)[:2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector / norm


def build_approach_candidates(
    *,
    legacy_pose: np.ndarray,
    target_position: np.ndarray,
    current_pose: np.ndarray,
    navmesh,
    max_snap_displacement: float = 0.45,
) -> tuple[list[ApproachCandidate], list[dict]]:
    """Create a small, fixed set of reachable target-side viewpoints.

    The original OF endpoint is retained first when its snapped point is on the
    same navmesh island and has minimal correction.  Remaining proposals lie on
    the observed side of the target and are used only for bounded recovery.
    """
    legacy_pose = np.asarray(legacy_pose, dtype=float)
    target = np.asarray(target_position, dtype=float)
    current = np.asarray(current_pose, dtype=float)
    legacy = legacy_pose[:3, 3].copy()
    outward = _unit_xy(current[:3, 3] - target)
    if outward is None:
        outward = _unit_xy(legacy - target)
    if outward is None:
        outward = np.array([1.0, 0.0])

    proposal_specs: list[tuple[str, np.ndarray, float]] = [
        ("legacy", legacy.copy(), 0.0)
    ]
    for angle_degrees, radius in (
        (0.0, 0.55),
        (0.0, 0.75),
        (35.0, 0.75),
        (-35.0, 0.75),
        (70.0, 0.90),
        (-70.0, 0.90),
        (0.0, 0.95),
        (0.0, 1.20),
    ):
        angle = np.deg2rad(angle_degrees)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        direction = rotation @ outward
        requested = target.copy()
        requested[:2] += radius * direction
        requested[2] = current[2, 3]
        proposal_specs.append(
            (f"radial_{radius:.2f}_{angle_degrees:+.0f}", requested, 1.0)
        )

    candidates: dict[tuple[int, int], ApproachCandidate] = {}
    rejected: list[dict] = []

    def consider(source: str, requested: np.ndarray, source_penalty: float) -> None:
        snapped = np.asarray(navmesh.snap_point(requested), dtype=float)
        rejection = {
            "source": source,
            "requested_position": requested.tolist(),
        }
        if not np.all(np.isfinite(snapped)):
            rejection["reason"] = "non_finite_snap"
            rejected.append(rejection)
            return
        snap_displacement = float(np.linalg.norm(snapped[:2] - requested[:2]))
        allowed_snap = 0.35 if source == "legacy" else max_snap_displacement
        if snap_displacement > allowed_snap:
            rejection.update(
                {"reason": "excessive_snap", "snap_displacement": snap_displacement}
            )
            rejected.append(rejection)
            return
        snapped[2] = current[2, 3]
        path_distance = float(navmesh.geodesic_distance(current[:3, 3], snapped))
        if not np.isfinite(path_distance):
            rejection.update(
                {"reason": "unreachable", "snap_displacement": snap_displacement}
            )
            rejected.append(rejection)
            return
        target_distance = float(np.linalg.norm(snapped[:2] - target[:2]))
        if target_distance < 0.35:
            rejection.update(
                {
                    "reason": "too_close_to_centroid",
                    "snap_displacement": snap_displacement,
                    "target_distance": target_distance,
                }
            )
            rejected.append(rejection)
            return
        # Preserve a valid OF endpoint; otherwise prefer a conservative 0.65 m
        # standoff, small snap, and short route.  This is fixed, not tuned online.
        score = (
            source_penalty
            + abs(target_distance - 0.65)
            + 0.5 * snap_displacement
            + 0.02 * path_distance
        )
        pose = pose_facing(snapped, target)
        candidate = ApproachCandidate(
            pose=pose,
            source=source,
            requested_position=requested.copy(),
            snapped_position=snapped.copy(),
            snap_displacement=snap_displacement,
            path_distance=path_distance,
            target_distance=target_distance,
            score=float(score),
        )
        key = tuple(np.round(snapped[:2] / 0.15).astype(int).tolist())
        if key not in candidates or candidate.score < candidates[key].score:
            candidates[key] = candidate

    for source, requested, source_penalty in proposal_specs:
        consider(source, requested, source_penalty)

    # A target may be visible through a doorway while most of its radial ring is
    # on a different navmesh island.  Around every reachable seed, add a few
    # small same-island alternatives.  These remain target-facing viewpoints,
    # not a new planner or an unbounded sampling policy.
    seed_candidates = list(candidates.values())
    lateral = np.array([-outward[1], outward[0]])
    local_offsets = (
        0.25 * outward,
        -0.25 * outward,
        0.25 * lateral,
        -0.25 * lateral,
        0.40 * lateral,
        -0.40 * lateral,
    )
    for seed_index, seed in enumerate(seed_candidates):
        for offset_index, offset in enumerate(local_offsets):
            requested = seed.snapped_position.copy()
            requested[:2] += offset
            consider(
                f"local_{seed_index}_{offset_index}", requested, 1.25
            )

    ordered = sorted(candidates.values(), key=lambda item: item.score)
    legacy_candidates = [item for item in ordered if item.source == "legacy"]
    if legacy_candidates:
        legacy_candidate = legacy_candidates[0]
        ordered = [legacy_candidate] + [
            item for item in ordered if item is not legacy_candidate
        ]
    return ordered, rejected
