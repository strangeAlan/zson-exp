"""Late-stage, behavior-preserving target-closure geometry.

Nothing in this module participates in target proposals, object merging, object
frontier creation, or global Qwen acceptance.  It is used only after OF-base has
already locked a target and is about to perform a risky centroid-triggered STOP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from nav.detected_object import DetectedObject


@dataclass
class RobustTargetObservation:
    mask: np.ndarray
    image_index: int
    viewpoint: np.ndarray
    raw_centroid: np.ndarray
    robust_position: np.ndarray
    surface_points: np.ndarray
    evidence_image: np.ndarray
    evidence_label: str
    evidence_labels: list[str]
    detection_score: Optional[float]
    gt_overlap: Optional[dict] = None

    def surface_distance(self, position: np.ndarray) -> float:
        points = self.surface_points
        if points is None or len(points) == 0:
            points = self.robust_position.reshape(1, 3)
        return float(
            np.min(
                np.linalg.norm(
                    points[:, :2] - np.asarray(position, dtype=float)[:2], axis=1
                )
            )
        )


@dataclass
class SafeClosureState:
    object_id: int
    accepted_step: int
    raw_centroid: np.ndarray
    legacy_endpoint_pose: Optional[np.ndarray] = None
    mode: str = "legacy_approach"
    intervention_attempts: int = 0
    orientation_attempts: int = 0
    translation_arrival_cycles: int = 0
    refinement_pose: Optional[np.ndarray] = None
    robust_observation: Optional[RobustTargetObservation] = None
    intervention_step: Optional[int] = None
    events: list[dict] = field(default_factory=list)


def robust_target_observation(
    mask_data: dict,
    depth: np.ndarray,
    viewpoint: np.ndarray,
    intrinsic_mat: np.ndarray,
    evidence_image: np.ndarray,
    evidence_label: str,
    evidence_labels: list[str],
    target_semantic_mask: np.ndarray | None = None,
    max_points: int = 512,
) -> RobustTargetObservation | None:
    """Build a robust, observation-local surface without changing OF tracks."""
    mask = np.asarray(mask_data.get("mask"), dtype=bool)
    depth = np.asarray(depth).squeeze()
    if mask.shape != depth.shape or not np.any(mask):
        return None

    raw_depth = np.where(mask & np.isfinite(depth) & (depth > 0), depth, 0.0)
    raw_centroid = DetectedObject.get_object_location(
        raw_depth, viewpoint, intrinsic_mat
    )
    if raw_centroid is None:
        return None

    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    minimum = max(20, int(np.count_nonzero(mask) * 0.35))
    core = eroded.astype(bool) if np.count_nonzero(eroded) >= minimum else mask
    valid = core & np.isfinite(depth) & (depth > 0)
    values = np.asarray(depth[valid], dtype=float)
    if values.size == 0:
        return None

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-5:
        valid &= np.abs(depth - median) <= 3.0 * 1.4826 * mad
    else:
        valid &= np.abs(depth - median) <= max(0.08, median * 0.05)
    points_camera = DetectedObject.depth_to_point_cloud(
        np.where(valid, depth, 0.0), intrinsic_mat
    )
    if points_camera.shape[0] == 0:
        return None
    points_h = np.hstack((points_camera, np.ones((len(points_camera), 1))))
    points_world = (viewpoint @ points_h.T).T[:, :3]
    if len(points_world) > max_points:
        indices = np.linspace(0, len(points_world) - 1, max_points, dtype=int)
        points_world = points_world[indices]
    robust_position = np.median(points_world, axis=0)

    return RobustTargetObservation(
        mask=mask.copy(),
        image_index=int(mask_data.get("image_index", 0)),
        viewpoint=np.asarray(viewpoint, dtype=float).copy(),
        raw_centroid=np.asarray(raw_centroid, dtype=float),
        robust_position=np.asarray(robust_position, dtype=float),
        surface_points=np.asarray(points_world, dtype=float),
        evidence_image=np.asarray(evidence_image).copy(),
        evidence_label=evidence_label,
        evidence_labels=list(evidence_labels),
        detection_score=mask_data.get("detection_score"),
        gt_overlap=mask_overlap(mask, target_semantic_mask),
    )


def mask_overlap(mask: np.ndarray, target_mask: np.ndarray | None) -> dict | None:
    """Evaluator-only overlap.  Policy callers must not branch on this result."""
    if target_mask is None:
        return None
    prediction = np.asarray(mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if prediction.shape != target.shape:
        return None
    intersection = int(np.count_nonzero(prediction & target))
    predicted = int(np.count_nonzero(prediction))
    actual = int(np.count_nonzero(target))
    union = predicted + actual - intersection
    return {
        "intersection_pixels": intersection,
        "predicted_pixels": predicted,
        "target_pixels": actual,
        "precision": float(intersection / predicted) if predicted else 0.0,
        "recall": float(intersection / actual) if actual else 0.0,
        "iou": float(intersection / union) if union else 0.0,
    }


def pose_facing(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an OpenFrontier camera pose at ``position`` facing ``target``."""
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


def local_refinement_candidates(
    legacy_endpoint: np.ndarray,
    observation: RobustTargetObservation,
    current_pose: np.ndarray,
    navmesh,
    max_correction: float = 0.45,
    max_snap_displacement: float = 0.35,
    desired_surface_distance: float = 0.7,
) -> list[dict]:
    """Generate small navmesh corrections around the original OF endpoint."""
    legacy = np.asarray(legacy_endpoint, dtype=float).copy()
    target = observation.robust_position
    radial = legacy[:2] - target[:2]
    if np.linalg.norm(radial) < 1e-6:
        radial = current_pose[:2, 3] - target[:2]
    if np.linalg.norm(radial) < 1e-6:
        radial = np.array([1.0, 0.0])
    radial /= np.linalg.norm(radial)
    lateral = np.array([-radial[1], radial[0]])

    offsets = (
        np.array([0.0, 0.0]),
        0.20 * radial,
        0.35 * radial,
        -0.20 * radial,
        0.20 * lateral,
        -0.20 * lateral,
        0.30 * lateral,
        -0.30 * lateral,
    )
    candidates = {}
    for offset in offsets:
        requested = legacy.copy()
        requested[:2] += offset
        requested[2] = current_pose[2, 3]
        snapped = np.asarray(navmesh.snap_point(requested), dtype=float)
        if not np.all(np.isfinite(snapped)):
            continue
        snap_displacement = float(np.linalg.norm(snapped[:2] - requested[:2]))
        correction = float(np.linalg.norm(snapped[:2] - legacy[:2]))
        if snap_displacement > max_snap_displacement or correction > max_correction:
            continue
        snapped[2] = current_pose[2, 3]
        path_distance = float(navmesh.geodesic_distance(current_pose[:3, 3], snapped))
        if not np.isfinite(path_distance):
            continue
        surface_distance = observation.surface_distance(snapped)
        key = tuple(np.round(snapped[:2] / 0.1).astype(int).tolist())
        score = (
            abs(surface_distance - desired_surface_distance)
            + 0.75 * correction
            + 0.5 * snap_displacement
            + 0.05 * path_distance
        )
        candidate = {
            "pose": pose_facing(snapped, target),
            "key": key,
            "requested_position": requested.tolist(),
            "snapped_position": snapped.tolist(),
            "snap_displacement": snap_displacement,
            "correction_from_legacy": correction,
            "path_distance": path_distance,
            "surface_distance": surface_distance,
            "score": float(score),
        }
        if key not in candidates or score < candidates[key]["score"]:
            candidates[key] = candidate
    return sorted(candidates.values(), key=lambda candidate: candidate["score"])
