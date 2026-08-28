"""Analysis-only target-approach Oracle for the frozen OF-base policy.

GT is used only to (1) audit candidate masks and (2) choose an endpoint from a
candidate-derived set (Oracle A) or the dataset goal viewpoints (Oracle B).
It is never imported by the default runner mode and never changes acceptance,
frontier selection, SAM3, Qwen, or PointNav weights/controller code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import habitat_sim
import numpy as np
from habitat_sim.geo import Ray

from nav.detected_object import DetectedObject
from nav.pointnav_agent import PointnavAgent
from utils.geometry import compute_alignment_transforms
from utils.transform import from_habitat_position, to_habitat_position


RING_RADII_M = (0.4, 0.7, 1.0, 1.4, 2.0)
ANGLE_COUNT = 24
MIN_SURFACE_DISTANCE_M = 0.35
MAX_SURFACE_DISTANCE_M = 2.1
MAX_SNAP_DISPLACEMENT_M = 0.5
ENDPOINT_ARRIVAL_M = 0.35


def mask_overlap_diagnostics(predicted: np.ndarray, target: np.ndarray) -> dict:
    """Apply one predeclared, conservative candidate/GT overlap rubric."""

    pred = np.asarray(predicted, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        return {
            "classification": "ambiguous",
            "ambiguity_reason": "shape_mismatch",
            "predicted_shape": list(pred.shape),
            "target_shape": list(gt.shape),
        }
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    precision = intersection / pred_n if pred_n else 0.0
    recall = intersection / gt_n if gt_n else 0.0
    iou = intersection / union if union else 0.0

    # Fixed before execution. "Correct" deliberately requires meaningful mask
    # purity, while the broad middle band is sent to overlay/manual review.
    if gt_n == 0:
        classification = "ambiguous"
        ambiguity_reason = "gt_not_visible_or_unannotated"
    elif pred_n < 20:
        classification = "ambiguous"
        ambiguity_reason = "candidate_too_small"
    elif intersection >= 20 and precision >= 0.25:
        classification = "correct"
        ambiguity_reason = None
    elif intersection < 10 or precision < 0.05:
        classification = "wrong"
        ambiguity_reason = None
    else:
        classification = "ambiguous"
        ambiguity_reason = "boundary_overlap"
    return {
        "classification": classification,
        "ambiguity_reason": ambiguity_reason,
        "predicted_pixels": pred_n,
        "target_pixels": gt_n,
        "intersection_pixels": intersection,
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
    }


def visible_surface_points(
    mask: np.ndarray,
    depth: np.ndarray,
    viewpoint: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Back-project a robust visible surface using candidate evidence only."""

    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=float)
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    if int(valid.sum()) < 20:
        return np.empty((0, 3), dtype=float)
    values = depth[valid]
    lo, hi = np.percentile(values, [5.0, 95.0])
    valid &= (depth >= lo) & (depth <= hi)
    masked_depth = np.where(valid, depth, 0.0)
    camera_points = DetectedObject.depth_to_point_cloud(masked_depth, intrinsic)
    if camera_points.size == 0:
        return np.empty((0, 3), dtype=float)
    ones = np.ones((camera_points.shape[0], 1), dtype=float)
    world = (np.asarray(viewpoint) @ np.hstack([camera_points, ones]).T).T[:, :3]
    # A fixed stride limits navmesh/nearest-surface work without changing its
    # spatial support.
    return world[:: max(1, len(world) // 2000)]


def _geodesic(pathfinder, start_hab: np.ndarray, end_hab: np.ndarray) -> float:
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start_hab, dtype=np.float32)
    path.requested_end = np.asarray(end_hab, dtype=np.float32)
    if not pathfinder.find_path(path):
        return float("inf")
    return float(path.geodesic_distance)


def _facing_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    direction = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    direction[2] = 0.0
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    pose = compute_alignment_transforms(
        origins=[np.asarray(position, dtype=float)],
        align_vec=direction,
        align_axis=[0, 0, 1],
        appr_vec=[0, 0, -1],
        appr_axis=[0, 1, 0],
    )[0]
    pose[:3, 3] = position
    return pose


class TargetApproachOracleAgent(PointnavAgent):
    """PointnavAgent with analysis-only post-acceptance Oracle execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_candidate_evidence = True
        mode = str(getattr(self.args, "target_approach_oracle", "a")).lower()
        if mode not in {"a", "b"}:
            raise ValueError(f"Unsupported target approach oracle mode: {mode}")
        self.oracle_mode = mode
        self.oracle_post_accept_steps = int(
            getattr(self.args, "oracle_post_accept_steps", 300)
        )
        self.oracle_active = False
        self.oracle_terminal_reason = None
        self.oracle_accepted_step = None
        self.oracle_fixed_pose = None
        self.oracle_endpoint_hab = None
        self.oracle_video_writer = None
        self.oracle_diagnostics = {
            "schema_version": 1,
            "mode": mode,
            "candidate_audit": None,
            "surface": None,
            "candidate_viewpoints": [],
            "selected_endpoint": None,
            "timeline": [],
            "outcome": None,
        }

    def _goal_viewpoints(self) -> list[dict[str, Any]]:
        viewpoints = []
        for goal_index, goal in enumerate(self.habitat_env.current_episode.goals):
            for view_index, view in enumerate(getattr(goal, "view_points", [])):
                position = np.asarray(view.agent_state.position, dtype=float)
                viewpoints.append(
                    {
                        "goal_index": goal_index,
                        "view_index": view_index,
                        "position_hab": position,
                        "iou": float(getattr(view, "iou", 0.0)),
                    }
                )
        return viewpoints

    def _latest_evidence(self, obj) -> dict | None:
        evidence = list(getattr(obj, "evidence", []))
        if not evidence:
            return None
        return max(
            evidence,
            key=lambda item: (
                -1 if item.get("source_step") is None else item["source_step"],
                -1 if item.get("image_index") is None else item["image_index"],
            ),
        )

    def _save_candidate_artifacts(
        self, evidence: dict, target_mask: np.ndarray, diagnostics: dict
    ) -> dict:
        assets = Path(self.save_dir) / "oracle_assets"
        assets.mkdir(parents=True, exist_ok=True)
        rgb = np.asarray(evidence["rgb"]).copy()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
        rgb = rgb[:, :, :3]
        pred = np.asarray(evidence["mask"], dtype=bool)
        gt = np.asarray(target_mask, dtype=bool)
        overlay = rgb.copy()
        pred_only = pred & ~gt
        gt_only = gt & ~pred
        overlap = pred & gt
        overlay[pred_only] = (
            0.45 * overlay[pred_only] + 0.55 * np.array([255, 0, 0])
        ).astype(np.uint8)
        overlay[gt_only] = (
            0.45 * overlay[gt_only] + 0.55 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay[overlap] = (
            0.3 * overlay[overlap] + 0.7 * np.array([255, 255, 0])
        ).astype(np.uint8)
        ys, xs = np.nonzero(pred)
        if len(xs):
            x1, y1, x2, y2 = (
                int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            )
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), 2)
        caption = (
            f"{diagnostics['classification']} P={diagnostics.get('precision', 0):.3f} "
            f"R={diagnostics.get('recall', 0):.3f} IoU={diagnostics.get('iou', 0):.3f}"
        )
        cv2.putText(overlay, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        overlay_path = assets / "candidate_gt_overlay.png"
        rgb_path = assets / "candidate_source_rgb.png"
        mask_path = assets / "candidate_masks.npz"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.savez_compressed(mask_path, candidate=pred, target=gt)
        return {
            "overlay": str(overlay_path),
            "source_rgb": str(rgb_path),
            "masks": str(mask_path),
        }

    def _line_of_sight(self, endpoint_hab: np.ndarray, surface_world: np.ndarray):
        try:
            surface_hab = to_habitat_position(surface_world)
            origin = np.asarray(endpoint_hab, dtype=float).copy()
            agent_state = self.habitat_env.sim.get_agent_state()
            camera_height = float(
                agent_state.sensor_states["rgb"].position[1]
                - agent_state.position[1]
            )
            origin[1] += camera_height
            delta = surface_hab - origin
            distance = float(np.linalg.norm(delta))
            if distance < 1e-6:
                return True
            results = self.habitat_env.sim.cast_ray(
                Ray(origin.astype(np.float32), (delta / distance).astype(np.float32)),
                max_distance=distance + 0.15,
            )
            if not results.has_hits():
                return True
            first = min(float(hit.ray_distance) for hit in results.hits)
            return bool(first >= distance - 0.2)
        except Exception:
            return None

    def _candidate_viewpoints(
        self, surface: np.ndarray, current_pose: np.ndarray
    ) -> list[dict[str, Any]]:
        center = np.median(surface, axis=0)
        pathfinder = self.habitat_env.sim.pathfinder
        current_hab = np.asarray(
            self.habitat_env.sim.get_agent_state().position, dtype=float
        )
        base_height = float(current_hab[1])
        camera_height = float(current_pose[2, 3])
        goal_views = self._goal_viewpoints()
        candidates = []
        seen = set()
        for radius in RING_RADII_M:
            for angle_index in range(ANGLE_COUNT):
                angle = 2.0 * math.pi * angle_index / ANGLE_COUNT
                raw_world = center.copy()
                raw_world[:2] += radius * np.array([math.cos(angle), math.sin(angle)])
                raw_hab = to_habitat_position(raw_world)
                raw_hab[1] = base_height
                snapped_hab = np.asarray(pathfinder.snap_point(raw_hab), dtype=float)
                if not np.isfinite(snapped_hab).all():
                    continue
                snap_xy = np.linalg.norm(
                    from_habitat_position(snapped_hab)[:2] - raw_world[:2]
                )
                if snap_xy > MAX_SNAP_DISPLACEMENT_M:
                    continue
                key = tuple(np.round(snapped_hab[[0, 2]] / 0.15).astype(int))
                if key in seen:
                    continue
                world = from_habitat_position(snapped_hab)
                world[2] = camera_height
                nearest_surface = float(
                    np.min(np.linalg.norm(surface[:, :2] - world[None, :2], axis=1))
                )
                if not MIN_SURFACE_DISTANCE_M <= nearest_surface <= MAX_SURFACE_DISTANCE_M:
                    continue
                current_distance = _geodesic(pathfinder, current_hab, snapped_hab)
                if not np.isfinite(current_distance):
                    continue
                los = self._line_of_sight(snapped_hab, center)
                if los is False:
                    continue
                gt_distances = [
                    _geodesic(pathfinder, snapped_hab, view["position_hab"])
                    for view in goal_views
                ]
                finite_gt = [value for value in gt_distances if np.isfinite(value)]
                pose = _facing_pose(world, center)
                candidates.append(
                    {
                        "pose": pose,
                        "position_hab": snapped_hab,
                        "radius": radius,
                        "angle_index": angle_index,
                        "snap_displacement": float(snap_xy),
                        "surface_distance": nearest_surface,
                        "current_geodesic": current_distance,
                        "gt_geodesic": min(finite_gt) if finite_gt else float("inf"),
                        "line_of_sight": los,
                    }
                )
                seen.add(key)
        return candidates

    @staticmethod
    def _json_viewpoint(item: dict[str, Any]) -> dict:
        return {
            key: (
                value.tolist() if isinstance(value, np.ndarray) else value
            )
            for key, value in item.items()
            if key != "pose"
        } | {"pose": np.asarray(item["pose"], dtype=float).tolist()}

    def _select_gt_executor_endpoint(self, current_pose: np.ndarray):
        pathfinder = self.habitat_env.sim.pathfinder
        current_hab = np.asarray(self.habitat_env.sim.get_agent_state().position)
        center = np.asarray(self.ft_manager.object_lockin.centroid, dtype=float)
        options = []
        for view in self._goal_viewpoints():
            distance = _geodesic(pathfinder, current_hab, view["position_hab"])
            if not np.isfinite(distance):
                continue
            world = from_habitat_position(view["position_hab"])
            world[2] = current_pose[2, 3]
            options.append(
                {
                    **view,
                    "pose": _facing_pose(world, center),
                    "current_geodesic": distance,
                    "gt_geodesic": 0.0,
                    "source": "gt_success_viewpoint",
                }
            )
        return min(options, key=lambda item: item["current_geodesic"]) if options else None

    def _reset_pointnav_tracking(self) -> None:
        self.planner.pointnav_policy.reset()
        self.planner.is_first_step = True
        self.planner.last_goal = None
        self.planner.minimum_rho = float("inf")
        self.planner.close_enough = False
        self.planner.forward_failure_heat = 0
        self.planner.rotation_heat = 0
        self.planner.action = []

    def on_target_accepted(
        self,
        *,
        current_pose: np.ndarray,
        depth: np.ndarray,
        verification_event: dict,
    ) -> None:
        locked = self.ft_manager.object_lockin
        evidence = self._latest_evidence(locked)
        audit = {
            "accepted_step": int(self.navigation_steps),
            "candidate_id": int(locked.id),
            "label": locked.label,
            "composition_bbox": (
                None if evidence is None else evidence.get("box_2d")
            ),
            "raw_centroid": np.asarray(locked.centroid, dtype=float).tolist(),
            "qwen_probability": verification_event.get("probability"),
            "qwen_accepted": True,
        }
        if evidence is None or evidence.get("semantic") is None:
            audit.update(
                {
                    "classification": "ambiguous",
                    "ambiguity_reason": "missing_candidate_or_semantic_evidence",
                }
            )
            self.oracle_diagnostics["candidate_audit"] = audit
            self.oracle_terminal_reason = "oracle_candidate_ambiguous"
            return

        target_mask = np.isin(
            np.asarray(evidence["semantic"]), tuple(self.target_semantic_ids)
        )
        overlap = mask_overlap_diagnostics(evidence["mask"], target_mask)
        mask_y, mask_x = np.nonzero(np.asarray(evidence["mask"], dtype=bool))
        source_bbox = (
            None
            if len(mask_x) == 0
            else [
                int(mask_x.min()),
                int(mask_y.min()),
                int(mask_x.max()),
                int(mask_y.max()),
            ]
        )
        audit.update(overlap)
        audit.update(
            {
                "source_frame_bbox": source_bbox,
                "source_step": evidence.get("source_step"),
                "image_index": evidence.get("image_index"),
                "artifacts": self._save_candidate_artifacts(
                    evidence, target_mask, overlap
                ),
            }
        )
        self.oracle_diagnostics["candidate_audit"] = audit
        verification_event["oracle_candidate_classification"] = overlap[
            "classification"
        ]
        if overlap["classification"] != "correct":
            self.oracle_terminal_reason = (
                "oracle_candidate_wrong"
                if overlap["classification"] == "wrong"
                else "oracle_candidate_ambiguous"
            )
            return

        self.oracle_accepted_step = int(self.navigation_steps)
        if self.oracle_mode == "a":
            surface = visible_surface_points(
                evidence["mask"],
                evidence["depth"],
                evidence["viewpoint"],
                self.cam_intrinsic.intrinsic_matrix,
            )
            self.oracle_diagnostics["surface"] = {
                "points": int(len(surface)),
                "median": (
                    None if len(surface) == 0 else np.median(surface, axis=0).tolist()
                ),
            }
            if len(surface) == 0:
                self.oracle_terminal_reason = "oracle_a_no_visible_surface"
                return
            options = self._candidate_viewpoints(surface, current_pose)
            self.oracle_diagnostics["candidate_viewpoints"] = [
                self._json_viewpoint(item) for item in options
            ]
            finite = [item for item in options if np.isfinite(item["gt_geodesic"])]
            if not finite:
                self.oracle_terminal_reason = "oracle_a_no_reachable_viewpoint"
                return
            selected = min(
                finite,
                key=lambda item: (
                    item["gt_geodesic"], item["current_geodesic"], item["radius"], item["angle_index"]
                ),
            )
            selected["source"] = "candidate_surface"
        else:
            selected = self._select_gt_executor_endpoint(current_pose)
            if selected is None:
                self.oracle_terminal_reason = "oracle_b_no_reachable_gt_viewpoint"
                return

        self.oracle_fixed_pose = np.asarray(selected["pose"], dtype=float).copy()
        self.oracle_endpoint_hab = np.asarray(selected["position_hab"], dtype=float)
        self.oracle_diagnostics["selected_endpoint"] = self._json_viewpoint(selected)
        self.ft_manager.oracle_fixed_goal_pose = self.oracle_fixed_pose.copy()
        self._reset_pointnav_tracking()
        self.path_to_go = (
            self.ft_manager.plan_path_to_goal(
                current_pose, depth=depth, use_graph=False
            )
            or []
        )
        self.oracle_active = True
        verification_event.update(
            {
                "oracle_mode": self.oracle_mode,
                "oracle_fixed_endpoint": self.oracle_fixed_pose[:3, 3].tolist(),
                "oracle_candidate_viewpoints": len(
                    self.oracle_diagnostics["candidate_viewpoints"]
                ),
            }
        )

    def _write_video_frame(self, timeline: dict) -> None:
        if self.navigation_steps % 2:
            return
        obs = self.habitat_env.sim.get_sensor_observations()
        frame = np.asarray(obs["rgb"])[..., :3]
        if frame.max() <= 1:
            frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        lines = [
            f"Oracle {self.oracle_mode.upper()} step={self.navigation_steps} action={timeline['action']}",
            f"rho={timeline['rho']:.2f} endpoint={timeline['endpoint_distance']:.2f} GT={timeline['gt_distance']:.2f}",
        ]
        for row, line in enumerate(lines):
            cv2.putText(frame, line, (8, 24 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if self.oracle_video_writer is None:
            path = Path(self.save_dir) / "oracle_assets" / f"oracle_{self.oracle_mode}_pursuit.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.oracle_video_writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (frame.shape[1], frame.shape[0])
            )
            self.oracle_diagnostics["pursuit_video"] = str(path)
        self.oracle_video_writer.write(frame)

    def handle_oracle_target_lockin(
        self,
        *,
        current_pose: np.ndarray,
        depth: np.ndarray,
        raw_centroid_distance: float,
    ):
        if self.oracle_terminal_reason is not None:
            self.oracle_diagnostics["outcome"] = self.oracle_terminal_reason
            return False, self.oracle_terminal_reason
        if not self.oracle_active:
            return None

        endpoint_distance = float(
            np.linalg.norm(current_pose[:2, 3] - self.oracle_fixed_pose[:2, 3])
        )
        metrics = self.habitat_env.get_metrics()
        gt_distance = float(metrics.get("distance_to_goal", float("nan")))
        action = getattr(self.planner, "action", None)
        action_value = int(action) if isinstance(action, (int, np.integer)) else None
        timeline = {
            "step": int(self.navigation_steps),
            "post_accept_step": int(self.navigation_steps - self.oracle_accepted_step),
            "position": current_pose[:3, 3].astype(float).tolist(),
            "fixed_endpoint": self.oracle_fixed_pose[:3, 3].tolist(),
            "effective_pointnav_goal": (
                None if self.planner.goal_pos is None else np.asarray(self.planner.goal_pos).tolist()
            ),
            "endpoint_distance": endpoint_distance,
            "gt_distance": gt_distance,
            "raw_centroid_distance": raw_centroid_distance,
            "rho": float(getattr(self.planner, "rho", endpoint_distance)),
            "theta": float(getattr(self.planner, "theta", 0.0)),
            "action": action_value,
            "path_active": bool(self.path_to_go),
            "forward_heat": int(getattr(self.planner, "forward_failure_heat", 0)),
            "rotation_heat": int(getattr(self.planner, "rotation_heat", 0)),
        }
        self.oracle_diagnostics["timeline"].append(timeline)
        self._write_video_frame(timeline)

        if self.navigation_steps - self.oracle_accepted_step >= self.oracle_post_accept_steps:
            reason = "oracle_post_accept_budget_exhausted"
            self.oracle_diagnostics["outcome"] = reason
            return False, reason
        if not self.path_to_go:
            if endpoint_distance <= ENDPOINT_ARRIVAL_M:
                reason = "oracle_endpoint_reached"
                self.target_diagnostics["termination_event"] = {
                    "step": int(self.navigation_steps),
                    "object_id": int(self.ft_manager.object_lockin.id),
                    "agent_position": current_pose[:3, 3].tolist(),
                    "distance_to_object": raw_centroid_distance,
                    "path_exhausted": True,
                    "oracle_mode": self.oracle_mode,
                    "endpoint_distance": endpoint_distance,
                }
            else:
                reason = "oracle_pointnav_failed_before_endpoint"
            self.oracle_diagnostics["outcome"] = reason
            return False, reason
        return True, "continue_navigation"

    def get_target_diagnostics(self) -> dict:
        diagnostics = super().get_target_diagnostics()
        diagnostics["target_approach_oracle"] = self.oracle_diagnostics
        return diagnostics

    def close(self) -> None:
        if self.oracle_video_writer is not None:
            self.oracle_video_writer.release()
            self.oracle_video_writer = None
        if self.ft_manager is not None:
            self.ft_manager.oracle_fixed_goal_pose = None
        super().close()
