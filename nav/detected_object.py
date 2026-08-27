import numpy as np
import cv2
from typing import Optional


class DetectedObject:

    _next_id = 1

    def __init__(
        self,
    ):

        self.id = DetectedObject._next_id
        DetectedObject._next_id += 1
        self.label = ""
        self.mask = None
        self.rgb = None
        self.evidence_image = None
        self.evidence_label = None
        self.evidence_labels = []
        self.evidence_step = None
        self.evidence_observations = []
        self.image_index = 0
        self.depth = None
        self.viewpoint = None
        self.centroid = None
        self.raw_centroid = None
        self.robust_position = None
        self.surface_points = np.zeros((0, 3), dtype=float)
        self.gt_overlap = None
        self.is_valid = True
        self.verification_status = "unverified"
        self.frontier = None
        self.detection_score = None
        self.box_2d = None
        self.observation_count = 1
        self.first_seen_step = None
        self.last_seen_step = None
        self.pursuit_state = "explore"
        self.bound_probability = None
        self.bound_reason = None
        self.last_bound_step = None
        self.active_approach_pose = None
        self.approach_viewpoints = []
        self.attempted_approach_keys = set()
        self.reobserve_cycles = 0
        self.pursuit_start_step = None

    @classmethod
    def from_mask(
        cls,
        mask: dict,
        depth: np.ndarray,
        viewpoint: np.ndarray,
        intrinsic_mat: np.ndarray,
        step: int | None = None,
        rgb: np.ndarray | None = None,
        evidence_image: np.ndarray | None = None,
        evidence_label: str | None = None,
        evidence_labels: list[str] | None = None,
        target_semantic_mask: np.ndarray | None = None,
    ):
        obj = cls()
        obj.label = mask.get("label", "")
        obj.image_index = mask.get("image_index", 0)
        obj.detection_score = mask.get("detection_score")
        obj.box_2d = mask.get("box_2d")
        obj.first_seen_step = step
        obj.last_seen_step = step
        obj.evidence_step = step

        mask_array = np.asarray(mask["mask"], dtype=bool)
        obj.mask = mask_array.copy()
        obj.rgb = None if rgb is None else np.asarray(rgb).copy()
        obj.evidence_image = evidence_image
        obj.evidence_label = evidence_label
        obj.evidence_labels = list(evidence_labels or [])
        mask_depth = mask_array * depth

        obj.depth = depth
        obj.viewpoint = viewpoint
        obj.raw_centroid = cls.get_object_location(
            mask_depth, viewpoint, intrinsic_mat
        )
        obj.surface_points, obj.robust_position = cls.get_robust_surface(
            mask_array, depth, viewpoint, intrinsic_mat
        )
        obj.centroid = (
            obj.robust_position.copy()
            if obj.robust_position is not None
            else obj.raw_centroid
        )
        obj.gt_overlap = cls.mask_overlap(mask_array, target_semantic_mask)
        obj.evidence_observations = [
            {
                "image": evidence_image,
                "label": evidence_label,
                "labels": list(evidence_labels or []),
                "mask": obj.mask,
                "rgb": obj.rgb,
                "viewpoint": np.asarray(viewpoint, dtype=float).copy(),
                "raw_centroid": (
                    None
                    if obj.raw_centroid is None
                    else np.asarray(obj.raw_centroid, dtype=float).copy()
                ),
                "robust_position": (
                    None
                    if obj.robust_position is None
                    else np.asarray(obj.robust_position, dtype=float).copy()
                ),
                "surface_points": obj.surface_points.copy(),
                "score": obj.detection_score,
                "step": step,
                "gt_overlap": obj.gt_overlap,
            }
        ]
        return obj

    @classmethod
    def get_robust_surface(
        cls,
        mask: np.ndarray,
        depth: np.ndarray,
        viewpoint: np.ndarray,
        intrinsic_mat: np.ndarray,
        max_points: int = 512,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return a foreground-robust visible surface and representative point.

        This is deliberately observation-local: it removes mask-edge/background
        contamination without constructing a persistent voxel fusion map.
        """
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape != depth.shape[:2] or not np.any(mask):
            return np.zeros((0, 3), dtype=float), None

        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        minimum = max(20, int(np.count_nonzero(mask) * 0.35))
        core = eroded.astype(bool) if np.count_nonzero(eroded) >= minimum else mask.astype(bool)
        valid = core & np.isfinite(depth) & (depth > 0)
        values = np.asarray(depth[valid], dtype=float)
        if values.size == 0:
            return np.zeros((0, 3), dtype=float), None

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 1e-5:
            scale = 1.4826 * mad
            valid &= np.abs(depth - median) <= 3.0 * scale
        else:
            tolerance = max(0.08, median * 0.05)
            valid &= np.abs(depth - median) <= tolerance
        filtered_depth = np.where(valid, depth, 0.0)
        points_camera = cls.depth_to_point_cloud(filtered_depth, intrinsic_mat)
        if points_camera.shape[0] == 0:
            return np.zeros((0, 3), dtype=float), None
        ones = np.ones((points_camera.shape[0], 1))
        points_world = (viewpoint @ np.hstack((points_camera, ones)).T).T[:, :3]
        if len(points_world) > max_points:
            sample_indices = np.linspace(
                0, len(points_world) - 1, max_points, dtype=int
            )
            points_world = points_world[sample_indices]
        representative = np.median(points_world, axis=0)
        return points_world.astype(float), representative.astype(float)

    @staticmethod
    def mask_overlap(mask: np.ndarray, target_mask: np.ndarray | None) -> dict | None:
        """Evaluator-only mask/GT overlap; callers must never use it for policy."""
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

    def best_evidence(self) -> dict | None:
        usable = [
            observation
            for observation in self.evidence_observations
            if observation.get("image") is not None
            and observation.get("label") is not None
        ]
        if not usable:
            return None
        return max(
            usable,
            key=lambda observation: (
                int(observation.get("step") or -1),
                float(observation.get("score") or 0.0),
            ),
        )

    def update_geometry(self, observation: "DetectedObject") -> None:
        """Replace geometry with a newly candidate-bound observation."""
        self.mask = observation.mask
        self.rgb = observation.rgb
        self.depth = observation.depth
        self.viewpoint = observation.viewpoint
        self.raw_centroid = observation.raw_centroid
        self.robust_position = observation.robust_position
        self.surface_points = observation.surface_points.copy()
        self.centroid = observation.centroid.copy()
        self.box_2d = observation.box_2d
        self.detection_score = observation.detection_score
        self.last_seen_step = observation.last_seen_step
        self.evidence_image = observation.evidence_image
        self.evidence_label = observation.evidence_label
        self.evidence_labels = list(observation.evidence_labels)
        self.evidence_step = observation.evidence_step
        self.gt_overlap = observation.gt_overlap
        self.evidence_observations.extend(observation.evidence_observations)
        self.observation_count += 1

    def surface_distance(self, position: np.ndarray, horizontal: bool = True) -> float:
        points = self.surface_points
        if points is None or len(points) == 0:
            if self.robust_position is None:
                return float("inf")
            points = np.asarray(self.robust_position).reshape(1, 3)
        dimensions = slice(0, 2) if horizontal else slice(0, 3)
        return float(
            np.min(
                np.linalg.norm(
                    points[:, dimensions] - np.asarray(position)[dimensions], axis=1
                )
            )
        )

    @classmethod
    def get_object_location(
        cls,
        mask_depth: np.ndarray,
        viewpoint: np.ndarray,
        intrinsic_mat: np.ndarray,
    ) -> Optional[np.ndarray]:
        points_3d = cls.depth_to_point_cloud(mask_depth, intrinsic_mat)  # Nx3
        if points_3d.shape[0] == 0:
            return None
        # Transform to world coordinates
        ones = np.ones((points_3d.shape[0], 1))
        points_homogeneous = np.hstack((points_3d, ones))  # Nx4
        points_world = (viewpoint @ points_homogeneous.T).T  # Nx4
        centroid = np.mean(points_world[:, :3], axis=0)
        return centroid

    @classmethod
    def depth_to_point_cloud(
        cls, depth: np.ndarray, intrinsic_mat: np.ndarray
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud in camera coordinates.
        """
        mask = depth > 0
        indices = np.array(np.nonzero(mask)).T  # Nx2 (v,u)
        if indices.shape[0] == 0:
            return np.zeros((0, 3))
        us = indices[:, 1]
        vs = indices[:, 0]
        zs = depth[vs, us]

        xs = (us - intrinsic_mat[0, 2]) * zs / intrinsic_mat[0, 0]
        ys = (vs - intrinsic_mat[1, 2]) * zs / intrinsic_mat[1, 1]

        points_3d = np.vstack((xs, ys, zs)).T  # Nx3
        return points_3d

    def to_dict(self) -> dict:
        """JSON-serializable summary for navigation state snapshots."""

        def _list(x):
            return None if x is None else np.asarray(x, dtype=float).tolist()

        return {
            "id": self.id,
            "label": self.label,
            "position": _list(self.centroid),
            "centroid": _list(self.centroid),
            "raw_centroid": _list(self.raw_centroid),
            "robust_position": _list(self.robust_position),
            "surface_point_count": int(len(self.surface_points)),
            "image_index": self.image_index,
            "viewpoint": _list(self.viewpoint),
            "is_valid": self.is_valid,
            "verification_status": self.verification_status,
            "detection_score": self.detection_score,
            "box_2d": self.box_2d,
            "observation_count": self.observation_count,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "evidence_label": self.evidence_label,
            "evidence_step": self.evidence_step,
            "evidence_observation_count": len(self.evidence_observations),
            "gt_overlap": self.gt_overlap,
            "pursuit_state": self.pursuit_state,
            "bound_probability": self.bound_probability,
            "last_bound_step": self.last_bound_step,
            "reobserve_cycles": self.reobserve_cycles,
            "frontier_id": (
                None if self.frontier is None else getattr(self.frontier, "id", None)
            ),
        }
