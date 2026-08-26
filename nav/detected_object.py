import numpy as np
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
        self.is_valid = True
        self.verification_status = "unverified"
        self.frontier = None
        self.detection_score = None
        self.box_2d = None
        self.observation_count = 1
        self.first_seen_step = None
        self.last_seen_step = None

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
    ):
        obj = cls()
        obj.label = mask.get("label", "")
        obj.image_index = mask.get("image_index", 0)
        obj.detection_score = mask.get("detection_score")
        obj.box_2d = mask.get("box_2d")
        obj.first_seen_step = step
        obj.last_seen_step = step
        obj.evidence_step = step

        mask_array = np.array(mask["mask"])
        obj.mask = mask_array.astype(bool)
        obj.rgb = None if rgb is None else np.asarray(rgb).copy()
        obj.evidence_image = evidence_image
        obj.evidence_label = evidence_label
        obj.evidence_labels = list(evidence_labels or [])
        mask_depth = mask_array * depth

        obj.depth = depth
        obj.viewpoint = viewpoint
        obj.centroid = cls.get_object_location(mask_depth, viewpoint, intrinsic_mat)
        obj.evidence_observations = [
            {
                "image": evidence_image,
                "label": evidence_label,
                "labels": list(evidence_labels or []),
                "mask": obj.mask,
                "rgb": obj.rgb,
                "viewpoint": np.asarray(viewpoint, dtype=float).copy(),
                "centroid": (
                    None
                    if obj.centroid is None
                    else np.asarray(obj.centroid, dtype=float).copy()
                ),
                "score": obj.detection_score,
                "step": step,
            }
        ]
        return obj

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
                float(observation.get("score") or 0.0),
                int(observation.get("step") or -1),
            ),
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
            "frontier_id": (
                None if self.frontier is None else getattr(self.frontier, "id", None)
            ),
        }
