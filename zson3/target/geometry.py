"""RGB-D geometry adapter for the T1 ApexFusion target state.

The filtering and voxel semantics follow ``vlfm.mapping.object_point_cloud_map``
from the evaluated T1 implementation.  The camera projection is adapted to
OpenFrontier's CV camera convention (x right, y down, z forward) and its metric
Habitat depth stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import open3d as o3d

VoxelKey = Tuple[int, int, int]


@dataclass(frozen=True)
class ObjectObservation:
    cloud: np.ndarray
    voxel_cloud: np.ndarray
    voxel_keys: Set[VoxelKey]
    centroid_xyz: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    camera_xyz: np.ndarray
    confidence: float
    step: int
    phrase: str
    canonical_label: str
    detector: str

    @property
    def volume(self) -> int:
        return len(self.voxel_keys)


class ObjectGeometryExtractor:
    """Stateless T1 object/depth cloud extraction in OpenFrontier world axes."""

    use_dbscan: bool = True

    def __init__(
        self,
        erosion_size: int = 1,
        voxel_size_m: float = 0.10,
        depth_cloud_stride: int = 4,
        max_depth_cloud_points: int = 30000,
    ) -> None:
        self._erosion_size = int(erosion_size)
        self._voxel_size_m = float(voxel_size_m)
        self._depth_cloud_stride = int(depth_cloud_stride)
        self._max_depth_cloud_points = int(max_depth_cloud_points)

    def extract_observation(
        self,
        *,
        depth_m: np.ndarray,
        object_mask: np.ndarray,
        world_from_camera: np.ndarray,
        min_depth_m: float,
        max_depth_m: float,
        intrinsic: np.ndarray,
        confidence: float,
        step: int,
        phrase: str,
        canonical_label: str,
        detector: str,
    ) -> Optional[ObjectObservation]:
        mask = (object_mask > 0).astype(np.uint8) * 255
        mask = cv2.erode(mask, None, iterations=self._erosion_size)
        valid = (
            (mask > 0)
            & np.isfinite(depth_m)
            & (depth_m >= min_depth_m)
            & (depth_m <= max_depth_m)
        )
        local_cloud = depth_to_camera_points(depth_m, valid, intrinsic)
        local_cloud = random_subarray(local_cloud, 5000)
        if self.use_dbscan:
            local_cloud = largest_dbscan_cluster(local_cloud)
        if len(local_cloud) == 0:
            return None

        global_cloud = transform_points(world_from_camera, local_cloud)
        camera_xyz = np.asarray(world_from_camera[:3, 3], dtype=float).copy()
        # This is the evaluated T1 close-surface rejection, retained verbatim in
        # meaning.  It prevents masks dominated by robot-adjacent geometry.
        if np.min(np.linalg.norm(global_cloud - camera_xyz[None, :], axis=1)) < 1.0:
            return None

        voxel_cloud, voxel_keys = voxel_downsample(global_cloud, self._voxel_size_m)
        if not voxel_keys:
            return None
        return ObjectObservation(
            cloud=global_cloud,
            voxel_cloud=voxel_cloud,
            voxel_keys=voxel_keys,
            centroid_xyz=np.median(voxel_cloud, axis=0),
            bounds_min=np.min(voxel_cloud, axis=0),
            bounds_max=np.max(voxel_cloud, axis=0),
            camera_xyz=camera_xyz,
            confidence=float(confidence),
            step=int(step),
            phrase=phrase,
            canonical_label=canonical_label,
            detector=detector,
        )

    def extract_depth_reobservation(
        self,
        *,
        depth_m: np.ndarray,
        world_from_camera: np.ndarray,
        min_depth_m: float,
        max_depth_m: float,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, Set[VoxelKey]]:
        valid = (
            np.isfinite(depth_m)
            & (depth_m >= min_depth_m)
            & (depth_m <= max_depth_m)
        )
        if self._depth_cloud_stride > 1:
            sampled = np.zeros_like(valid, dtype=bool)
            sampled[:: self._depth_cloud_stride, :: self._depth_cloud_stride] = True
            valid &= sampled
        local_cloud = depth_to_camera_points(depth_m, valid, intrinsic)
        local_cloud = random_subarray(local_cloud, self._max_depth_cloud_points)
        return voxel_downsample(
            transform_points(world_from_camera, local_cloud), self._voxel_size_m
        )


def depth_to_camera_points(
    depth_m: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Project metric depth into OpenFrontier's CV camera frame."""
    v, u = np.where(mask)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = np.asarray(depth_m[v, u], dtype=np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.hstack((points, np.ones((len(points), 1), dtype=points.dtype)))
    transformed = np.asarray(transform, dtype=float).dot(homogeneous.T).T
    return (transformed[:, :3] / transformed[:, 3:]).astype(np.float32)


def voxel_downsample(
    points: np.ndarray, voxel_size_m: float
) -> tuple[np.ndarray, Set[VoxelKey]]:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32), set()
    voxel_indices = np.floor(points / voxel_size_m).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    voxel_cloud = points[unique_indices].astype(np.float32)
    keys = {tuple(index.tolist()) for index in voxel_indices[unique_indices]}
    return voxel_cloud, keys


def largest_dbscan_cluster(
    points: np.ndarray, eps: float = 0.2, min_points: int = 100
) -> np.ndarray:
    if len(points) == 0:
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(cloud.cluster_dbscan(eps, min_points, print_progress=False))
    valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(valid_labels) == 0:
        return np.empty((0, 3), dtype=points.dtype)
    label = valid_labels[np.argmax(counts)]
    return points[labels == label]


def random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """Match T1's seeded point-cloud subsampling."""
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size=size, replace=False)
    return points[indices]
