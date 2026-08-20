"""Strict T1 ApexFusion target-perception pipeline for OpenFrontier."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np

from zson3.services.apex_target import ApexTargetServiceClient

from .fusion import FusionTarget, TargetFusionManager
from .geometry import ObjectGeometryExtractor


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

HM3D_TO_T1_TARGET = {
    "plant": "potted plant",
    "potted_plant": "potted plant",
    "tv_monitor": "tv",
    "television_screen": "tv",
    "sofa": "couch",
    "loveseat": "couch",
}


@dataclass
class ApexTargetGoal:
    """Planner-facing view of a reliable T1 cluster, not an OF DetectedObject."""

    id: int
    label: str
    centroid: np.ndarray
    confidence: float
    positive_volume: int
    positive_observation_count: int

    def to_dict(self) -> dict:
        return {
            "source": "t1_apex_fusion",
            "cluster_id": self.id,
            "label": self.label,
            "centroid": self.centroid.tolist(),
            "confidence": float(self.confidence),
            "positive_volume": int(self.positive_volume),
            "positive_observation_count": int(self.positive_observation_count),
        }


class ApexTargetPipeline:
    """Per-frame detector -> MobileSAM -> T1 fusion, with no navigation authority."""

    def __init__(self, *, raw_target: str, intrinsic: np.ndarray, config: dict) -> None:
        canonical_target = HM3D_TO_T1_TARGET.get(raw_target.strip().lower(), raw_target.strip().lower())
        self.raw_target = raw_target
        self.canonical_target = canonical_target
        self.intrinsic = np.asarray(intrinsic, dtype=float)
        self.min_depth_m = float(config.get("apex_target_min_depth_m", 0.5))
        self.max_depth_m = float(config.get("apex_target_max_depth_m", 3.5))
        self.yolo_threshold = float(config.get("apex_target_yolo_threshold", 0.8))
        self.dino_threshold = float(config.get("apex_target_dino_threshold", 0.4))
        self.services = ApexTargetServiceClient()
        self.geometry = ObjectGeometryExtractor(
            erosion_size=int(config.get("apex_target_erosion_size", 1)),
            voxel_size_m=float(config.get("apex_target_voxel_size_m", 0.10)),
            depth_cloud_stride=int(config.get("apex_target_depth_cloud_stride", 4)),
        )
        self.fusion = TargetFusionManager(
            coco_classes=COCO_CLASSES,
            reliable_confidence_threshold=float(config.get("apex_target_reliable_confidence", 0.65)),
            reliable_min_volume=int(config.get("apex_target_reliable_min_volume", 8)),
            reliable_min_positive_observations=int(config.get("apex_target_min_positive_observations", 2)),
            cluster_match_radius_m=float(config.get("apex_target_cluster_match_radius_m", 0.60)),
            cluster_strict_match_radius_m=float(config.get("apex_target_cluster_strict_match_radius_m", 0.25)),
            cluster_min_overlap_voxels=int(config.get("apex_target_cluster_min_overlap_voxels", 3)),
            cluster_bounds_margin_m=float(config.get("apex_target_cluster_bounds_margin_m", 0.25)),
            lambda_neg=float(config.get("apex_target_lambda_neg", 0.20)),
            min_negative_overlap_voxels=int(config.get("apex_target_min_negative_overlap_voxels", 8)),
        )
        self.fusion.set_goal(canonical_target)
        self.last_trace = self.fusion.trace
        # Ephemeral, evaluator-only masks from the most recent update. They are
        # intentionally excluded from JSON traces and never affect fusion.
        self.last_masks: list[np.ndarray] = []

    def update(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        world_from_camera: np.ndarray,
        robot_xy: np.ndarray,
        step: int,
    ) -> tuple[Optional[ApexTargetGoal], dict]:
        start = time.time()
        labels = self.fusion.goal_labels
        detections = self.services.detect(image=rgb, backend=labels.detector_backend, caption=labels.caption)
        threshold = self.yolo_threshold if labels.detector_backend == "yolov7" else self.dino_threshold
        allowed = set(labels.detector_labels)
        detections = [d for d in detections if d.phrase in allowed and d.confidence >= threshold]

        height, width = rgb.shape[:2]
        observations = []
        records = []
        self.last_masks = []
        for detection in detections:
            bbox = detection.box_xyxy_normalized * np.array([width, height, width, height], dtype=float)
            mask = self.services.segment_bbox(image=rgb, bbox_xyxy=bbox)
            self.last_masks.append(np.asarray(mask, dtype=bool).copy())
            canonical_label = self.fusion.canonical_label_for_phrase(detection.phrase)
            observation = self.geometry.extract_observation(
                depth_m=np.asarray(depth_m).squeeze(),
                object_mask=mask,
                world_from_camera=world_from_camera,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
                intrinsic=self.intrinsic,
                confidence=detection.confidence,
                step=step,
                phrase=detection.phrase,
                canonical_label=canonical_label,
                detector=labels.detector_backend,
            )
            records.append({
                "phrase": detection.phrase,
                "canonical_label": canonical_label,
                "confidence": float(detection.confidence),
                "bbox_xyxy_normalized": detection.box_xyxy_normalized.tolist(),
                "mask_pixels": int(np.count_nonzero(mask)),
                "geometry_accepted": observation is not None,
                "centroid_xyz": None if observation is None else observation.centroid_xyz.tolist(),
                "voxel_volume": 0 if observation is None else int(observation.volume),
            })
            if observation is not None:
                observations.append(observation)

        _, depth_voxels = self.geometry.extract_depth_reobservation(
            depth_m=np.asarray(depth_m).squeeze(),
            world_from_camera=world_from_camera,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            intrinsic=self.intrinsic,
        )
        fusion_trace = self.fusion.ingest(observations, depth_voxels, np.asarray(robot_xy), step)
        reliable = self.fusion.reliable_target(np.asarray(robot_xy))
        goal = None if reliable is None else self._goal_from_target(reliable)
        self.last_trace = {
            "step": int(step),
            "raw_target": self.raw_target,
            "canonical_target": self.canonical_target,
            "detector": labels.detector_backend,
            "confidence_threshold": threshold,
            "detections": records,
            "geometry_observation_count": len(observations),
            "depth_reobservation_voxels": len(depth_voxels),
            "fusion": fusion_trace,
            "reliable_target": None if goal is None else goal.to_dict(),
            "elapsed_seconds": float(time.time() - start),
        }
        return goal, self.last_trace

    def _goal_from_target(self, target: FusionTarget) -> ApexTargetGoal:
        return ApexTargetGoal(
            id=target.cluster_id,
            label=self.canonical_target,
            centroid=target.goal_xyz.copy(),
            confidence=target.confidence,
            positive_volume=target.positive_volume,
            positive_observation_count=target.positive_observation_count,
        )
