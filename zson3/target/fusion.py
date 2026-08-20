"""Direct Python port of the evaluated T1 ApexFusion target state.

Detector outputs are label evidence, never navigation commands.  This module
keeps the T1 association, confidence fusion, competing-label, negative-evidence
and reliability gates.  It deliberately contains no OpenFrontier policy logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from .geometry import ObjectObservation, VoxelKey

TARGET_LABEL = "target"
LABEL_TABLE_VERSION = "hm3d_apexfusion_static_v1"

STATIC_GOAL_LABELS: Dict[str, Dict[str, List[str]]] = {
    "chair": {"target_aliases": ["chair"], "confusable_labels": ["couch", "bed", "dining table", "bench"]},
    "bed": {"target_aliases": ["bed"], "confusable_labels": ["couch", "chair", "dining table", "bench"]},
    "potted plant": {"target_aliases": ["potted plant"], "confusable_labels": ["vase", "cup", "bowl", "chair"]},
    "toilet": {"target_aliases": ["toilet"], "confusable_labels": ["sink", "chair", "bowl", "bench"]},
    "tv": {"target_aliases": ["tv"], "confusable_labels": ["laptop", "microwave", "oven", "book"]},
    "couch": {"target_aliases": ["couch"], "confusable_labels": ["chair", "bed", "bench", "dining table"]},
}


@dataclass(frozen=True)
class GoalLabels:
    target_object: str
    target_aliases: List[str]
    confusable_labels: List[str]
    detector_labels: List[str]
    phrase_to_canonical: Dict[str, str]
    detector_backend: str
    label_table_version: str = LABEL_TABLE_VERSION

    @property
    def caption(self) -> str:
        return " . ".join(self.detector_labels) + " ."

    def to_trace(self) -> Dict[str, object]:
        return {
            "target_object": self.target_object,
            "target_aliases": self.target_aliases,
            "confusable_labels": self.confusable_labels,
            "detector_labels": self.detector_labels,
            "detector_backend": self.detector_backend,
            "label_table_version": self.label_table_version,
        }


@dataclass
class FusionTarget:
    cluster_id: int
    goal_xyz: np.ndarray
    confidence: float
    positive_volume: int
    positive_observation_count: int

    @property
    def goal_xy(self) -> np.ndarray:
        return self.goal_xyz[:2]


@dataclass
class LabelEvidence:
    label: str
    fused_confidence: float = 0.0
    fusion_weight: float = 0.0
    positive_volume: int = 0
    negative_reobservation_volume: float = 0.0
    positive_observation_count: int = 0
    voxel_keys: Set[VoxelKey] = field(default_factory=set)
    negative_voxels: Set[VoxelKey] = field(default_factory=set)
    cloud: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    last_positive_step: Optional[int] = None
    last_negative_step: Optional[int] = None

    @property
    def score(self) -> float:
        return self.fused_confidence * self.positive_volume

    def add_positive(self, observation: ObjectObservation) -> Dict[str, object]:
        old_confidence, old_volume = self.fused_confidence, self.positive_volume
        new_voxels = observation.voxel_keys - self.voxel_keys
        self.voxel_keys.update(observation.voxel_keys)
        self.positive_volume = len(self.voxel_keys)
        self.positive_observation_count += 1
        self.last_positive_step = observation.step
        self.cloud = (
            observation.voxel_cloud.copy()
            if self.cloud.size == 0
            else np.concatenate([self.cloud, observation.voxel_cloud], axis=0)
        )
        weight = max(float(observation.volume), 1.0)
        self.fused_confidence = _weighted_mean(
            self.fused_confidence, self.fusion_weight, observation.confidence, weight
        )
        self.fusion_weight += weight
        return {
            "label": self.label,
            "event": "positive_fusion",
            "observation_volume": int(observation.volume),
            "new_voxels": int(len(new_voxels)),
            "confidence_before": float(old_confidence),
            "confidence_after": float(self.fused_confidence),
            "positive_volume_before": int(old_volume),
            "positive_volume_after": int(self.positive_volume),
            "positive_observation_count": int(self.positive_observation_count),
        }

    def add_negative(
        self, depth_voxels: Set[VoxelKey], lambda_neg: float, min_overlap_voxels: int, step: int
    ) -> Optional[Dict[str, object]]:
        overlap = (self.voxel_keys & depth_voxels) - self.negative_voxels
        if len(overlap) < min_overlap_voxels:
            return None
        old_confidence = self.fused_confidence
        effective_volume = float(lambda_neg * len(overlap))
        self.fused_confidence = _weighted_mean(
            self.fused_confidence, self.fusion_weight, 0.0, effective_volume
        )
        self.fusion_weight += effective_volume
        self.negative_reobservation_volume += effective_volume
        self.negative_voxels.update(overlap)
        self.last_negative_step = step
        return {
            "label": self.label,
            "event": "weak_negative_fusion",
            "raw_overlap_voxels": int(len(overlap)),
            "lambda_neg": float(lambda_neg),
            "effective_negative_volume": effective_volume,
            "confidence_before": float(old_confidence),
            "confidence_after": float(self.fused_confidence),
            "negative_reobservation_volume": float(self.negative_reobservation_volume),
        }

    def to_trace(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "fused_confidence": float(self.fused_confidence),
            "fusion_weight": float(self.fusion_weight),
            "positive_volume": int(self.positive_volume),
            "negative_reobservation_volume": float(self.negative_reobservation_volume),
            "positive_observation_count": int(self.positive_observation_count),
            "score": float(self.score),
            "last_positive_step": self.last_positive_step,
            "last_negative_step": self.last_negative_step,
        }


@dataclass
class ObjectCluster:
    cluster_id: int
    geometry_cloud: np.ndarray
    geometry_voxels: Set[VoxelKey]
    centroid_xyz: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    created_step: int
    last_update_step: int
    labels: Dict[str, LabelEvidence] = field(default_factory=dict)
    best_label: Optional[str] = None
    best_score: float = 0.0
    previous_best_label: Optional[str] = None

    def add_positive(self, observation: ObjectObservation) -> Dict[str, object]:
        self.last_update_step = observation.step
        self.geometry_voxels.update(observation.voxel_keys)
        self.geometry_cloud = np.concatenate([self.geometry_cloud, observation.voxel_cloud], axis=0)
        self.centroid_xyz = np.median(self.geometry_cloud, axis=0)
        self.bounds_min = np.minimum(self.bounds_min, observation.bounds_min)
        self.bounds_max = np.maximum(self.bounds_max, observation.bounds_max)
        evidence = self.labels.setdefault(
            observation.canonical_label, LabelEvidence(observation.canonical_label)
        )
        event = evidence.add_positive(observation)
        event.update(cluster_id=self.cluster_id, phrase=observation.phrase, detector=observation.detector)
        label_flip = self.refresh_best_label()
        if label_flip is not None:
            event["label_flip"] = label_flip
        return event

    def add_soft_negatives(
        self,
        depth_voxels: Set[VoxelKey],
        observed_labels: Set[str],
        lambda_neg: float,
        min_overlap_voxels: int,
        step: int,
    ) -> List[Dict[str, object]]:
        events = []
        for label, evidence in self.labels.items():
            if label in observed_labels:
                continue
            event = evidence.add_negative(depth_voxels, lambda_neg, min_overlap_voxels, step)
            if event is None:
                continue
            event["cluster_id"] = self.cluster_id
            label_flip = self.refresh_best_label()
            if label_flip is not None:
                event["label_flip"] = label_flip
            events.append(event)
        return events

    def refresh_best_label(self) -> Optional[Dict[str, object]]:
        previous = self.best_label
        if not self.labels:
            self.best_label, self.best_score = None, 0.0
            return None
        best = max(
            self.labels.values(),
            key=lambda item: (item.score, item.fused_confidence, item.positive_volume),
        )
        self.best_label, self.best_score = best.label, best.score
        if previous is None or previous == self.best_label:
            return None
        self.previous_best_label = previous
        return {"from": previous, "to": self.best_label, "reason": "volume_weighted_label_score_argmax"}

    def target_evidence(self) -> Optional[LabelEvidence]:
        return self.labels.get(TARGET_LABEL)

    def stable_target_xyz(self) -> Optional[np.ndarray]:
        evidence = self.target_evidence()
        if evidence is None or evidence.cloud.size == 0:
            return None
        center_xy = np.median(evidence.cloud[:, :2], axis=0)
        index = int(np.argmin(np.linalg.norm(evidence.cloud[:, :2] - center_xy[None, :], axis=1)))
        return evidence.cloud[index, :3].copy()

    def to_trace(self, reliable: bool) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "centroid_xyz": self.centroid_xyz.tolist(),
            "bounds_min": self.bounds_min.tolist(),
            "bounds_max": self.bounds_max.tolist(),
            "geometry_volume": int(len(self.geometry_voxels)),
            "best_label": self.best_label,
            "best_score": float(self.best_score),
            "reliable_target": bool(reliable),
            "labels": {label: evidence.to_trace() for label, evidence in self.labels.items()},
        }


class GoalLabelResolver:
    def __init__(self, coco_classes: Sequence[str]) -> None:
        self._coco_classes = set(coco_classes)

    def resolve(self, target_object: str) -> GoalLabels:
        requested = _unique(_clean_label(item) for item in target_object.split("|") if _clean_label(item))
        primary = requested[0] if requested else _clean_label(target_object)
        table = STATIC_GOAL_LABELS.get(primary, {"target_aliases": requested, "confusable_labels": []})
        aliases = _unique([*requested, *table.get("target_aliases", [])])
        confusables = _unique(table.get("confusable_labels", []))
        detector_labels = _unique([*aliases, *confusables])
        backend = "yolov7" if all(label in self._coco_classes for label in detector_labels) else "grounding_dino"
        canonical = {label: TARGET_LABEL for label in aliases}
        canonical.update({label: label for label in confusables})
        return GoalLabels(target_object, aliases, confusables, detector_labels, canonical, backend)


class TargetFusionManager:
    def __init__(
        self,
        coco_classes: Sequence[str],
        reliable_confidence_threshold: float = 0.65,
        reliable_min_volume: int = 8,
        reliable_min_positive_observations: int = 2,
        cluster_match_radius_m: float = 0.60,
        cluster_strict_match_radius_m: float = 0.25,
        cluster_min_overlap_voxels: int = 3,
        cluster_bounds_margin_m: float = 0.25,
        lambda_neg: float = 0.20,
        min_negative_overlap_voxels: int = 8,
    ) -> None:
        self._resolver = GoalLabelResolver(coco_classes)
        self._reliable_confidence_threshold = reliable_confidence_threshold
        self._reliable_min_volume = reliable_min_volume
        self._reliable_min_positive_observations = reliable_min_positive_observations
        self._cluster_match_radius_m = cluster_match_radius_m
        self._cluster_strict_match_radius_m = cluster_strict_match_radius_m
        self._cluster_min_overlap_voxels = cluster_min_overlap_voxels
        self._cluster_bounds_margin_m = cluster_bounds_margin_m
        self._lambda_neg = lambda_neg
        self._min_negative_overlap_voxels = min_negative_overlap_voxels
        self.reset()

    def reset(self) -> None:
        self._goal_labels: Optional[GoalLabels] = None
        self._clusters: List[ObjectCluster] = []
        self._next_cluster_id = 1
        self._last_goal_by_cluster: Dict[int, np.ndarray] = {}
        self._last_trace: Dict[str, object] = {"event": "reset", "clusters": []}

    def set_goal(self, target_object: str) -> None:
        self._goal_labels = self._resolver.resolve(target_object)
        self._clusters, self._next_cluster_id, self._last_goal_by_cluster = [], 1, {}
        self._last_trace = {"event": "set_goal", "goal_labels": self.goal_labels.to_trace(), "clusters": []}

    @property
    def goal_labels(self) -> GoalLabels:
        if self._goal_labels is None:
            raise RuntimeError("TargetFusionManager goal has not been set")
        return self._goal_labels

    @property
    def trace(self) -> Dict[str, object]:
        return self._last_trace

    def canonical_label_for_phrase(self, phrase: str) -> str:
        return self.goal_labels.phrase_to_canonical[_clean_label(phrase)]

    def ingest(
        self,
        observations: Iterable[ObjectObservation],
        depth_voxels: Set[VoxelKey],
        robot_xy: np.ndarray,
        step: int,
    ) -> Dict[str, object]:
        events: List[Dict[str, object]] = []
        observed: Dict[int, Set[str]] = {}
        for observation in observations:
            cluster = self._associate(observation)
            if cluster is None:
                cluster = self._create_cluster(observation)
                events.append({"event": "created_cluster", "cluster_id": cluster.cluster_id, "label": observation.canonical_label})
            events.append(cluster.add_positive(observation))
            observed.setdefault(cluster.cluster_id, set()).add(observation.canonical_label)
        for cluster in self._clusters:
            events.extend(cluster.add_soft_negatives(depth_voxels, observed.get(cluster.cluster_id, set()), self._lambda_neg, self._min_negative_overlap_voxels, step))
        reliable = self.reliable_target(robot_xy)
        self._last_trace = {
            "event": "ingest",
            "goal_labels": self.goal_labels.to_trace(),
            "events": events,
            "reliable_target": None if reliable is None else {
                "cluster_id": reliable.cluster_id,
                "goal_xyz": reliable.goal_xyz.tolist(),
                "confidence": float(reliable.confidence),
                "positive_volume": int(reliable.positive_volume),
                "positive_observation_count": int(reliable.positive_observation_count),
            },
            "clusters": [cluster.to_trace(self._is_reliable_target(cluster)) for cluster in self._clusters],
        }
        return self._last_trace

    def reliable_target(self, robot_xy: np.ndarray) -> Optional[FusionTarget]:
        clusters = [cluster for cluster in self._clusters if self._is_reliable_target(cluster)]
        if not clusters:
            return None
        cluster = min(clusters, key=lambda item: float(np.linalg.norm(item.centroid_xyz[:2] - robot_xy)))
        raw_goal = cluster.stable_target_xyz()
        if raw_goal is None:
            return None
        goal = self._stable_goal(cluster.cluster_id, raw_goal, robot_xy)
        evidence = cluster.target_evidence()
        assert evidence is not None
        return FusionTarget(cluster.cluster_id, goal, evidence.fused_confidence, evidence.positive_volume, evidence.positive_observation_count)

    def has_reliable_target(self, robot_xy: np.ndarray) -> bool:
        return self.reliable_target(robot_xy) is not None

    def reliable_target_cloud(self, cluster_id: int) -> np.ndarray:
        for cluster in self._clusters:
            if cluster.cluster_id != cluster_id or not self._is_reliable_target(cluster):
                continue
            evidence = cluster.target_evidence()
            if evidence is not None:
                return evidence.cloud.copy()
        return np.empty((0, 3), dtype=np.float32)

    def _create_cluster(self, observation: ObjectObservation) -> ObjectCluster:
        cluster = ObjectCluster(
            self._next_cluster_id, observation.voxel_cloud.copy(), set(observation.voxel_keys),
            observation.centroid_xyz.copy(), observation.bounds_min.copy(), observation.bounds_max.copy(),
            observation.step, observation.step,
        )
        self._next_cluster_id += 1
        self._clusters.append(cluster)
        return cluster

    def _associate(self, observation: ObjectObservation) -> Optional[ObjectCluster]:
        overlaps = [(len(observation.voxel_keys & cluster.geometry_voxels), cluster) for cluster in self._clusters]
        overlaps = [item for item in overlaps if item[0] >= self._cluster_min_overlap_voxels]
        if overlaps:
            return max(overlaps, key=lambda item: item[0])[1]
        candidates = []
        for cluster in self._clusters:
            distance = float(np.linalg.norm(observation.centroid_xyz - cluster.centroid_xyz))
            if distance <= self._cluster_match_radius_m and (
                distance <= self._cluster_strict_match_radius_m
                or _bounds_overlap(observation.bounds_min, observation.bounds_max, cluster.bounds_min, cluster.bounds_max, self._cluster_bounds_margin_m)
            ):
                candidates.append((distance, cluster))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _is_reliable_target(self, cluster: ObjectCluster) -> bool:
        evidence = cluster.target_evidence()
        return bool(
            cluster.best_label == TARGET_LABEL
            and evidence is not None
            and evidence.fused_confidence >= self._reliable_confidence_threshold
            and evidence.positive_volume >= self._reliable_min_volume
            and evidence.positive_observation_count >= self._reliable_min_positive_observations
        )

    def _stable_goal(self, cluster_id: int, goal: np.ndarray, robot_xy: np.ndarray) -> np.ndarray:
        previous = self._last_goal_by_cluster.get(cluster_id)
        if previous is None:
            self._last_goal_by_cluster[cluster_id] = goal.copy()
            return goal
        delta = float(np.linalg.norm(goal[:2] - previous[:2]))
        if delta < 0.1 or (delta < 0.5 and np.linalg.norm(robot_xy - goal[:2]) > 2.0):
            return previous
        self._last_goal_by_cluster[cluster_id] = goal.copy()
        return goal


def _weighted_mean(old_value: float, old_weight: float, new_value: float, new_weight: float) -> float:
    if new_weight <= 0:
        return old_value
    total = old_weight + new_weight
    return new_value if total <= 0 else (old_value * old_weight + new_value * new_weight) / total


def _bounds_overlap(first_min: np.ndarray, first_max: np.ndarray, second_min: np.ndarray, second_max: np.ndarray, margin: float) -> bool:
    return bool(np.all(first_min <= second_max + margin) and np.all(second_min <= first_max + margin))


def _clean_label(label: str) -> str:
    return label.strip().lower()


def _unique(labels: Iterable[str]) -> List[str]:
    seen, result = set(), []
    for label in labels:
        clean = _clean_label(label)
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
