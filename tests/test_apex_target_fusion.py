import json

import numpy as np

from zson3.target.fusion import TARGET_LABEL, TargetFusionManager
from zson3.target.geometry import ObjectGeometryExtractor, ObjectObservation
from zson3.target.pipeline import COCO_CLASSES


def _observation(
    label: str,
    phrase: str,
    confidence: float,
    step: int,
    x: float = 2.0,
    key_start: int = 0,
    volume: int = 12,
) -> ObjectObservation:
    voxel_keys = {(20 + key_start + i, 0, 5) for i in range(volume)}
    cloud = np.array([[x + i * 0.01, 0.0, 0.5] for i in range(volume)], dtype=np.float32)
    return ObjectObservation(
        cloud=cloud,
        voxel_cloud=cloud,
        voxel_keys=voxel_keys,
        centroid_xyz=np.median(cloud, axis=0),
        bounds_min=np.min(cloud, axis=0),
        bounds_max=np.max(cloud, axis=0),
        camera_xyz=np.array([0.0, 0.0, 0.5], dtype=np.float32),
        confidence=confidence,
        step=step,
        phrase=phrase,
        canonical_label=label,
        detector="yolov7",
    )


def _manager(**kwargs) -> TargetFusionManager:
    manager = TargetFusionManager(
        COCO_CLASSES,
        reliable_confidence_threshold=0.6,
        reliable_min_volume=8,
        reliable_min_positive_observations=2,
        min_negative_overlap_voxels=2,
        **kwargs,
    )
    manager.set_goal("toilet")
    return manager


def test_label_resolver_uses_static_confusables_and_one_backend():
    manager = _manager()
    labels = manager.goal_labels
    assert labels.detector_backend == "yolov7"
    assert labels.phrase_to_canonical["toilet"] == TARGET_LABEL
    assert "sink" in labels.confusable_labels
    json.dumps(manager.trace)


def test_positive_observation_count_is_reliable_gate():
    manager, robot_xy = _manager(), np.zeros(2)
    first = _observation(TARGET_LABEL, "toilet", 0.9, 0)
    manager.ingest([first], first.voxel_keys, robot_xy, 0)
    assert manager.reliable_target(robot_xy) is None
    second = _observation(TARGET_LABEL, "toilet", 0.85, 1, key_start=3)
    manager.ingest([second], first.voxel_keys | second.voxel_keys, robot_xy, 1)
    assert manager.reliable_target(robot_xy).positive_observation_count == 2


def test_reliable_target_goal_uses_cluster_medoid_not_front_surface():
    manager, robot_xy = _manager(), np.zeros(2)
    front = _observation(TARGET_LABEL, "toilet", 0.9, 0, x=2.0, volume=10)
    back = _observation(TARGET_LABEL, "toilet", 0.9, 1, x=4.0, key_start=1, volume=10)
    manager.ingest([front], front.voxel_keys, robot_xy, 0)
    manager.ingest([back], front.voxel_keys | back.voxel_keys, robot_xy, 1)
    assert manager.reliable_target(robot_xy).goal_xy[0] > 2.5


def test_confusable_label_competes_inside_same_physical_cluster():
    manager, robot_xy = _manager(), np.zeros(2)
    first = _observation(TARGET_LABEL, "toilet", 0.9, 0, volume=10)
    second = _observation(TARGET_LABEL, "toilet", 0.9, 1, key_start=2, volume=10)
    manager.ingest([first], first.voxel_keys, robot_xy, 0)
    manager.ingest([second], first.voxel_keys | second.voxel_keys, robot_xy, 1)
    competitor = _observation("sink", "sink", 0.95, 2, volume=24)
    manager.ingest([competitor], first.voxel_keys | competitor.voxel_keys, robot_xy, 2)
    assert manager.reliable_target(robot_xy) is None
    assert manager.trace["clusters"][0]["best_label"] == "sink"


def test_weak_negative_uses_lambda_and_deduplicates_reobserved_voxels():
    manager, robot_xy = _manager(lambda_neg=0.2), np.zeros(2)
    first = _observation(TARGET_LABEL, "toilet", 0.9, 0, volume=20)
    second = _observation(TARGET_LABEL, "toilet", 0.9, 1, key_start=1, volume=20)
    depth_voxels = first.voxel_keys | second.voxel_keys
    manager.ingest([first], depth_voxels, robot_xy, 0)
    manager.ingest([second], depth_voxels, robot_xy, 1)
    before = manager.trace["clusters"][0]["labels"][TARGET_LABEL]["fused_confidence"]
    manager.ingest([], depth_voxels, robot_xy, 2)
    after_first = manager.trace["clusters"][0]["labels"][TARGET_LABEL]
    manager.ingest([], depth_voxels, robot_xy, 3)
    after_duplicate = manager.trace["clusters"][0]["labels"][TARGET_LABEL]
    assert after_first["fused_confidence"] < before
    assert after_first["negative_reobservation_volume"] == after_duplicate["negative_reobservation_volume"]


def test_competitor_flip_keeps_cluster_without_revoked_state():
    manager, robot_xy = _manager(), np.zeros(2)
    target = _observation(TARGET_LABEL, "toilet", 0.88, 0, volume=12)
    again = _observation(TARGET_LABEL, "toilet", 0.86, 1, key_start=1, volume=12)
    chair = _observation("chair", "chair", 0.93, 2, volume=30)
    manager.ingest([target], target.voxel_keys, robot_xy, 0)
    manager.ingest([again], target.voxel_keys | again.voxel_keys, robot_xy, 1)
    manager.ingest([chair], target.voxel_keys | chair.voxel_keys, robot_xy, 2)
    assert manager.trace["reliable_target"] is None
    assert manager.trace["clusters"][0]["best_label"] == "chair"
    assert "revoked" not in json.dumps(manager.trace).lower()


def test_openfrontier_camera_axes_project_to_expected_world_side():
    extractor = ObjectGeometryExtractor(erosion_size=0)
    extractor.use_dbscan = False
    depth = np.full((40, 60), 2.0, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[15:25, 38:48] = 1  # right of principal point
    intrinsic = np.array([[50.0, 0.0, 29.5], [0.0, 50.0, 19.5], [0.0, 0.0, 1.0]])
    world_from_camera = np.eye(4)
    world_from_camera[:3, 3] = [10.0, -3.0, 1.0]
    observation = extractor.extract_observation(
        depth_m=depth,
        object_mask=mask,
        world_from_camera=world_from_camera,
        min_depth_m=0.5,
        max_depth_m=3.5,
        intrinsic=intrinsic,
        confidence=0.9,
        step=0,
        phrase="chair",
        canonical_label=TARGET_LABEL,
        detector="yolov7",
    )
    assert observation is not None
    assert observation.centroid_xyz[0] > 10.0  # image right -> camera/world +x
    assert abs(observation.centroid_xyz[1] + 3.0) < 0.25
    assert abs(observation.centroid_xyz[2] - 3.0) < 0.15  # camera z + 2m
