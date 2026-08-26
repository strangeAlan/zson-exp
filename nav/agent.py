import os
import json
import time
import logging
import argparse
import gc
from typing import Optional, List, Callable
import numpy as np
import torch
import open3d as o3d
import cv2
from vlm.utils import (
    detect_bound_target_candidates,
    detect_frontier_probabilities,
    segment_target_object,
    detect_target_object,
    COMPOSITIONS,
    COMPRESSION,
)

from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

# OpenFrontier
from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from frontier.model.utils.preprocess import preprocess, resize_centercrop_img
from utils.frontier_utils import ft_pos_direct_distance, read_config_yaml

# Mapping
from mapping.wavemap import WaveMapper

# Frontier Manager
from frontier.manager import FrontierManager
from frontier.geometric_completion import GeometricFrontierCompletion
from nav.detected_object import DetectedObject

np.set_printoptions(precision=3, suppress=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product
from planner.planner_base import PlannerBase
from utils.api_key import get_google_api_key, mask_api_key
from utils.vis_utils import is_same_pose, pose_difference
from zson3.services import QwenServiceError, Sam3ServiceError


class NavigationAgent:
    """
    A lightweight wrapper for the navigation loop and state.
    """

    # ---------- constants / defaults ----------
    REFRESH_RATE = 50  # Hz
    VOX_SIZE = 0.1

    def __init__(
        self,
        args: argparse.Namespace,
        target: str,
        planner: PlannerBase,
        cam_intrinsic: np.ndarray,
        save_dir: str,
        bbox: List[float],
        get_rgbd: Callable,
        get_cam_extrinsic: Callable,
        move_fn: Callable,
        rotate_fn: Callable,
        cam_to_agent: np.ndarray = np.eye(4),
        fix_view_level = False
    ) -> None:

        self.args = args
        self.goal = target
        self.bbox = bbox
        self.cam_intrinsic = cam_intrinsic
        self.save_dir = save_dir
        self.save_images = bool(getattr(args, "save_images", False))
        self.C_T_R = cam_to_agent

        self.get_rgbd = get_rgbd
        self.get_cam_extrinsic = get_cam_extrinsic
        self.move_simulation = move_fn
        self.rotate_simulation = rotate_fn

        self.planner: PlannerBase = planner
        # Config
        self.config = read_config_yaml(args.config)
        self.predict_interval: int = int(self.config.get("predict_interval", 5))
        self.plan_interval: int = int(self.config.get("plan_interval", 10))
        self.termination_threshold: float = float(
            self.config.get("termination_threshold", 0.7)
        )
        self.success_threshold: float = float(self.config.get("success_threshold", 1.0))
        self.segmentation_source: str = self.config.get(
            "segmentation_source", "gemini-2.5-flash"
        )
        self.probabilities_source: str = self.config.get(
            "probabilities_source", "gemini-2.5-flash"
        )
        self.detection_source: str = self.config.get(
            "detection_source", "gemini-2.5-flash"
        )

        # Frontier, mapping, detector
        self.mapper: Optional[WaveMapper] = None
        self.ft_manager: Optional[FrontierManager] = None
        self.ft_detector: Optional[FrontierDetector] = None
        self.VOX_SIZE = (
            self.config["voxel_size"]
            if self.config["voxel_size"] is not None
            else self.VOX_SIZE
        )

        use_free = self.config.get("use_free_grid", True)
        use_occ = self.config.get("use_occ_grid", True)
        self.use_map = True

        # Path & motion tracking
        self.path_to_go: List[np.ndarray] = []
        self.move_enough: bool = True
        self.last_W_T_C2: np.ndarray = np.eye(4)

        self.rotated_degrees = 0

        self.vlm_log_file = os.path.join(save_dir, f"vlm_log.txt")
        self.path_taken_file = os.path.join(save_dir, f"navigation_path.npy")
        self.logging_file = os.path.join(save_dir, f"navigation_log.txt")
        self.composition_dir = os.path.join(save_dir, f"compositions")
        self.som_dir = os.path.join(save_dir, f"som_visualizations")
        self.termination_dir = os.path.join(save_dir, f"termination_visualizations")
        self.segmentation_dir = os.path.join(save_dir, f"segmentation_visualizations")
        self.depth_seg_dir = os.path.join(save_dir, f"depth_segmentations")
        args_dict = vars(args)
        self.json_path = args_dict.get(
            "write_path", os.path.join(save_dir, f"navigation_state.json")
        )

        self.planner.logging_file = os.path.join(save_dir, f"navigation_log.txt")

        # Clear directories
        for dir_path in [
            self.composition_dir,
            self.som_dir,
            self.segmentation_dir,
            self.termination_dir,
            self.depth_seg_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))

        self.last_rgb = None
        self.last_som_img = None
        self.path_taken = []

        self.stuck_counter = 0
        self.frozen_counter = 0

        # Reset the files
        open(self.vlm_log_file, "w").close()
        open(self.logging_file, "w").close()

        self.n_images = self.config.get("n_images", 4)

        try:
            self.composition_dims = COMPOSITIONS[self.n_images]
        except KeyError:
            raise ValueError(
                f"Unsupported n_images: {self.n_images}. Supported values are: {list(COMPOSITIONS.keys())}"
            )

        self.segmentation_image = None
        self.composition_images = []
        self.termination_images = []
        self.termination_depths = []
        self.termination_viewpoints = []
        self.composition_depths = []
        self.composition_viewpoints = []
        self.detected_objects = []
        self.goal_object = None
        self.appoaching_object = False
        self.target_diagnostics = {
            "segmentation_events": [],
            "verification_events": [],
            "path_exhausted_recovery_events": [],
            "visibility_events": [],
            "termination_event": None,
        }
        self.target_candidate_bound_verification = bool(
            self.config.get("target_candidate_bound_verification", False)
        )
        self.candidate_verification_cache = {}
        self.target_path_exhausted_recovery = bool(
            self.config.get("target_path_exhausted_recovery", False)
        )
        self.target_path_exhausted_max_retries = int(
            self.config.get("target_path_exhausted_max_retries", 1)
        )
        self.target_reassociation_distance = float(
            self.config.get("target_reassociation_distance", 1.0)
        )
        self.path_exhausted_recovery_attempts = 0
        self.geometry_frontier_enabled = bool(
            self.config.get("geometry_frontier_enabled", False)
        )
        self.geometry_frontier_mode = self.config.get(
            "geometry_frontier_mode", "semantic_fallback_v1"
        )
        self.geometry_completion = (
            GeometricFrontierCompletion(self.config)
            if self.geometry_frontier_enabled
            else None
        )
        self.geometry_override_candidate = None
        self.geometry_diagnostics = []
        self.geometry_force_refresh = False
        self.geometry_cooldowns = {}
        self.geometry_cooldown_steps = int(
            self.config.get("geometry_frontier_cooldown_steps", 120)
        )
        self.geometry_identity_resolution = float(
            self.config.get("geometry_frontier_identity_resolution", 0.5)
        )
        self.attempted_visual_features = []
        self.active_visual_feature = None
        self.geometry_keyframes = []
        self.geometry_keyframe_limit = int(
            self.config.get("geometry_keyframe_limit", 96)
        )
        self.geometry_grounding_min_distance = float(
            self.config.get("geometry_grounding_min_distance", 0.5)
        )
        self.geometry_grounding_max_distance = float(
            self.config.get("geometry_grounding_max_distance", 3.5)
        )
        self.geometry_grounding_min_alignment = float(
            self.config.get("geometry_grounding_min_alignment", 0.5)
        )
        self.geometry_grounding_margin = float(
            self.config.get("geometry_grounding_margin", 0.1)
        )
        self.geometry_grounding_depth_tolerance = float(
            self.config.get("geometry_grounding_depth_tolerance", 0.35)
        )
        self.geometry_grounded_candidate_limit = int(
            self.config.get("geometry_grounded_candidate_limit", 3)
        )
        self.geometry_probability_cache = {}
        self.selector_oracle_file = os.path.join(save_dir, "selector_oracle.jsonl")
        if self.geometry_frontier_enabled:
            open(self.selector_oracle_file, "w").close()

        self.optimal_path_length = float("inf")

        self.pure_exploration = (
            True if self.config.get("seeking_weight", 0.0) == 0.0 else False
        )

        self.google_api_key = get_google_api_key(required=False)
        self.navigation_steps = -1
        self.emergency_rotation = False
        self.fix_view_level = fix_view_level
        self.map_loop = 0
        self.timings = {
            "frontier_detection_step": {"total_time": 0.0, "calls": 0},
            "frontier_detection_global": {"total_time": 0.0, "calls": 0},
            "vlm_probabilities": {"total_time": 0.0, "calls": 0},
            "vlm_call": {"total_time": 0.0, "calls": 0},
            "total_segmentation": {"total_time": 0.0, "calls": 0},
            "mapping_update": {"total_time": 0.0, "calls": 0},
            "frontier_update_no_som": {"total_time": 0.0, "calls": 0},
            "frontier_manager_update": {"total_time": 0.0, "calls": 0},
            "pointnav_planning": {"total_time": 0.0, "calls": 0},
            "navigation_time": {"total_time": 0.0, "calls": 0},
            "navigation_time_no_ai": {"total_time": 0.0, "calls": 0},
            "navigation_time_no_map": {"total_time": 0.0, "calls": 0},
            "navigation_time_no_ai_no_map": {"total_time": 0.0, "calls": 0},
            "move_time": {"total_time": 0.0, "calls": 0},
            "rotate_time": {"total_time": 0.0, "calls": 0},
            "full_frontier_update": {"total_time": 0.0, "calls": 0},
            "sam": {"total_time": 0.0, "calls": 0},
            "move_habitat": {"total_time": 0.0, "calls": 0},
            "read_habitat": {"total_time": 0.0, "calls": 0}
        }
        
    def get_google_api_key(self) -> str:
        if self.google_api_key is None:
            self.google_api_key = get_google_api_key()

        return self.google_api_key

    def get_api_key_for_model(self, model) -> Optional[str]:
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.value
        model_name = model_name.lower()

        if model_name.startswith("gemini") or model_name.endswith("-api"):
            return self.get_google_api_key()

        return None

    def handle_vlm_exception(self, error: Exception, context: str) -> None:
        message = str(error)
        self.log(
            "error",
            self.logging_file,
            f"{context}: {message}",
        )

        message_lower = message.lower()
        if (
            "set gemini_api_key" in message_lower
            or "google_api_key" in message_lower
            or "api key must be provided" in message_lower
        ):
            raise RuntimeError(message) from error

        if "exhausted" in message_lower or "429" in message:
            self.log(
                "info",
                self.logging_file,
                f"Google API key {mask_api_key(self.google_api_key)} is rate limited or quota exhausted.",
            )
            raise RuntimeError("Google API quota exhausted or rate limited.") from error

        if "permission denied" in message_lower or "403" in message:
            self.log(
                "info",
                self.logging_file,
                f"Google API key {mask_api_key(self.google_api_key)} was rejected.",
            )
            raise RuntimeError("Google API permission denied.") from error

    # ---------- main navigation logic ----------

    def initialize(self) -> None:
        assert self.ft_manager is not None and (self.mapper is not None or not self.use_map)

        self.max_steps = self.args.max_steps
        self.max_time_s = self.args.max_time
        self.start_time = time.time()

        # Initial mapping (one frame) to bootstrap map
        rgb, depth0 = self.get_rgbd()

        self.segmentation_image = None
        self.last_som_img = rgb.copy()

        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)

        if self.use_map:
            self.mapper.insert_depth_to_buffer(
                depth=depth0, transform=W_T_C2
            )  # camera IN world frame
            self.log("info", self.logging_file, "Initial mapping round started.")

            self.mapper.integrate_from_buffer()
            self.mapper.interpolate_occupancy_grid()
            og = self.mapper.get_occupancy_grid()
            self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

        self.poses = [W_T_C2.copy()]

        self.is_pointnav = self.planner.is_pointnav_planner

    def navigation(self, save_images: Optional[bool] = None) -> None:
        """
        Main navigation loop.
        """
        if save_images is None:
            save_images = self.save_images

        navigation_start = time.time()
        sam_time = 0.0
        mapping_time = 0.0
        full_frontier_time = 0.0
        som_time = 0.0
        full_frontier_refresh = False
        
        self.timings["frontier_detection_global"]["calls"] += 1
        
        self.navigation_steps += 1
        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)
        self.poses.append(W_T_C2.copy())

        self.planner.nav_level = W_T_C2[2, 3] - self.C_T_R[2, 3]
        self.ft_manager.planner.nav_level = W_T_C2[2, 3] - self.C_T_R[2, 3]
        if hasattr(self, "navmesh_planner"):
            self.navmesh_planner.nav_level = self.planner.nav_level

        if self.fix_view_level:
            self.ft_detector.fix_view_level(W_T_C2[2, 3])

        if self.navigation_steps >= self.max_steps:
            self.log(
                "info", self.logging_file, "Maximum steps reached, navigation finished."
            )
            return False, "max_steps_reached"

        read_start = time.time()
        rgb, depth = self.get_rgbd()
        self.timings["read_habitat"]["total_time"] += (time.time() - read_start)
        self.timings["read_habitat"]["calls"] += 1
        
        # Always keep the latest n_images for termination
        if len(self.termination_images) == self.n_images:
            self.termination_images.pop(0)
            self.termination_depths.pop(0)
            self.termination_viewpoints.pop(0)

        self.composition_images.append(rgb)
        self.termination_images.append(rgb)
        self.termination_depths.append(depth)
        self.termination_viewpoints.append(W_T_C2.copy())
        self.composition_depths.append(depth)
        self.composition_viewpoints.append(W_T_C2.copy())

        if (
            len(self.composition_images) == self.n_images
            and self.ft_manager.object_lockin is None
        ):
            start_segm = time.time()

            rgb_composition = self.compose_images(self.composition_images)
            if save_images:
                composition_path = os.path.join(
                    self.composition_dir,
                    f"{self.navigation_steps:06d}_composition.png",
                )
                self.save_rgb_image(rgb_composition, composition_path)

            try:
                start_sam = time.time()
                response = segment_target_object(
                    rgb_composition=rgb_composition,
                    n_images=self.n_images,
                    target_object=self.goal,
                    segmentation_model=self.segmentation_source,
                    api_key=self.get_api_key_for_model(self.segmentation_source),
                )
                end_sam = time.time()
                self.timings["sam"]["total_time"] += (end_sam - start_sam)
                self.timings["sam"]["calls"] += 1
                sam_time = (end_sam - start_sam)

            except (QwenServiceError, Sam3ServiceError):
                raise
            except Exception as e:
                print(e)
                response = None
                self.handle_vlm_exception(e, "Error during target object segmentation")

            if response is None or len(response) != 2 or len(response[0]) == 0:
                self.log(
                    "info",
                    self.logging_file,
                    f"No masks returned.",
                )
                self.segmentation_image = None
                self.target_diagnostics["segmentation_events"].append(
                    {
                        "step": self.navigation_steps,
                        "mask_count": 0,
                        "candidates": [],
                    }
                )
            else:
                masks, image = response
                self.segmentation_image = np.array(image)
                evidence_image, evidence_labels = self._build_candidate_evidence(
                    masks, self.composition_images
                )
                if save_images:
                    segmentation_path = os.path.join(
                        self.segmentation_dir,
                        f"{self.navigation_steps:06d}_segmentation.png",
                    )
                    self.save_rgb_image(self.segmentation_image, segmentation_path)

                depth_compositions = {}

                for mask_index, mask in enumerate(masks):
                    image_index = mask["image_index"]
                    mask_depth = self.composition_depths[image_index]
                    viewpoint = self.composition_viewpoints[image_index]

                    obj = DetectedObject.from_mask(
                        mask=mask,
                        depth=mask_depth,
                        viewpoint=viewpoint,
                        intrinsic_mat=self.cam_intrinsic.intrinsic_matrix,
                        step=self.navigation_steps,
                        rgb=self.composition_images[image_index],
                        evidence_image=evidence_image,
                        evidence_label=evidence_labels[mask_index],
                        evidence_labels=evidence_labels,
                    )

                    self.detected_objects.append(obj)

                    # For debugging
                    depth_image = (mask_depth * 1000).astype(np.uint16)
                    if depth_compositions.get(image_index) is None:
                        depth_compositions[image_index] = depth_image
                    else:
                        depth_compositions[image_index] += depth_image

                if save_images:
                    for image_index, depth_image in depth_compositions.items():
                        depth_path = os.path.join(
                            self.depth_seg_dir,
                            f"{self.navigation_steps:06d}_{image_index}_depth.png",
                        )
                        cv2.imwrite(depth_path, depth_image)

                self.target_diagnostics["segmentation_events"].append(
                    {
                        "step": self.navigation_steps,
                        "mask_count": len(masks),
                        "candidates": [
                            obj.to_dict()
                            for obj in self.detected_objects[-len(masks) :]
                        ],
                    }
                )

                self.merge_objects()

                if self.detected_objects:
                    new_ids = self.ft_manager.add_robot_poses([W_T_C2])
                    self.ft_manager.add_objects(
                        objects=self.detected_objects, parent_ids=new_ids
                    )
                    self.ft_manager.filter_frontiers()
                    
                self.log(
                    "info",
                    self.logging_file,
                    f"Target object detection - masks obtained: {len(masks)}. Detected objects so far: {len(self.detected_objects)}",
                )

            self.composition_images = []
            self.composition_depths = []
            self.composition_viewpoints = []
            end_segm = time.time()
            self.timings["total_segmentation"]["total_time"] += (end_segm - start_segm)
            self.timings["total_segmentation"]["calls"] += 1
        
        # if time.time() - self.start_time > self.max_time_s:
        # self.log("info", self.logging_file, "Time limit reached, navigation finished.")
        # return False, "time_limit_reached"

        no_more_frontier = (
            len(self.ft_manager.valid_frontiers) == 0 and self.navigation_steps > 10
        )

        reach_next_update = len(self.path_to_go) == 0 or (
            (self.navigation_steps - 1) % self.predict_interval == 0
        )
        

        geometry_in_progress = self.geometry_override_candidate is not None
        if self.geometry_force_refresh or (
            not geometry_in_progress
            and (
                self.emergency_rotation
                or self.ft_manager.object_lockin is None
                and (no_more_frontier or reach_next_update)
            )
        ):
            self.geometry_force_refresh = False
            full_frontier_refresh = True
            full_frontier_start = time.time()
            self.log("info", self.logging_file, "Updating frontiers.")

            ft_start = time.time()
            # Frontier detection + anchoring
            self.ft_detector.detect(
                rgb=rgb,
                depth=depth,
                df_normalizer=self.config["df_normalizer"],
                df_thr=self.config["df_thr"],

            )
            if (
                self.geometry_frontier_enabled
                and self.geometry_frontier_mode == "grounded_unified_v2"
            ):
                self._store_geometry_keyframe(rgb, depth, W_T_C2)
            ft_list = self.ft_detector.anchor_fts(depth=depth, extrinsic=C2_T_W)
            ft_end = time.time()

            self.timings["frontier_detection_step"]["total_time"] += (ft_end - ft_start)
            self.timings["frontier_detection_step"]["calls"] += 1
            self.timings["frontier_detection_global"]["total_time"] += (ft_end - full_frontier_start)
            
            if ft_list is not None and len(ft_list) > 0:
                for i, ft in enumerate(ft_list):
                    ft.label = chr(65 + i) if i < 26 else str(i)  # A-Z, then 0, 1, 2...

                    if "snap_point" in dir(self.planner):
                        ft.pos3d = self.planner.snap_point(ft.pos3d)
                        ft.pos3d[2] += self.C_T_R[1, 3]  # adjust for camera height

            # Add into manager
            if ft_list is not None:
                start_ft_manager = time.time()
                new_ids = self.ft_manager.add_robot_poses([W_T_C2])
                self.ft_manager.add_frontiers(frontiers=ft_list, parent_ids=new_ids)
                self.ft_manager.filter_frontiers()
                self.ft_manager.gain_adjustment()
                self.ft_manager.filter_frontiers()
                end_ft_manager = time.time()
                self.timings["frontier_manager_update"]["total_time"] += (end_ft_manager - start_ft_manager)
                self.timings["frontier_manager_update"]["calls"] += 1   

            # Remove the frontiers that are not valid
            if ft_list is not None and len(ft_list) > 0:
                ft_list = [ft for ft in ft_list if ft.is_valid]

            if not self.pure_exploration and ft_list is not None and len(ft_list) > 0:
                som_start = time.time()

                # visualize and save marks
                image_prompt, labels, unmarked_image = self.ft_detector.get_SoM_img(
                    ft_list=ft_list, radius=20, alpha=0.5
                )

                self.last_som_img = image_prompt.copy()
                if save_images:
                    som_path = os.path.join(
                        self.som_dir,
                        f"{self.navigation_steps:06d}_som.png",
                    )
                    self.save_rgb_image(image_prompt, som_path)
                    unmarked_path = os.path.join(
                        self.som_dir,
                        f"{self.navigation_steps:06d}_unmarked.png",
                    )
                    self.save_rgb_image(unmarked_image, unmarked_path)

                attempts = 0
                while attempts < 10:
                    try:
                        start = time.time()
                        success, result, raw_response = detect_frontier_probabilities(
                            rgb_image=image_prompt,
                            labels=labels,
                            target_object=self.goal,
                            vlm_model=self.probabilities_source,
                            api_key=self.get_api_key_for_model(self.probabilities_source),
                        )
                        end = time.time()
                        self.timings["vlm_call"]["total_time"] += (end - start)
                        self.timings["vlm_call"]["calls"] += 1
                        self.log(
                            "info",
                            self.vlm_log_file,
                            f"Frontier probabilities: {raw_response}",
                        )

                        if (
                            not success
                            or not isinstance(result, dict)
                            or len(result) == 0
                        ):
                            self.log(
                                "error",
                                self.logging_file,
                                f"Invalid or unexpected probabilities returned from VLM. {raw_response}",
                            )
                            probabilities = {}
                            attempts += 1
                            continue
                        else:
                            probabilities = result

                        if probabilities is not None and len(probabilities) > 0:
                            self.log(
                                "info",
                                self.logging_file,
                                f"{len(ft_list)} frontiers detected. {len(probabilities.keys())} with probabilities.",
                            )
                           
                            for i, label in enumerate(labels):
                                if i < len(ft_list):
                                    ft = ft_list[i]
                                    if label in probabilities:
                                        ft.probability = probabilities[label][0]
                                        ft.justification = probabilities[label][1]
                                    else:
                                        ft.probability = 0.5
                                        ft.justification = (
                                            "Failed to get probability from VLM."
                                        )
                            break
                        else:
                            self.log(
                                "error",
                                self.logging_file,
                                f"Invalid or unexpected probabilities returned from VLM. {raw_response}",
                            )
                            attempts += 1
                            continue

                    except (QwenServiceError, Sam3ServiceError):
                        raise
                    except Exception as e:
                        self.handle_vlm_exception(
                            e, "Failed to get frontier probabilities from VLM"
                        )
                        attempts += 1
                        time.sleep(15.0)
                        
                som_end = time.time()
                som_time = (som_end - som_start)
                self.timings["vlm_probabilities"]["total_time"] += som_time
                self.timings["vlm_probabilities"]["calls"] += 1
                
            full_frontier_end = time.time()
            full_frontier_time = (full_frontier_end - full_frontier_start)
            self.timings["full_frontier_update"]["total_time"] += full_frontier_time
            self.timings["full_frontier_update"]["calls"] += 1
            
            frontiers_no_som_time = full_frontier_time - som_time
            self.timings["frontier_update_no_som"]["total_time"] += frontiers_no_som_time
            self.timings["frontier_update_no_som"]["calls"] += 1

        # Update mapper continuously
        if self.use_map:
            if self.map_loop == 5:
                mapping_start = time.time()
                self.mapper.integrate_from_buffer()
                self.mapper.interpolate_occupancy_grid()
                og = self.mapper.get_occupancy_grid()
                self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])
                mapping_end = time.time()
                mapping_time = (mapping_end - mapping_start)
                self.timings["mapping_update"]["total_time"] += mapping_time
                self.timings["mapping_update"]["calls"] += 1
                
                self.map_loop = 0
            else:
                self.map_loop += 1
            
        start_ft_manager = time.time()
        self.ft_manager.gain_adjustment()
        self.ft_manager.filter_frontiers()
        self.ft_manager.merge_frontiers()
        self.ft_manager.filter_frontiers()
        self.ft_manager.update_utility(current_pos=W_T_C2[:3, 3])
        end_ft_manager = time.time()
        self.timings["frontier_manager_update"]["total_time"] += (end_ft_manager - start_ft_manager)
        self.timings["frontier_manager_update"]["calls"] += 1

        if full_frontier_refresh and self.geometry_frontier_enabled:
            self._refresh_geometry_completion(W_T_C2)

        if (
            self.ft_manager.valid_frontiers is None
            or len(self.ft_manager.valid_frontiers) == 0
        ) and self.geometry_override_candidate is None:
            self.log(
                "info", self.logging_file, "No frontiers, adding rotation frontiers..."
            )
            self.ft_manager.add_rotation_frontiers(W_T_C=W_T_C2)
            self.emergency_rotation = True
            if (
                self.ft_manager.valid_frontiers is None
                or len(self.ft_manager.valid_frontiers) == 0
            ):
                self.log("info", self.logging_file, "Still no frontiers, rotating...")
                reach_next_update = False
                self.move_enough = False

                if self.rotate() >= 360:
                    keep_rotating = self.ft_manager.next_emergency_rotation()
                    if not keep_rotating:
                        self.log(
                            "info",
                            self.logging_file,
                            "Maximum emergency rotations reached, ending navigation.",
                        )
                        return False, "no_frontiers"
                    else:
                        self.log(
                            "info",
                            self.logging_file,
                            "Relaxed gain threshold, continuing emergency rotations.",
                        )
                        self.rotated_degrees = 0
        else:
            if self.emergency_rotation:
                self.log(
                    "info",
                    self.logging_file,
                    "Frontiers found, ending emergency rotation.",
                )
            self.emergency_rotation = False
            self.rotated_degrees = 0
            self.ft_manager.reset_emergency_rotation()

        # Replan if needed
        if self.is_pointnav or (reach_next_update and self.move_enough):
            self.log("info", self.logging_file, "Replanning...")
            logging.debug(f"Replanning (interval={self.plan_interval}).")

            pointnav_start = time.time()
            object_candidates = [
                ft for ft in self.ft_manager.valid_frontiers if ft.is_object
            ]
            use_geometry = (
                self.geometry_override_candidate is not None
                and self.ft_manager.object_lockin is None
                and self.goal_object is None
                and not object_candidates
            )
            if use_geometry:
                selected_frontier = self.geometry_override_candidate
                self.path_to_go = (
                    self.ft_manager.plan_path_to_frontier(
                        W_T_C2, selected_frontier, depth=depth
                    )
                    or []
                )
                selected_source = "geometry"
                if not self.path_to_go:
                    transient_status = self.ft_manager.last_transient_plan_status
                    if transient_status == "reached":
                        self._consume_geometry_frontier(
                            selected_frontier, depth, W_T_C2
                        )
                    else:
                        self._finish_geometry_frontier(
                            selected_frontier, transient_status
                        )
            else:
                self.geometry_override_candidate = None
                self.path_to_go = (
                    self.ft_manager.plan_path_to_goal(
                        W_T_C2, depth=depth, use_graph=False
                    )
                    or []
                )
                selected_frontier = (
                    self.ft_manager.get_frontier(
                        self.ft_manager.current_goal_ft_id
                    )
                    if self.ft_manager.current_goal_ft_id is not None
                    else None
                )
                selected_source = (
                    selected_frontier.source
                    if selected_frontier is not None
                    else "object"
                    if self.ft_manager.object_lockin is not None
                    else "visual"
                )
                if (
                    selected_frontier is not None
                    and selected_frontier.source == "visual"
                    and self.path_to_go
                ):
                    feature = self._frontier_feature(selected_frontier)
                    self.active_visual_feature = feature
                    if not self._feature_matches_any(
                        feature, self.attempted_visual_features
                    ):
                        self.attempted_visual_features.append(feature)
                elif selected_frontier is None or selected_frontier.source != "visual":
                    self.active_visual_feature = None
                elif not self.path_to_go:
                    self.active_visual_feature = None
            selected_position = (
                np.asarray(selected_frontier.pos3d, dtype=float).tolist()
                if selected_frontier is not None
                else None
            )
            self.log(
                "info",
                self.logging_file,
                "Frontier selection - "
                f"source={selected_source} position={selected_position}",
            )
            pointnav_end = time.time()
            self.timings["pointnav_planning"]["total_time"] += (pointnav_end - pointnav_start)
            self.timings["pointnav_planning"]["calls"] += 1

            if self.ft_manager.current_goal_ft_id is not None:
                goal_frontier = self.ft_manager.get_frontier(
                    self.ft_manager.current_goal_ft_id
                )

                if goal_frontier.is_object:
                    self.goal_object = goal_frontier.linked_object

            if self.path_to_go:
                self.log(
                    "info",
                    self.logging_file,
                    f"Path to goal found with {len(self.path_to_go)} steps.",
                )
                self.move_enough = False
            else:
                self.path_to_go = []
                self.move_enough = True  # try again next cycle

        # Check if reached target viewpoint
        if self.ft_manager.object_lockin is None and self.goal_object is not None:

            # If planning failed and path is empty, consider reached to trigger detection
            if self.ft_manager.current_goal_pose is None:
                reached = len(self.path_to_go) == 0

            else:
                trans_diff, rot_diff = pose_difference(
                    self.ft_manager.current_goal_pose.reshape(1, 4, 4),
                    W_T_C2.reshape(1, 4, 4),
                )

                self.log(
                    "info",
                    self.logging_file,
                    f"Approaching object viewpoint. Trans diff: {trans_diff[0, 0]:.2f} m, Rot diff: {rot_diff[0, 0]:.2f}",
                )

                same_pose = is_same_pose(
                    self.ft_manager.current_goal_pose,
                    W_T_C2,
                    trans_thre=0.2,
                    rot_thre=0.2,
                )

                reached = same_pose or len(self.path_to_go) == 0

            if reached:
                self.log(
                    "info",
                    self.logging_file,
                    f"Reached object viewpoint, performing target object detection.",
                )

                termination = self.compose_images(self.termination_images)
                if save_images:
                    termination_path = os.path.join(
                        self.termination_dir,
                        f"{self.navigation_steps:06d}_termination.png",
                    )
                    self.save_rgb_image(termination, termination_path)

                attempts = 0
                binding = {
                    "candidate_bound": False,
                    "candidate_label": None,
                    "evidence_step": None,
                }
                while attempts < 10:
                    try:
                        detect_start = time.time()
                        evidence = self.goal_object.best_evidence()
                        if self.target_candidate_bound_verification:
                            if evidence is None:
                                success = True
                                response = {
                                    "probability": 0.0,
                                    "reason": "Selected candidate has no bound SAM evidence.",
                                }
                                raw_response = json.dumps(response)
                            else:
                                cache_key = (
                                    int(evidence.get("step") or -1),
                                    tuple(evidence.get("labels", [])),
                                    self.goal,
                                )
                                cached = self.candidate_verification_cache.get(cache_key)
                                if cached is None:
                                    success, candidate_scores, raw_response = (
                                        detect_bound_target_candidates(
                                            rgb_image=evidence["image"],
                                            labels=evidence.get("labels", []),
                                            target_object=self.goal,
                                            vlm_model=self.detection_source,
                                            api_key=self.get_api_key_for_model(
                                                self.detection_source
                                            ),
                                        )
                                    )
                                    if success:
                                        self.candidate_verification_cache[cache_key] = (
                                            candidate_scores,
                                            raw_response,
                                        )
                                else:
                                    candidate_scores, raw_response = cached
                                    success = True
                                if success:
                                    selected_score = candidate_scores.get(
                                        evidence["label"],
                                        [0.0, "Candidate label missing."],
                                    )
                                    response = {
                                        "probability": float(selected_score[0]),
                                        "reason": str(selected_score[1]),
                                    }
                                    binding = {
                                        "candidate_bound": True,
                                        "candidate_label": evidence["label"],
                                        "evidence_step": evidence.get("step"),
                                    }
                                else:
                                    response = None
                        else:
                            success, response, raw_response = detect_target_object(
                                rgb=termination,
                                target_object=self.goal,
                                vlm_model=self.detection_source,
                                api_key=self.get_api_key_for_model(
                                    self.detection_source
                                ),
                            )
                        detect_end = time.time()
                        self.timings["vlm_call"]["total_time"] += (
                            detect_end - detect_start
                        )
                        self.timings["vlm_call"]["calls"] += 1
                        self.log(
                            "info",
                            self.vlm_log_file,
                            f"Target detection: {raw_response}",
                        )
                        if success:
                            break

                        else:
                            response = None
                            attempts += 1
                            self.log(
                                "info",
                                self.logging_file,
                                f"JSON error during target object detection. {raw_response}",
                            )

                    except (QwenServiceError, Sam3ServiceError):
                        raise
                    except Exception as e:
                        response = None
                        self.handle_vlm_exception(
                            e, "Error during target object detection"
                        )
                        attempts += 1
                        time.sleep(15.0)

                if (
                    response is None
                    or not isinstance(response, dict)
                    or len(response) == 0
                ):
                    self.log(
                        "info",
                        self.logging_file,
                        f"No valid response from VLM for target object detection: {response}",
                    )
                else:
                    probability = response.get("probability", 0.0)
                    reason = response.get("reason", "")
                    self.target_diagnostics["verification_events"].append(
                        {
                            "step": self.navigation_steps,
                            "object_id": self.goal_object.id,
                            "object_centroid": np.asarray(
                                self.goal_object.centroid, dtype=float
                            ).tolist(),
                            "agent_position": W_T_C2[:3, 3].astype(float).tolist(),
                            "probability": float(probability),
                            "threshold": self.termination_threshold,
                            "accepted": bool(
                                probability >= self.termination_threshold
                            ),
                            "reason": reason,
                            **binding,
                        }
                    )
                    self.log(
                        "info",
                        self.logging_file,
                        f"Target object detection - Probability: {probability}, Reason: {reason}",
                    )
                    # Check termination condition
                    if probability >= self.termination_threshold:
                        self.goal_object.verification_status = "true_positive"
                        self.log(
                            "info",
                            self.logging_file,
                            f"Object found with probability {probability}. Approaching object.",
                        )
                        self.ft_manager.lock_into_object(self.goal_object)
                        self.path_exhausted_recovery_attempts = 0
                        self.geometry_override_candidate = None
                        self.path_to_go = (
                            self.ft_manager.plan_path_to_goal(
                                W_T_C2, depth=depth, use_graph=False
                            )
                            or []
                        )
                        verification_event = self.target_diagnostics[
                            "verification_events"
                        ][-1]
                        verification_event["approach_path_steps"] = len(
                            self.path_to_go
                        )
                        verification_event["approach_path_endpoint"] = (
                            self.path_to_go[-1][:3, 3].astype(float).tolist()
                            if self.path_to_go
                            else None
                        )
                    else:
                        self.goal_object.is_valid = False
                        self.goal_object.verification_status = "false_positive"
                        self.log(
                            "info",
                            self.logging_file,
                            f"Object not found (probability {probability}), continuing exploration.",
                        )

                # remove the current goal frontier to avoid repeated checks
                self.goal_object.frontier.set_invalid()
                goal_ft = self.ft_manager.get_frontier(self.ft_manager.current_goal_ft_id)
                if goal_ft is not None:
                    goal_ft.set_invalid()
                self.goal_object = None
                self.ft_manager.filter_frontiers()
                reach_next_update = True
                self.move_enough = True

        if self.ft_manager.object_lockin is not None:
            object_pos = self.ft_manager.object_lockin.centroid
            dist = np.linalg.norm(object_pos - W_T_C2[:3, 3])

            self.log(
                "info",
                self.logging_file,
                f"Approaching locked-in object. Distance to object center: {dist:.2f} m",
            )

            stop_trigger = None
            if dist < self.success_threshold:
                stop_trigger = "centroid_distance"
            elif len(self.path_to_go) == 0:
                if self.target_path_exhausted_recovery:
                    recovery = self._recover_path_exhausted_object(
                        W_T_C=W_T_C2,
                        depth=depth,
                    )
                    if recovery == "stop":
                        object_pos = self.ft_manager.object_lockin.centroid
                        dist = np.linalg.norm(object_pos - W_T_C2[:3, 3])
                        stop_trigger = "distance_after_bound_reobservation"
                    elif recovery == "released":
                        reach_next_update = True
                        self.move_enough = True
                else:
                    stop_trigger = "path_exhausted_legacy"

            if stop_trigger is not None:
                self.target_diagnostics["termination_event"] = {
                    "step": self.navigation_steps,
                    "object_id": self.ft_manager.object_lockin.id,
                    "object_centroid": np.asarray(object_pos, dtype=float).tolist(),
                    "agent_position": W_T_C2[:3, 3].astype(float).tolist(),
                    "distance_to_object": float(dist),
                    "path_exhausted": len(self.path_to_go) == 0,
                    "stop_trigger": stop_trigger,
                    "path_exhausted_recovery_attempts": int(
                        self.path_exhausted_recovery_attempts
                    ),
                    "success_threshold": self.success_threshold,
                }
                self.log(
                    "info",
                    self.logging_file,
                    f"Reached object, navigation finished successfully.",
                )
                return False, "object_found"
            
        # Execute one movement step if path exists
        if self.path_to_go:
            logging.debug("Moving along the path.")
            self.move(steps=1)

        if np.allclose(W_T_C2[:3, 3], self.last_W_T_C2[:3, 3]):
            self.stuck_counter += 1

            if self.stuck_counter == 200:
                self.log(
                    "info",
                    self.logging_file,
                    "Robot stuck detected after 200 steps, ending navigation.",
                )
                return False, "robot_stuck"
        else:
            self.stuck_counter = 0

        if is_same_pose(W_T_C2, self.last_W_T_C2):
            self.frozen_counter += 1

            if self.stuck_counter % 5 == 0:
                self.log(
                    "info",
                    self.logging_file,
                    f"Robot frozen for {self.frozen_counter} steps.",
                )
                reach_next_update = True
                self.move_enough = True
                self.rotate()

        else:
            self.frozen_counter = 0
    
        navigation_time = time.time() - navigation_start
        nav_no_ai = navigation_time - sam_time - full_frontier_time
        nav_no_map = navigation_time - mapping_time
        nav_no_ai_no_map = navigation_time - sam_time - full_frontier_time - mapping_time
        self.timings["navigation_time"]["total_time"] += navigation_time
        self.timings["navigation_time"]["calls"] += 1
        self.timings["navigation_time_no_ai"]["total_time"] += nav_no_ai
        self.timings["navigation_time_no_ai"]["calls"] += 1
        self.timings["navigation_time_no_map"]["total_time"] += nav_no_map
        self.timings["navigation_time_no_map"]["calls"] += 1
        self.timings["navigation_time_no_ai_no_map"]["total_time"] += nav_no_ai_no_map
        self.timings["navigation_time_no_ai_no_map"]["calls"] += 1

        if self.json_path is not None:
            self.ft_manager.write_to_file(
                file_path=self.json_path, detected_objects=self.detected_objects
            )
        
        return True, "continue_navigation"

    def _recover_path_exhausted_object(
        self, W_T_C: np.ndarray, depth: np.ndarray
    ) -> str:
        """Re-observe a locked candidate; path exhaustion itself never authorizes STOP."""
        locked = self.ft_manager.object_lockin
        if locked is None:
            return "released"
        event = {
            "step": int(self.navigation_steps),
            "object_id": int(locked.id),
            "old_centroid": np.asarray(locked.centroid, dtype=float).tolist(),
            "attempt": int(self.path_exhausted_recovery_attempts + 1),
            "outcome": None,
        }
        if (
            self.path_exhausted_recovery_attempts
            >= self.target_path_exhausted_max_retries
        ):
            event["outcome"] = "released_retry_limit"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_retry_limit")
            return "released"

        self.path_exhausted_recovery_attempts += 1
        if not self.termination_images:
            event["outcome"] = "released_no_observation"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_no_observation")
            return "released"

        composition = self.compose_images(self.termination_images)
        start_sam = time.time()
        try:
            response = segment_target_object(
                rgb_composition=composition,
                n_images=self.n_images,
                target_object=self.goal,
                segmentation_model=self.segmentation_source,
                api_key=self.get_api_key_for_model(self.segmentation_source),
            )
        except (QwenServiceError, Sam3ServiceError):
            raise
        except Exception as error:
            event["outcome"] = "released_reobservation_error"
            event["error"] = f"{type(error).__name__}: {error}"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_reobservation_error")
            return "released"
        self.timings["sam"]["total_time"] += time.time() - start_sam
        self.timings["sam"]["calls"] += 1
        if response is None or len(response) != 2 or not response[0]:
            event["outcome"] = "released_no_mask"
            event["mask_count"] = 0
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_no_mask")
            return "released"

        masks, _ = response
        evidence_image, labels = self._build_candidate_evidence(
            masks, self.termination_images
        )
        candidates = []
        for index, mask in enumerate(masks):
            image_index = int(mask.get("image_index", -1))
            if not 0 <= image_index < len(self.termination_depths):
                continue
            candidate = DetectedObject.from_mask(
                mask=mask,
                depth=self.termination_depths[image_index],
                viewpoint=self.termination_viewpoints[image_index],
                intrinsic_mat=self.cam_intrinsic.intrinsic_matrix,
                step=self.navigation_steps,
                rgb=self.termination_images[image_index],
                evidence_image=evidence_image,
                evidence_label=labels[index],
                evidence_labels=labels,
            )
            if candidate.centroid is not None:
                candidates.append(candidate)

        event["mask_count"] = len(candidates)
        if not candidates:
            event["outcome"] = "released_no_3d_candidate"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_no_3d_candidate")
            return "released"

        distances = [
            float(np.linalg.norm(candidate.centroid - locked.centroid))
            for candidate in candidates
        ]
        selected_index = int(np.argmin(distances))
        candidate = candidates[selected_index]
        association_distance = distances[selected_index]
        event["association_distance"] = association_distance
        event["candidate_label"] = candidate.evidence_label
        if association_distance > self.target_reassociation_distance:
            event["outcome"] = "released_no_bound_candidate"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_no_bound_candidate")
            return "released"

        start_vlm = time.time()
        try:
            success, scores, raw_response = detect_bound_target_candidates(
                rgb_image=evidence_image,
                labels=labels,
                target_object=self.goal,
                vlm_model=self.detection_source,
                api_key=self.get_api_key_for_model(self.detection_source),
            )
        except (QwenServiceError, Sam3ServiceError):
            raise
        except Exception as error:
            event["outcome"] = "released_verification_error"
            event["error"] = f"{type(error).__name__}: {error}"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_verification_error")
            return "released"
        self.timings["vlm_call"]["total_time"] += time.time() - start_vlm
        self.timings["vlm_call"]["calls"] += 1
        self.log(
            "info",
            self.vlm_log_file,
            f"Path-exhausted bound target recovery: {raw_response}",
        )
        selected_score = scores.get(
            candidate.evidence_label, [0.0, "Candidate label missing."]
        )
        probability = float(selected_score[0]) if success else 0.0
        event["probability"] = probability
        event["reason"] = str(selected_score[1])
        if probability < self.termination_threshold:
            event["outcome"] = "released_candidate_rejected"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            self._release_locked_object("path_exhausted_candidate_rejected")
            return "released"

        locked.centroid = np.asarray(candidate.centroid, dtype=float)
        locked.viewpoint = np.asarray(candidate.viewpoint, dtype=float)
        locked.last_seen_step = self.navigation_steps
        locked.evidence_observations.extend(candidate.evidence_observations)
        event["new_centroid"] = locked.centroid.tolist()
        distance = float(np.linalg.norm(locked.centroid - W_T_C[:3, 3]))
        event["updated_distance_to_object"] = distance
        if distance < self.success_threshold:
            event["outcome"] = "stop_after_bound_reobservation"
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            return "stop"

        self.path_to_go = (
            self.ft_manager.plan_path_to_goal(
                W_T_C, depth=depth, use_graph=False
            )
            or []
        )
        if self.path_to_go:
            event["outcome"] = "replanned"
            event["replanned_path_steps"] = len(self.path_to_go)
            self.target_diagnostics["path_exhausted_recovery_events"].append(event)
            return "replanned"

        event["outcome"] = "released_no_replan_progress"
        self.target_diagnostics["path_exhausted_recovery_events"].append(event)
        self._release_locked_object("path_exhausted_no_replan_progress")
        return "released"

    def _release_locked_object(self, reason: str) -> None:
        locked = self.ft_manager.object_lockin
        if locked is not None:
            locked.is_valid = False
            locked.verification_status = reason
            if locked.frontier is not None:
                locked.frontier.set_invalid()
        self.ft_manager.object_lockin = None
        self.goal_object = None
        self.path_to_go = []
        self.ft_manager.current_goal_pose = None
        self.ft_manager.current_goal_ft_id = None
        self.ft_manager.filter_frontiers()
        self.log(
            "info",
            self.logging_file,
            f"Released locked target after path exhaustion: {reason}",
        )

    def get_target_diagnostics(self) -> dict:
        """Return episode-level target evidence without changing policy state."""

        return {
            **self.target_diagnostics,
            "final_object_tracks": [obj.to_dict() for obj in self.detected_objects],
            "geometry_frontier": self.geometry_diagnostics,
        }

    def _store_geometry_keyframe(
        self, rgb: np.ndarray, depth: np.ndarray, W_T_C: np.ndarray
    ) -> None:
        """Keep one calibrated RGB-D keyframe per normal OF visual refresh."""
        processed_rgb = np.asarray(
            resize_centercrop_img(
                rgb,
                self.ft_detector.scale_factor,
                (self.ft_detector.img_size_model[1], self.ft_detector.img_size_model[0]),
            )
        ).astype(np.uint8)
        processed_depth = preprocess(
            depth,
            self.ft_detector.scale_factor,
            *self.ft_detector.img_size_model,
            is_depth=True,
            normalize_depth=False,
        ).squeeze().astype(np.float32)
        self.geometry_keyframes.append(
            {
                "step": int(self.navigation_steps),
                "rgb": processed_rgb,
                "depth": processed_depth,
                "K": np.asarray(self.ft_detector.pro_intrin, dtype=float).copy(),
                "W_T_C": np.asarray(W_T_C, dtype=float).copy(),
            }
        )
        if len(self.geometry_keyframes) > self.geometry_keyframe_limit:
            self.geometry_keyframes.pop(0)

    @staticmethod
    def _project_world_point(point: np.ndarray, keyframe: dict):
        camera_point = np.linalg.inv(keyframe["W_T_C"]) @ np.append(point, 1.0)
        if camera_point[2] <= 1e-6:
            return None
        intrinsic = keyframe["K"]
        u = float(intrinsic[0, 0] * camera_point[0] / camera_point[2] + intrinsic[0, 2])
        v = float(intrinsic[1, 1] * camera_point[1] / camera_point[2] + intrinsic[1, 2])
        return u, v, float(camera_point[2])

    def _ground_geometry_candidate(self, frontier) -> bool:
        """Attach the best real historical RGB-D view to one geometry proposal."""
        best = None
        rejection_counts = {
            "behind_camera": 0,
            "image_margin": 0,
            "distance": 0,
            "alignment": 0,
            "occluded": 0,
        }
        direction = np.asarray(frontier.view_direction, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        anchor = np.asarray(frontier.evidence_anchor, dtype=float)
        for keyframe in self.geometry_keyframes:
            projection = self._project_world_point(anchor, keyframe)
            if projection is None:
                rejection_counts["behind_camera"] += 1
                continue
            u, v, camera_depth = projection
            height, width = keyframe["depth"].shape
            margin_x = self.geometry_grounding_margin * width
            margin_y = self.geometry_grounding_margin * height
            if not (
                margin_x <= u < width - margin_x
                and margin_y <= v < height - margin_y
            ):
                rejection_counts["image_margin"] += 1
                continue
            distance = float(
                np.linalg.norm(anchor - keyframe["W_T_C"][:3, 3])
            )
            if not (
                self.geometry_grounding_min_distance
                <= distance
                <= self.geometry_grounding_max_distance
            ):
                rejection_counts["distance"] += 1
                continue
            camera_forward = np.asarray(keyframe["W_T_C"][:3, 2], dtype=float)
            camera_forward /= max(float(np.linalg.norm(camera_forward)), 1e-8)
            alignment = float(np.dot(camera_forward, direction))
            if alignment < self.geometry_grounding_min_alignment:
                rejection_counts["alignment"] += 1
                continue
            pixel_u = int(round(u))
            pixel_v = int(round(v))
            observed_depth = float(keyframe["depth"][pixel_v, pixel_u])
            if (
                observed_depth > 1e-3
                and camera_depth
                > observed_depth + self.geometry_grounding_depth_tolerance
            ):
                rejection_counts["occluded"] += 1
                continue
            center_distance = float(
                np.linalg.norm(
                    np.array([u / width, v / height]) - np.array([0.5, 0.5])
                )
            )
            score = alignment - 0.5 * center_distance - 0.1 * abs(distance - 1.75)
            if best is None or score > best[0]:
                best = (score, keyframe, (pixel_u, pixel_v))
        if best is None:
            frontier.grounding_rejections = rejection_counts
            return False
        frontier.evidence_keyframe = best[1]
        frontier.evidence_pixel = best[2]
        frontier.evidence_keyframe_step = int(best[1]["step"])
        frontier.grounding_rejections = rejection_counts
        return True

    def _mark_grounded_geometry(self, frontier, label: str) -> np.ndarray:
        image = np.asarray(frontier.evidence_keyframe["rgb"]).copy()
        pixel = tuple(int(value) for value in frontier.evidence_pixel)
        cv2.circle(image, pixel, 25, (255, 255, 255), 4)
        cv2.circle(image, pixel, 21, (32, 220, 255), 3)
        cv2.putText(
            image,
            label,
            (pixel[0] - 11, pixel[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        arrow_point = np.asarray(frontier.evidence_anchor, dtype=float) + 0.75 * np.asarray(
            frontier.view_direction, dtype=float
        )
        arrow_projection = self._project_world_point(
            arrow_point, frontier.evidence_keyframe
        )
        if arrow_projection is not None:
            endpoint = (int(round(arrow_projection[0])), int(round(arrow_projection[1])))
            if 0 <= endpoint[0] < image.shape[1] and 0 <= endpoint[1] < image.shape[0]:
                cv2.arrowedLine(
                    image, pixel, endpoint, (32, 220, 255), 4, tipLength=0.25
                )
        return image

    @staticmethod
    def _compose_square_panels(images: list[np.ndarray]) -> np.ndarray:
        if len(images) == 1:
            return images[0]
        height, width = images[0].shape[:2]
        columns = 2
        rows = int(np.ceil(len(images) / columns))
        canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
        for index, image in enumerate(images):
            row, column = divmod(index, columns)
            canvas[
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
            ] = image
        return canvas

    def _score_grounded_geometry(self, candidates: list) -> list:
        accepted = []
        missing = []
        for candidate in candidates:
            identity = self._geometry_identity(candidate.navigation_point)
            cache_key = (
                identity,
                tuple(
                    np.round(
                        np.asarray(candidate.view_direction, dtype=float), 2
                    ).tolist()
                ),
                int(candidate.evidence_keyframe_step),
                self.goal,
            )
            candidate.geometry_identity = identity
            candidate.probability_cache_key = cache_key
            cached = self.geometry_probability_cache.get(cache_key)
            if cached is None:
                missing.append(candidate)
            else:
                candidate.probability, candidate.justification = cached
                accepted.append(candidate)

        if missing:
            labels = [
                chr(ord("G") + index) if index < 20 else str(index)
                for index in range(len(missing))
            ]
            panels = []
            for candidate, label in zip(missing, labels):
                candidate.label = label
                panels.append(self._mark_grounded_geometry(candidate, label))
            evidence = self._compose_square_panels(panels)
            start = time.time()
            success, scores, raw_response = detect_frontier_probabilities(
                rgb_image=evidence,
                labels=labels,
                target_object=self.goal,
                vlm_model=self.probabilities_source,
                api_key=self.get_api_key_for_model(self.probabilities_source),
            )
            elapsed = time.time() - start
            self.timings["vlm_call"]["total_time"] += elapsed
            self.timings["vlm_call"]["calls"] += 1
            self.timings["vlm_probabilities"]["total_time"] += elapsed
            self.timings["vlm_probabilities"]["calls"] += 1
            self.log(
                "info",
                self.vlm_log_file,
                f"Grounded geometry probabilities: {raw_response}",
            )
            if success:
                for candidate, label in zip(missing, labels):
                    score = scores.get(label)
                    if not isinstance(score, (list, tuple)) or len(score) < 2:
                        continue
                    candidate.probability = float(score[0])
                    candidate.justification = str(score[1])
                    self.geometry_probability_cache[
                        candidate.probability_cache_key
                    ] = (candidate.probability, candidate.justification)
                    accepted.append(candidate)
        return accepted

    def _refresh_grounded_geometry_pool(self, W_T_C: np.ndarray) -> None:
        """Score grounded geometry and visual proposals in one OF utility space."""
        self.geometry_override_candidate = None
        if self.mapper is None or self.ft_manager.object_lockin is not None:
            return
        self._expire_geometry_cooldowns()
        projection = self.mapper.get_navigation_projection(
            nav_level=self.planner.nav_level,
            min_height=float(
                self.config.get("geometry_frontier_slice_min_height", 0.1)
            ),
            max_height=float(
                self.config.get("geometry_frontier_slice_max_height", 1.3)
            ),
            height_step=float(
                self.config.get("geometry_frontier_slice_height_step", 0.2)
            ),
        )
        visual = [
            frontier
            for frontier in self.ft_manager.valid_frontiers
            if frontier.source == "visual"
            and frontier.justification != "Rotation Frontier"
        ]
        object_candidates = [
            frontier
            for frontier in self.ft_manager.valid_frontiers
            if frontier.is_object
        ]
        generated = self.geometry_completion.generate(
            projection=projection,
            current_position=W_T_C[:3, 3],
            nav_level=self.planner.nav_level,
            planner=self.navmesh_planner,
            unreachable_positions=self.ft_manager._unreachable_positions,
            suppressed_positions=[
                np.asarray(record["position"], dtype=float)
                for record in self.geometry_cooldowns.values()
            ],
            suppression_distance=self.geometry_identity_resolution,
            bbox=self.bbox,
            occupancy_planner=self.planner,
        )
        # Compare geometry and visual proposals at the same camera-height frame.
        # The navmesh point remains separately available for PointNav.
        for candidate in generated:
            candidate.navigation_point = np.asarray(
                candidate.navigation_point, dtype=float
            )
            candidate.pos3d = candidate.navigation_point.copy()
            candidate.pos3d[2] = float(W_T_C[2, 3])
            candidate.evidence_anchor = np.asarray(
                candidate.evidence_anchor, dtype=float
            ).copy()
            candidate.evidence_anchor[2] = float(W_T_C[2, 3])
            candidate.pose6d = self.ft_manager.get_frontier_pose(candidate)
        unmatched = self.geometry_completion.unmatched(generated, visual)

        self.ft_manager.adjust_transient_frontier_gains(unmatched)
        gain_eligible = [
            candidate
            for candidate in unmatched
            if candidate.u_gain >= self.ft_manager.filter_min_gain
        ]
        grounded = [
            candidate
            for candidate in gain_eligible
            if self._ground_geometry_candidate(candidate)
        ]
        for candidate in grounded:
            candidate.probability = 1.0
        self.ft_manager.update_transient_frontier_utilities(
            grounded, W_T_C[:3, 3]
        )
        grounded.sort(key=lambda candidate: candidate.utility, reverse=True)
        grounded = grounded[: self.geometry_grounded_candidate_limit]
        scored = self._score_grounded_geometry(grounded)
        self.ft_manager.update_transient_frontier_utilities(
            scored, W_T_C[:3, 3]
        )

        best_geometry = max(
            scored,
            key=lambda candidate: float(candidate.utility),
            default=None,
        )
        best_visual = max(
            visual,
            key=lambda candidate: (
                float(candidate.utility)
                if candidate.utility is not None and np.isfinite(candidate.utility)
                else -np.inf
            ),
            default=None,
        )
        best_geometry_utility = (
            float(best_geometry.utility) if best_geometry is not None else 0.0
        )
        best_visual_utility = (
            float(best_visual.utility)
            if best_visual is not None
            and best_visual.utility is not None
            and np.isfinite(best_visual.utility)
            else 0.0
        )
        selected = (
            best_geometry
            if best_geometry is not None
            and (best_visual is None or best_geometry_utility > best_visual_utility)
            else None
        )
        if object_candidates or self.goal_object is not None:
            selected = None
        self.geometry_override_candidate = selected
        event = {
            "event": "grounded_unified_refresh",
            "step": int(self.navigation_steps),
            "visual_count": len(visual),
            "geometric_count": len(generated),
            "unmatched_geometric_count": len(unmatched),
            "gain_eligible_geometric_count": len(gain_eligible),
            "grounded_geometric_count": len(grounded),
            "scored_geometric_count": len(scored),
            "best_visual_utility": best_visual_utility,
            "best_geometry_utility": best_geometry_utility,
            "selected_source": "geometry" if selected is not None else "visual",
            "geometry_override_position": (
                np.asarray(selected.navigation_point, dtype=float).tolist()
                if selected is not None
                else None
            ),
            "geometry_candidates": [
                {
                    **candidate.to_dict(),
                    "navigation_point": np.asarray(
                        candidate.navigation_point, dtype=float
                    ).tolist(),
                    "evidence_keyframe_step": int(
                        candidate.evidence_keyframe_step
                    ),
                    "evidence_pixel": list(candidate.evidence_pixel),
                    "raw_geometry_gain": float(candidate.raw_geometry_gain),
                    "theoretical_gain": float(candidate.theoretical_gain),
                    "theoretical_u_gain": float(candidate.theoretical_u_gain),
                }
                for candidate in scored
            ],
            "grounding_rejections": [
                {
                    "navigation_point": np.asarray(
                        candidate.navigation_point, dtype=float
                    ).tolist(),
                    "evidence_anchor": np.asarray(
                        candidate.evidence_anchor, dtype=float
                    ).tolist(),
                    "counts": candidate.grounding_rejections,
                }
                for candidate in gain_eligible
                if not hasattr(candidate, "evidence_keyframe")
            ],
        }
        self.geometry_diagnostics.append(event)
        self.log(
            "info",
            self.logging_file,
            "Grounded geometry pool - "
            f"visual={len(visual)} generated={len(generated)} "
            f"unmatched={len(unmatched)} gain_eligible={len(gain_eligible)} "
            f"grounded={len(grounded)} "
            f"scored={len(scored)} visual_utility={best_visual_utility:.6f} "
            f"geometry_utility={best_geometry_utility:.6f} "
            f"selected_source={event['selected_source']}",
        )
        self._record_selector_oracle(visual + scored, visual, selected)

    def _refresh_geometry_completion(self, W_T_C: np.ndarray) -> None:
        """Offer one geometry fallback only after visual opportunities are exhausted."""
        if self.geometry_frontier_mode == "grounded_unified_v2":
            self._refresh_grounded_geometry_pool(W_T_C)
            return
        self.geometry_override_candidate = None
        if self.mapper is None or self.ft_manager.object_lockin is not None:
            return

        self._expire_geometry_cooldowns()

        projection = self.mapper.get_navigation_projection(
            nav_level=self.planner.nav_level,
            min_height=float(
                self.config.get("geometry_frontier_slice_min_height", 0.1)
            ),
            max_height=float(
                self.config.get("geometry_frontier_slice_max_height", 1.3)
            ),
            height_step=float(
                self.config.get("geometry_frontier_slice_height_step", 0.2)
            ),
        )
        visual = [
            ft
            for ft in self.ft_manager.valid_frontiers
            if not ft.is_object and ft.justification != "Rotation Frontier"
        ]
        object_candidates = [
            ft for ft in self.ft_manager.valid_frontiers if ft.is_object
        ]
        fresh_visual = [
            ft
            for ft in visual
            if not self._feature_matches_any(
                self._frontier_feature(ft), self.attempted_visual_features
            )
        ]
        active_visual = bool(
            self.active_visual_feature is not None
            and any(
                self._features_match(
                    self._frontier_feature(ft), self.active_visual_feature
                )
                for ft in visual
            )
        )
        geometric = self.geometry_completion.generate(
            projection=projection,
            current_position=W_T_C[:3, 3],
            nav_level=self.planner.nav_level,
            planner=self.navmesh_planner,
            unreachable_positions=self.ft_manager._unreachable_positions,
            suppressed_positions=[
                np.asarray(item["position"], dtype=float)
                for item in self.geometry_cooldowns.values()
            ],
            suppression_distance=self.geometry_identity_resolution,
            bbox=self.bbox,
            occupancy_planner=self.planner,
        )
        best_geometry, stats = self.geometry_completion.select_completion(
            geometric=geometric,
            visual=visual,
            projection=projection,
            current_position=W_T_C[:3, 3],
            planner=self.navmesh_planner,
        )
        eligibility_reason = None
        if not visual:
            eligibility_reason = "no_visual_frontier"
        elif not fresh_visual and not active_visual:
            eligibility_reason = "all_visual_frontiers_attempted"

        completion = best_geometry if eligibility_reason is not None else None
        if object_candidates or self.goal_object is not None:
            completion = None
            eligibility_reason = "object_priority"
        self.geometry_override_candidate = completion
        event = {
            "event": "refresh",
            "step": int(self.navigation_steps),
            "visual_count": len(visual),
            "fresh_visual_count": len(fresh_visual),
            "active_visual_in_progress": active_visual,
            "geometric_count": stats.geometric_count,
            "unmatched_geometric_count": stats.unmatched_count,
            "best_visual_coverage": stats.best_visual_coverage,
            "best_geometry_coverage": stats.best_geometry_coverage,
            "geometry_eligibility_reason": eligibility_reason,
            "selected_source": "geometry" if completion is not None else "visual",
            "geometry_override_position": (
                np.asarray(completion.pos3d, dtype=float).tolist()
                if completion is not None
                else None
            ),
            "frontiers": [ft.to_dict() for ft in visual + geometric],
        }
        self.geometry_diagnostics.append(event)
        self.log(
            "info",
            self.logging_file,
            "Geometry completion - "
            f"visual={len(visual)} fresh_visual={len(fresh_visual)} "
            f"active_visual={active_visual} geometry={stats.geometric_count} "
            f"unmatched={stats.unmatched_count} "
            f"best_visual_coverage={stats.best_visual_coverage:.4f} "
            f"best_geometry_coverage={stats.best_geometry_coverage:.4f} "
            f"eligibility={eligibility_reason} "
            f"selected_source={event['selected_source']} "
            f"override_position={event['geometry_override_position']}",
        )
        self._record_selector_oracle(visual + geometric, visual, completion)

    def _frontier_feature(self, frontier) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(frontier.pos3d, dtype=float),
                np.asarray(frontier.view_direction, dtype=float),
            )
        )

    def _features_match(self, first: np.ndarray, second: np.ndarray) -> bool:
        return ft_pos_direct_distance(
            first,
            second,
            weights=list(self.geometry_completion.match_weights),
        ) <= self.geometry_completion.match_threshold

    def _feature_matches_any(self, feature: np.ndarray, features: list) -> bool:
        return any(self._features_match(feature, item) for item in features)

    def _geometry_identity(self, position: np.ndarray) -> tuple:
        resolution = max(self.geometry_identity_resolution, 1e-6)
        return tuple(
            np.round(np.asarray(position, dtype=float)[:2] / resolution).astype(int)
        )

    def _expire_geometry_cooldowns(self) -> None:
        self.geometry_cooldowns = {
            identity: record
            for identity, record in self.geometry_cooldowns.items()
            if int(record["until_step"]) > self.navigation_steps
        }

    def _finish_geometry_frontier(self, frontier, status: str) -> None:
        identity = self._geometry_identity(frontier.pos3d)
        self.geometry_cooldowns[identity] = {
            "position": np.asarray(frontier.pos3d, dtype=float).tolist(),
            "until_step": int(self.navigation_steps + self.geometry_cooldown_steps),
        }
        self.geometry_diagnostics.append(
            {
                "event": "completed" if status == "reached" else "failed",
                "step": int(self.navigation_steps),
                "status": status,
                "position": np.asarray(frontier.pos3d, dtype=float).tolist(),
                "heading_to_unknown": np.asarray(
                    frontier.view_direction, dtype=float
                ).tolist(),
                "pose_error": self.ft_manager.last_transient_pose_error,
                "cooldown_until_step": int(
                    self.navigation_steps + self.geometry_cooldown_steps
                ),
            }
        )
        self.geometry_override_candidate = None
        self.path_to_go = []

    def _consume_geometry_frontier(
        self, frontier, depth: np.ndarray, W_T_C: np.ndarray
    ) -> None:
        """Consume a viewpoint only after PointNav has faced its unknown side."""
        self._finish_geometry_frontier(frontier, "reached")
        if self.use_map:
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C)
            self.mapper.integrate_from_buffer()
            self.mapper.interpolate_occupancy_grid()
            occupancy = self.mapper.get_occupancy_grid()
            self.ft_manager.update_map(
                free_map=occupancy["free"], occ_map=occupancy["occupied"]
            )
            self.map_loop = 0
        # The next loop runs the untouched FrontierNet + Qwen refresh from the
        # just-aligned camera view before geometry can become eligible again.
        self.geometry_force_refresh = True

    def _record_selector_oracle(
        self, candidates: list, visual: list, override
    ) -> None:
        """Log the navmesh-best candidate to GT; never feed it back to policy."""
        goal_points = getattr(self, "oracle_goal_points", [])
        if not goal_points or not candidates:
            return
        records = []
        for frontier in candidates:
            distances = [
                self.navmesh_planner.geodesic_distance(frontier.pos3d, goal_point)
                for goal_point in goal_points
            ]
            records.append(
                {
                    "source": frontier.source,
                    "id": int(frontier.id),
                    "position": np.asarray(frontier.pos3d, dtype=float).tolist(),
                    "coverage": frontier.coverage,
                    "distance_to_gt": min(distances, default=float("inf")),
                }
            )
        finite = [item for item in records if np.isfinite(item["distance_to_gt"])]
        oracle = min(finite, key=lambda item: item["distance_to_gt"], default=None)
        qwen_choice = max(
            visual,
            key=lambda ft: (
                float(ft.utility) if np.isfinite(ft.utility) else -np.inf,
                float(ft.u_gain) if ft.u_gain is not None else -np.inf,
                -int(ft.id),
            ),
            default=None,
        )
        payload = {
            "step": int(self.navigation_steps),
            "oracle_definition": "minimum candidate-to-GT-viewpoint navmesh distance",
            "oracle": oracle,
            "qwen_visual_choice_id": int(qwen_choice.id) if qwen_choice else None,
            "geometry_override_position": (
                np.asarray(override.pos3d, dtype=float).tolist()
                if override is not None
                else None
            ),
            "candidates": records,
        }
        with open(self.selector_oracle_file, "a") as stream:
            stream.write(json.dumps(payload, allow_nan=True) + "\n")

    def move(self, steps: int) -> None:
        """
        Execute up to `steps` motions along the path, acquire depth, and update mapper & manager.
        """
        start = time.time()
        self.rotated_degrees = 0

        if not self.path_to_go:
            self.log("info", self.logging_file, "No path to follow.")
            return

        for _ in range(steps):
            if not self.path_to_go:
                self.log("info", self.logging_file, "Path exhausted.")
                break

            if not self.is_pointnav:
                self.next_W_T_C = self.path_to_go.pop(0)

            move_start = time.time()
            self.move_simulation()
            move_end = time.time()
            self.timings["move_habitat"]["total_time"] += (move_end - move_start)
            self.timings["move_habitat"]["calls"] += 1

            self.log(
                "info",
                self.logging_file,
                f" -------Current navigation step: {self.navigation_steps} -------",
            )


            # Capture new depth
            read_start = time.time()
            _, depth = self.get_rgbd()
            read_end = time.time()
            self.timings["read_habitat"]["total_time"] += (read_end - read_start)
            self.timings["read_habitat"]["calls"] += 1

            # Insert into mapper
            C2_T_W = self.get_cam_extrinsic()
            W_T_C2 = np.linalg.inv(C2_T_W)

            if self.use_map:
                self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C2)

            # If we truly moved, update path bookkeeping
            if not is_same_pose(self.last_W_T_C2, W_T_C2):
                self.last_W_T_C2 = W_T_C2.copy()
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C2])
                self.move_enough = True
                
        self.timings["move_time"]["total_time"] += (time.time() - start)
        self.timings["move_time"]["calls"] += 1


    def get_agent_pose(self, W_T_C2: np.ndarray) -> np.ndarray:
        """
        Get the agent's pose in world coordinates given the camera extrinsic.
        """
        W_T_A = W_T_C2 @ self.C_T_R
        return W_T_A

    def get_camera_pose(self, W_T_A: np.ndarray) -> np.ndarray:
        """
        Get the camera's pose in world coordinates given the agent extrinsic.
        """
        A_T_C2 = np.linalg.inv(self.C_T_R)
        W_T_C2 = W_T_A @ A_T_C2
        return W_T_C2

    def plot_3d(self, W_T_C2, depth=None) -> None:

        axes = plt.figure().add_subplot(111, projection="3d")
        for pose in [self.poses[-1]]:
            robot_pos = pose[:3, 3]
            robot_dir_x = pose[:3, 0]
            robot_dir_y = pose[:3, 1]
            robot_dir_z = pose[:3, 2]
            axes.quiver(
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_dir_x[0],
                robot_dir_x[1],
                robot_dir_x[2],
                length=0.2,
                color="r",
                arrow_length_ratio=0.3,
            )  # X axis
            axes.quiver(
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_dir_y[0],
                robot_dir_y[1],
                robot_dir_y[2],
                length=0.2,
                color="g",
                arrow_length_ratio=0.3,
            )  # Y axis
            axes.quiver(
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_dir_z[0],
                robot_dir_z[1],
                robot_dir_z[2],
                length=0.2,
                color="b",
                arrow_length_ratio=0.3,
            )  # Z axis

        # Set limits around bbox
        min_min = min(self.bbox)
        max_max = max(self.bbox)

        axes.set_xlim(min_min - 1, max_max + 1)
        axes.set_ylim(min_min - 1, max_max + 1)
        axes.set_zlim(min_min - 1, max_max + 1)

        # label axes
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")

        # Draw x, y, z axes as red, green, blue arrows
        axes.quiver(
            0,
            0,
            0,
            1,
            0,
            0,
            length=1.0,
            color="r",
            arrow_length_ratio=0.1,
        )  # X axis

        axes.quiver(
            0,
            0,
            0,
            0,
            1,
            0,
            length=1.0,
            color="g",
            arrow_length_ratio=0.1,
        )  # Y axis

        axes.quiver(
            0,
            0,
            0,
            0,
            0,
            1,
            length=1.0,
            color="b",
            arrow_length_ratio=0.1,
        )  # Z axis

        if depth is not None:
            # Plot the depth points
            height, width = depth.shape

            points = self.depth_to_point_cloud(
                depth, self.cam_intrinsic.intrinsic_matrix
            )  # Nx3

            points_homogeneous = np.hstack(
                (points, np.ones((points.shape[0], 1), dtype=points.dtype))
            )  # Nx4

            # Convert points to world coordinates
            points = (W_T_C2 @ points_homogeneous.T).T[:, :3]  # Nx3

            # Plot one every 10 points
            points_sparse = points[::10, :]

            axes.scatter(
                points_sparse[:, 0],
                points_sparse[:, 1],
                points_sparse[:, 2],
                c="gray",
                s=10,
            )

        if self.ft_manager is not None:
            # Plot frontiers as arrows along the z direction
            for ft in self.ft_manager.all_frontiers:
                pose = ft.pose6d
                ft_pos = pose[:3, 3]
                ft_dir = pose[:3, 2]  # Z direction
                axes.quiver(
                    ft_pos[0],
                    ft_pos[1],
                    ft_pos[2],
                    ft_dir[0],
                    ft_dir[1],
                    ft_dir[2],
                    length=0.3,
                    color="magenta",
                    arrow_length_ratio=0.3,
                )

        # View the plot so +x is up and +z is right
        axes.view_init(elev=90, azim=-90)

        # Draw bounding box
        if self.bbox is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self.bbox
            r = [
                [x_min, x_max],
                [y_min, y_max],
                [z_min, z_max],
            ]
            for s, e in combinations(
                np.array(list(product(*r))), 2
            ):  # all combinations of corners
                if (
                    np.sum(np.abs(s - e)) == r[0][1] - r[0][0]
                    or np.sum(np.abs(s - e)) == r[1][1] - r[1][0]
                    or np.sum(np.abs(s - e)) == r[2][1] - r[2][0]
                ):
                    axes.plot3D(*zip(s, e), color="black")

        occ_map = self.ft_manager.occ_map

        axes.scatter(
            occ_map[:, 0],
            occ_map[:, 1],
            occ_map[:, 2],
            c="red",
            s=10,
        )

        # Plot path to goal
        if self.path_to_go:
            path_points = np.array([pose[:3, 3] for pose in self.path_to_go])  # Nx3
            axes.plot(
                path_points[:, 0],
                path_points[:, 1],
                path_points[:, 2],
                c="orange",
                linewidth=2,
            )

        plt.show(block=True)

    def rotate(self):
        """
        Rotate in place 30 degrees
        """
        start = time.time()        

        self.next_W_T_C = np.linalg.inv(self.get_cam_extrinsic())
        R_curr = self.next_W_T_C[:3, :3]
        R_delta = R.from_euler("Z", -30, degrees=True).as_matrix()
        R_next = R_delta @ R_curr
        self.next_W_T_C[:3, :3] = R_next
        self.rotated_degrees += 30

        move_start = time.time()
        self.rotate_simulation()
        move_end = time.time()
        self.timings["move_habitat"]["total_time"] += (move_end - move_start)
        self.timings["move_habitat"]["calls"] += 1

        # Capture new depth
        read_start = time.time()
        _, depth = self.get_rgbd()
        read_end = time.time()
        self.timings["read_habitat"]["total_time"] += (read_end - read_start)
        self.timings["read_habitat"]["calls"] += 1

        # Insert into mapper
        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)
        if self.use_map:
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C2)

        # If we truly moved, update path bookkeeping
        if not np.allclose(W_T_C2, self.last_W_T_C2):
            self.last_W_T_C2 = W_T_C2
            if self.ft_manager is not None:
                self.ft_manager.add_robot_poses([W_T_C2])
            self.move_enough = True

        end = time.time()
        self.timings["rotate_time"]["total_time"] += (end - start)
        self.timings["rotate_time"]["calls"] += 1
        return self.rotated_degrees

    def save_rgb_image(self, image: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output = np.asarray(image)
        if np.max(output) <= 1:
            output = (output * 255).astype(np.uint8)
        else:
            output = output.astype(np.uint8)
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        elif output.shape[2] == 4:
            output = output[:, :, :3]
        cv2.imwrite(path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.composition_images.clear()
        self.termination_images.clear()
        self.termination_depths.clear()
        self.termination_viewpoints.clear()
        self.geometry_keyframes.clear()
        self.geometry_probability_cache.clear()
        self.composition_depths.clear()
        self.composition_viewpoints.clear()
        self.last_rgb = None
        self.last_som_img = None
        self.segmentation_image = None

        if self.mapper is not None:
            self.mapper.close()
            self.mapper = None

        gc.collect()

    def compose_images(self, image_array, compress=True) -> np.ndarray:
        """
        Compose multiple images into a grid for visualization.
        """
        rows, cols = self.composition_dims
        img_h, img_w = image_array[0].shape[:2]
        canvas_h = rows * img_h
        canvas_w = cols * img_w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # black background

        for idx, img in enumerate(image_array):
            r = idx // cols
            c = idx % cols
            start_y = r * img_h
            start_x = c * img_w
            canvas[start_y : start_y + img_h, start_x : start_x + img_w, :] = img

        if compress:
            canvas = cv2.resize(
                canvas,
                None,
                fx=COMPRESSION,
                fy=COMPRESSION,
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    def _build_candidate_evidence(
        self, masks: list[dict], image_array: list[np.ndarray]
    ) -> tuple[np.ndarray, list[str]]:
        """Render stable SoM labels on the exact SAM masks that created objects."""
        marked = [np.asarray(image).copy() for image in image_array]
        labels = [chr(65 + index) if index < 26 else str(index) for index in range(len(masks))]
        for label, mask_data in zip(labels, masks):
            image_index = int(mask_data.get("image_index", 0))
            if not 0 <= image_index < len(marked):
                continue
            mask = np.asarray(mask_data.get("mask"), dtype=bool)
            image = marked[image_index]
            if mask.shape != image.shape[:2] or not np.any(mask):
                continue
            overlay_color = np.array([255, 32, 32], dtype=np.float32)
            image[mask] = (
                0.55 * image[mask].astype(np.float32) + 0.45 * overlay_color
            ).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            center = (int(np.median(xs)), int(np.median(ys)))
            radius = 24
            cv2.circle(image, center, radius, (255, 255, 255), 4)
            cv2.circle(image, center, radius - 4, (255, 32, 32), 3)
            cv2.putText(
                image,
                label,
                (center[0] - 10, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
        return self.compose_images(marked), labels

    def merge_objects(self, distance_threshold: float = 0.5) -> None:
        """
        Merge detected objects that are close to each other with DBSCAN
        """

        # Extract centroids
        centroids = []
        for obj in self.detected_objects:
            if obj.centroid is not None:
                centroids.append(obj.centroid)

        if len(centroids) == 0:
            return
        centroids = np.array(centroids)
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=distance_threshold, min_samples=1, metric="euclidean"
        ).fit_predict(centroids)
        merged_objects = []
        for cluster_id in np.unique(clustering):
            cluster_indices = np.where(clustering == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            if len(cluster_indices) == 1:
                merged_objects.append(self.detected_objects[cluster_indices[0]])
                continue
            # Merge objects in this cluster
            merged_obj = self.detected_objects[
                cluster_indices[0]
            ]  # start with the first object's data

            merged_centroids = []
            merged_viewpoints = []
            observation_count = 0
            detection_scores = []
            first_seen_steps = []
            last_seen_steps = []
            evidence_observations = []
            for idx in cluster_indices:
                obj = self.detected_objects[idx]
                if obj.centroid is not None:
                    merged_centroids.append(obj.centroid)
                if obj.viewpoint is not None:
                    merged_viewpoints.append(obj.viewpoint)
                observation_count += int(getattr(obj, "observation_count", 1))
                if getattr(obj, "detection_score", None) is not None:
                    detection_scores.append(float(obj.detection_score))
                if getattr(obj, "first_seen_step", None) is not None:
                    first_seen_steps.append(int(obj.first_seen_step))
                if getattr(obj, "last_seen_step", None) is not None:
                    last_seen_steps.append(int(obj.last_seen_step))
                evidence_observations.extend(
                    list(getattr(obj, "evidence_observations", []))
                )

                # All objects must be valid to keep the merged one valid
                merged_obj.is_valid = merged_obj.is_valid and obj.is_valid

                # if any object is verified true positive, mark merged as true positive
                if obj.verification_status == "true_positive":
                    merged_obj.verification_status = "true_positive"
                elif obj.verification_status == "false_positive":
                    if merged_obj.verification_status != "true_positive":
                        merged_obj.verification_status = "false_positive"

                if self.goal_object is not None and obj.id == self.goal_object.id:
                    self.goal_object = merged_obj

            if len(merged_centroids) > 0:
                merged_obj.centroid = np.mean(merged_centroids, axis=0)
            if len(merged_viewpoints) > 0:
                merged_obj.viewpoint = np.mean(merged_viewpoints, axis=0)
            merged_obj.observation_count = observation_count
            if detection_scores:
                merged_obj.detection_score = max(detection_scores)
            if first_seen_steps:
                merged_obj.first_seen_step = min(first_seen_steps)
            if last_seen_steps:
                merged_obj.last_seen_step = max(last_seen_steps)
            merged_obj.evidence_observations = evidence_observations
            best_evidence = merged_obj.best_evidence()
            if best_evidence is not None:
                merged_obj.evidence_image = best_evidence["image"]
                merged_obj.evidence_label = best_evidence["label"]
                merged_obj.evidence_labels = list(best_evidence.get("labels", []))
                merged_obj.evidence_step = best_evidence.get("step")

            merged_objects.append(merged_obj)

        self.detected_objects = merged_objects

    # ---------- setup ----------
    def setup_system(self) -> None:
        """Init mapper, detector, and manager."""
        # Mapper
        intr2 = self.cam_intrinsic

        params = {
            "min_cell_width": self.VOX_SIZE / 2.0,
            "width": intr2.width,
            "height": intr2.height,
            "fx": intr2.intrinsic_matrix[0, 0],
            "fy": intr2.intrinsic_matrix[1, 1],
            "cx": intr2.intrinsic_matrix[0, 2],
            "cy": intr2.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": (
                self.config["depth_range"]
                if self.config["depth_range"] is not None
                else 3.5
            ),
            "resolution": self.VOX_SIZE,
            "observed_ray_stride": int(
                self.config.get("geometry_frontier_observed_ray_stride", 12)
            ),
        }

        # save intrinsics for later
        json_intr_path = os.path.join(self.save_dir, "camera_intrinsics.json")
        o3d.io.write_pinhole_camera_intrinsic(json_intr_path, intr2)
        self.log(
            "info", self.logging_file, f"Saved camera intrinsics to {json_intr_path}"
        )

        if self.use_map:
            self.mapper = WaveMapper(params=params)

        # OpenFrontier
        unet = load_model(
            path=self.args.unet_weight,
            num_classes=self.config["num_classes"],
            use_depth=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ft_detector = FrontierDetector(
            model=unet,
            camera_intrinsic=intr2.intrinsic_matrix.copy(),
            use_depth=True,
            img_size_model=self.config["input_img_size"],
            device=device,
            log_level=self.args.log_level,
        )

        # Frontier Manager
        self.ft_manager = FrontierManager(
            params=self.config, log_level=self.args.log_level, planner=self.planner
        )
        self.ft_manager.logging_file = self.logging_file
        self.ft_manager.filter_bbox = self.bbox
        self.ft_manager.planner.set_bounds(self.bbox)
        self.ft_manager.planner.logging_file = self.logging_file

    def log(self, level: str, file: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        folder = os.path.dirname(file).split("/")
        folder = f"{folder[-2]}/{folder[-1]}" if len(folder) >= 2 else folder[0]
        message = f"[{folder}]\t[{timestamp}]\t[AGENT]\t[{self.navigation_steps}]\t[{level.upper()}]\t{message}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "a") as f:
            print(message)
            f.write(message + "\n")
