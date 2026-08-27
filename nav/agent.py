import os
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
from utils.frontier_utils import read_config_yaml

# Mapping
from mapping.wavemap import WaveMapper

# Frontier Manager
from frontier.manager import FrontierManager
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
        self.termination_semantics = []
        self.composition_depths = []
        self.composition_viewpoints = []
        self.composition_semantics = []
        self.detected_objects = []
        self.goal_object = None
        self.appoaching_object = False
        self.target_diagnostics = {
            "segmentation_events": [],
            "verification_events": [],
            "pursuit_events": [],
            "approach_events": [],
            "reobservation_events": [],
            "stop_gate_events": [],
            "visibility_events": [],
            "termination_event": None,
        }
        self.target_closure_enabled = bool(
            self.config.get("target_closure_enabled", False)
        )
        self.target_pursuit_threshold = float(
            self.config.get("target_pursuit_threshold", 0.35)
        )
        self.target_stop_threshold = float(
            self.config.get("target_stop_threshold", self.termination_threshold)
        )
        self.target_surface_stop_distance = float(
            self.config.get("target_surface_stop_distance", 1.0)
        )
        self.target_reassociation_distance = float(
            self.config.get("target_reassociation_distance", 1.25)
        )
        self.target_max_reobserve_cycles = int(
            self.config.get("target_max_reobserve_cycles", 2)
        )
        self.target_max_pursuit_steps = int(
            self.config.get("target_max_pursuit_steps", 120)
        )
        self.candidate_verification_cache = {}

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
        
        self.timings["frontier_detection_global"]["calls"] += 1
        
        self.navigation_steps += 1
        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)
        self.poses.append(W_T_C2.copy())

        self.planner.nav_level = W_T_C2[2, 3] - self.C_T_R[2, 3]
        self.ft_manager.planner.nav_level = W_T_C2[2, 3] - self.C_T_R[2, 3]

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
            self.termination_semantics.pop(0)

        self.composition_images.append(rgb)
        self.termination_images.append(rgb)
        self.termination_depths.append(depth)
        self.termination_viewpoints.append(W_T_C2.copy())
        semantic_mask = getattr(self, "current_target_semantic_mask", None)
        semantic_copy = None if semantic_mask is None else semantic_mask.copy()
        self.termination_semantics.append(semantic_copy)
        self.composition_depths.append(depth)
        self.composition_viewpoints.append(W_T_C2.copy())
        self.composition_semantics.append(semantic_copy)

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
                        target_semantic_mask=self.composition_semantics[image_index],
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
            self.composition_semantics = []
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
        

        if (
            self.emergency_rotation  # When doing lifesaving rotation, check on every step
            or self.ft_manager.object_lockin is None
            and (no_more_frontier or reach_next_update)
        ):
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

        if (
            self.ft_manager.valid_frontiers is None
            or len(self.ft_manager.valid_frontiers) == 0
        ):
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
            self.path_to_go = (
                self.ft_manager.plan_path_to_goal(W_T_C2, depth=depth, use_graph=False)
                or []
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
                while attempts < 10:
                    try:
                        detect_start = time.time()
                        success, response, raw_response = detect_target_object(
                            rgb=termination,
                            target_object=self.goal,
                            vlm_model=self.detection_source,
                            api_key=self.get_api_key_for_model(self.detection_source),
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
                    bound_probability = None
                    bound_reason = None
                    bound_label = None
                    if self.target_closure_enabled:
                        (
                            bound_probability,
                            bound_reason,
                            bound_label,
                        ) = self._score_bound_object(self.goal_object)
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
                            "candidate_bound_probability": bound_probability,
                            "candidate_bound_reason": bound_reason,
                            "candidate_bound_label": bound_label,
                            "candidate_gt_overlap": self.goal_object.gt_overlap,
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
                        if self.target_closure_enabled:
                            self._start_target_pursuit(
                                self.goal_object,
                                W_T_C2,
                                depth,
                                bound_probability,
                                bound_reason,
                            )
                        else:
                            self.ft_manager.lock_into_object(self.goal_object)
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

            if self.target_closure_enabled:
                closure = self._advance_target_closure(W_T_C2, depth)
                if closure == "stop":
                    return False, "object_found"
                if closure == "released":
                    reach_next_update = True
                    self.move_enough = True
            elif len(self.path_to_go) == 0 or dist < self.success_threshold:
                self.target_diagnostics["termination_event"] = {
                    "step": self.navigation_steps,
                    "object_id": self.ft_manager.object_lockin.id,
                    "object_centroid": np.asarray(object_pos, dtype=float).tolist(),
                    "agent_position": W_T_C2[:3, 3].astype(float).tolist(),
                    "distance_to_object": float(dist),
                    "path_exhausted": len(self.path_to_go) == 0,
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

    def _score_bound_evidence(self, evidence: dict) -> tuple[dict, str]:
        labels = list(evidence.get("labels", []))
        cache_key = (
            int(evidence.get("step") or -1),
            tuple(labels),
            self.goal,
        )
        cached = self.candidate_verification_cache.get(cache_key)
        if cached is not None:
            return cached
        started = time.time()
        success, scores, raw_response = detect_bound_target_candidates(
            rgb_image=evidence["image"],
            labels=labels,
            target_object=self.goal,
            vlm_model=self.detection_source,
            api_key=self.get_api_key_for_model(self.detection_source),
        )
        self.timings["vlm_call"]["total_time"] += time.time() - started
        self.timings["vlm_call"]["calls"] += 1
        self.log(
            "info",
            self.vlm_log_file,
            f"Candidate-bound target detection: {raw_response}",
        )
        parsed = scores if success and isinstance(scores, dict) else {}
        result = (parsed, raw_response)
        self.candidate_verification_cache[cache_key] = result
        return result

    def _score_bound_object(
        self, obj: DetectedObject
    ) -> tuple[float, str, str | None]:
        evidence = obj.best_evidence()
        if evidence is None:
            return 0.0, "Candidate has no bound SAM evidence.", None
        scores, _ = self._score_bound_evidence(evidence)
        label = evidence.get("label")
        value = scores.get(label, [0.0, "Candidate label missing from response."])
        try:
            probability = float(value[0])
            reason = str(value[1])
        except (TypeError, ValueError, IndexError):
            probability = 0.0
            reason = "Malformed candidate-bound score."
        return probability, reason, label

    @staticmethod
    def _approach_event(candidate: dict) -> dict:
        return {
            key: value
            for key, value in candidate.items()
            if key != "pose"
        }

    def _select_next_approach(
        self, obj: DetectedObject, current_pose: np.ndarray, depth: np.ndarray
    ) -> bool:
        candidates = self.ft_manager.generate_object_approach_viewpoints(
            obj, current_pose, separation=0.7
        )
        obj.approach_viewpoints = [
            self._approach_event(candidate) for candidate in candidates
        ]
        remaining = [
            candidate
            for candidate in candidates
            if candidate["key"] not in obj.attempted_approach_keys
        ]
        event = {
            "step": int(self.navigation_steps),
            "object_id": int(obj.id),
            "raw_centroid": (
                None
                if obj.raw_centroid is None
                else np.asarray(obj.raw_centroid, dtype=float).tolist()
            ),
            "robust_position": (
                None
                if obj.robust_position is None
                else np.asarray(obj.robust_position, dtype=float).tolist()
            ),
            "legacy_raw_centroid_endpoint": (
                None
                if obj.raw_centroid is None
                else np.asarray(
                    self.ft_manager.get_object_free_point(
                        current_pose, np.asarray(obj.raw_centroid, dtype=float)
                    ),
                    dtype=float,
                ).tolist()
            ),
            "generated": [self._approach_event(candidate) for candidate in candidates],
            "selected": None,
            "path_steps": 0,
        }
        if not remaining:
            event["outcome"] = "no_untried_reachable_viewpoint"
            self.target_diagnostics["approach_events"].append(event)
            return False

        selected = remaining[0]
        obj.attempted_approach_keys.add(selected["key"])
        obj.active_approach_pose = np.asarray(selected["pose"], dtype=float).copy()
        obj.pursuit_state = "approach"
        self.ft_manager.object_approach_pose = obj.active_approach_pose.copy()
        self.path_to_go = (
            self.ft_manager.plan_path_to_goal(
                current_pose, depth=depth, use_graph=False
            )
            or []
        )
        event["selected"] = self._approach_event(selected)
        event["path_steps"] = len(self.path_to_go)
        event["path_endpoint"] = (
            self.path_to_go[-1][:3, 3].astype(float).tolist()
            if self.path_to_go
            else None
        )
        event["outcome"] = "planned" if self.path_to_go else "already_at_or_no_path"
        self.target_diagnostics["approach_events"].append(event)
        return True

    def _start_target_pursuit(
        self,
        obj: DetectedObject,
        current_pose: np.ndarray,
        depth: np.ndarray,
        bound_probability: float | None,
        bound_reason: str | None,
    ) -> None:
        obj.bound_probability = float(bound_probability or 0.0)
        obj.bound_reason = bound_reason
        obj.last_bound_step = self.navigation_steps
        obj.pursuit_start_step = self.navigation_steps
        obj.pursuit_state = "approach"
        self.ft_manager.lock_into_object(obj)
        planned = self._select_next_approach(obj, current_pose, depth)
        if not planned:
            obj.pursuit_state = "reobserve"
        self.target_diagnostics["pursuit_events"].append(
            {
                "step": int(self.navigation_steps),
                "event": "lock",
                "object_id": int(obj.id),
                "global_gate": "accepted",
                "candidate_bound_probability": obj.bound_probability,
                "candidate_bound_reason": obj.bound_reason,
                "candidate_bound_status": (
                    "confirmed"
                    if obj.bound_probability >= self.target_stop_threshold
                    else "uncertain_pursue_for_better_view"
                ),
            }
        )

    def _facing_target(
        self, current_pose: np.ndarray, target_position: np.ndarray
    ) -> tuple[bool, float]:
        forward = np.asarray(current_pose[:3, 2], dtype=float)
        direction = np.asarray(target_position, dtype=float) - current_pose[:3, 3]
        forward[2] = 0.0
        direction[2] = 0.0
        if np.linalg.norm(forward) < 1e-6 or np.linalg.norm(direction) < 1e-6:
            return True, 0.0
        cosine = float(
            np.clip(
                np.dot(forward, direction)
                / (np.linalg.norm(forward) * np.linalg.norm(direction)),
                -1.0,
                1.0,
            )
        )
        angle = float(np.degrees(np.arccos(cosine)))
        return angle <= 60.0, angle

    def _reobserve_target(self, locked: DetectedObject) -> tuple[DetectedObject | None, dict]:
        event = {
            "step": int(self.navigation_steps),
            "object_id": int(locked.id),
            "trigger": "path_exhausted",
            "mask_count": 0,
            "candidates": [],
            "selected_label": None,
            "association_distance": None,
            "bound_probability": 0.0,
            "outcome": None,
        }
        if len(self.termination_images) != self.n_images:
            event["outcome"] = "insufficient_observation_buffer"
            return None, event
        composition = self.compose_images(self.termination_images)
        started = time.time()
        response = segment_target_object(
            rgb_composition=composition,
            n_images=self.n_images,
            target_object=self.goal,
            segmentation_model=self.segmentation_source,
            api_key=self.get_api_key_for_model(self.segmentation_source),
        )
        self.timings["sam"]["total_time"] += time.time() - started
        self.timings["sam"]["calls"] += 1
        if response is None or len(response) != 2 or not response[0]:
            event["outcome"] = "no_mask"
            return None, event

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
                target_semantic_mask=self.termination_semantics[image_index],
            )
            if candidate.robust_position is not None:
                candidates.append(candidate)
        event["mask_count"] = len(candidates)
        event["candidates"] = [candidate.to_dict() for candidate in candidates]
        if not candidates:
            event["outcome"] = "no_3d_candidate"
            return None, event

        evidence = candidates[0].best_evidence()
        scores, raw_response = self._score_bound_evidence(evidence)
        self.log(
            "info", self.vlm_log_file, f"Target reobservation binding: {raw_response}"
        )
        ranked = []
        origin = np.asarray(
            locked.robust_position if locked.robust_position is not None else locked.centroid,
            dtype=float,
        )
        for candidate in candidates:
            distance = float(np.linalg.norm(candidate.robust_position[:2] - origin[:2]))
            value = scores.get(candidate.evidence_label, [0.0, "Missing label."])
            try:
                probability = float(value[0])
                reason = str(value[1])
            except (TypeError, ValueError, IndexError):
                probability, reason = 0.0, "Malformed candidate-bound score."
            if distance <= self.target_reassociation_distance:
                ranked.append((probability - 0.15 * distance, probability, distance, reason, candidate))
        if not ranked:
            event["outcome"] = "no_associated_candidate"
            return None, event
        _, probability, distance, reason, selected = max(ranked, key=lambda item: item[0])
        selected.bound_probability = probability
        selected.bound_reason = reason
        selected.last_bound_step = self.navigation_steps
        event.update(
            {
                "selected_label": selected.evidence_label,
                "association_distance": distance,
                "bound_probability": probability,
                "bound_reason": reason,
                "selected_gt_overlap": selected.gt_overlap,
                "outcome": (
                    "confirmed"
                    if probability >= self.target_stop_threshold
                    else "uncertain_or_rejected"
                ),
            }
        )
        return selected, event

    def _release_target(self, reason: str) -> None:
        locked = self.ft_manager.object_lockin
        if locked is not None:
            locked.is_valid = False
            locked.verification_status = reason
            locked.pursuit_state = "released"
            if locked.frontier is not None:
                locked.frontier.set_invalid()
            self.target_diagnostics["pursuit_events"].append(
                {
                    "step": int(self.navigation_steps),
                    "event": "release",
                    "object_id": int(locked.id),
                    "reason": reason,
                    "reobserve_cycles": int(locked.reobserve_cycles),
                }
            )
        self.ft_manager.object_lockin = None
        self.ft_manager.object_approach_pose = None
        self.ft_manager.current_goal_pose = None
        self.ft_manager.current_goal_ft_id = None
        self.goal_object = None
        self.path_to_go = []
        self.ft_manager.filter_frontiers()

    def _advance_target_closure(
        self, current_pose: np.ndarray, depth: np.ndarray
    ) -> str:
        locked = self.ft_manager.object_lockin
        if locked is None:
            return "released"
        if (
            locked.pursuit_start_step is not None
            and self.navigation_steps - locked.pursuit_start_step
            > self.target_max_pursuit_steps
        ):
            self._release_target("pursuit_budget_exhausted")
            return "released"
        if self.path_to_go:
            return "approaching"

        locked.pursuit_state = "reobserve"
        selected, event = self._reobserve_target(locked)
        locked.reobserve_cycles += 1
        event["reobserve_cycle"] = int(locked.reobserve_cycles)
        self.target_diagnostics["reobservation_events"].append(event)

        if selected is not None and selected.bound_probability >= self.target_pursuit_threshold:
            locked.update_geometry(selected)
            locked.bound_probability = selected.bound_probability
            locked.bound_reason = selected.bound_reason
            locked.last_bound_step = self.navigation_steps
            surface_distance = locked.surface_distance(current_pose[:3, 3])
            facing, facing_angle = self._facing_target(
                current_pose, locked.robust_position
            )
            gate = {
                "step": int(self.navigation_steps),
                "object_id": int(locked.id),
                "active_hypothesis": True,
                "recent_bound_candidate": locked.last_bound_step == self.navigation_steps,
                "bound_probability": float(locked.bound_probability),
                "bound_threshold": self.target_stop_threshold,
                "approach_completed": True,
                "surface_distance": surface_distance,
                "surface_distance_threshold": self.target_surface_stop_distance,
                "facing_target": facing,
                "facing_angle_degrees": facing_angle,
            }
            gate["passed"] = bool(
                gate["recent_bound_candidate"]
                and locked.bound_probability >= self.target_stop_threshold
                and surface_distance < self.target_surface_stop_distance
                and facing
            )
            self.target_diagnostics["stop_gate_events"].append(gate)
            if gate["passed"]:
                locked.pursuit_state = "stop"
                self.target_diagnostics["termination_event"] = {
                    "step": int(self.navigation_steps),
                    "object_id": int(locked.id),
                    "object_centroid": np.asarray(locked.centroid, dtype=float).tolist(),
                    "raw_centroid": (
                        None
                        if locked.raw_centroid is None
                        else np.asarray(locked.raw_centroid, dtype=float).tolist()
                    ),
                    "robust_position": np.asarray(
                        locked.robust_position, dtype=float
                    ).tolist(),
                    "agent_position": current_pose[:3, 3].astype(float).tolist(),
                    "distance_to_object": float(
                        np.linalg.norm(locked.centroid - current_pose[:3, 3])
                    ),
                    "surface_distance": surface_distance,
                    "path_exhausted": True,
                    "stop_trigger": "candidate_bound_surface_closure",
                    "bound_probability": float(locked.bound_probability),
                    "reobserve_cycles": int(locked.reobserve_cycles),
                }
                self.log(
                    "info",
                    self.logging_file,
                    "Target closure verified; issuing ObjectNav STOP.",
                )
                return "stop"
            if self._select_next_approach(locked, current_pose, depth):
                return "replanned"

        elif selected is not None:
            self.target_diagnostics["stop_gate_events"].append(
                {
                    "step": int(self.navigation_steps),
                    "object_id": int(locked.id),
                    "active_hypothesis": True,
                    "recent_bound_candidate": True,
                    "bound_probability": float(selected.bound_probability),
                    "bound_threshold": self.target_stop_threshold,
                    "approach_completed": True,
                    "passed": False,
                    "failure": "candidate_bound_below_pursuit_threshold",
                }
            )

        if locked.reobserve_cycles < self.target_max_reobserve_cycles:
            if self._select_next_approach(locked, current_pose, depth):
                return "replanned"
            locked.pursuit_state = "reobserve"
            self.rotate()
            return "reobserve"

        self._release_target("reobservation_budget_exhausted")
        return "released"

    def get_target_diagnostics(self) -> dict:
        """Return episode-level target evidence without changing policy state."""

        return {
            **self.target_diagnostics,
            "final_object_tracks": [obj.to_dict() for obj in self.detected_objects],
        }

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
        self.termination_semantics.clear()
        self.composition_depths.clear()
        self.composition_viewpoints.clear()
        self.composition_semantics.clear()
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
        """Render stable labels on the exact SAM masks that created candidates."""
        marked = [np.asarray(image).copy() for image in image_array]
        labels = [
            chr(65 + index) if index < 26 else str(index)
            for index in range(len(masks))
        ]
        for label, mask_data in zip(labels, masks):
            image_index = int(mask_data.get("image_index", 0))
            if not 0 <= image_index < len(marked):
                continue
            mask = np.asarray(mask_data.get("mask"), dtype=bool)
            image = marked[image_index]
            if mask.shape != image.shape[:2] or not np.any(mask):
                continue
            image[mask] = (
                0.55 * image[mask].astype(np.float32)
                + 0.45 * np.array([255, 32, 32], dtype=np.float32)
            ).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            center = (int(np.median(xs)), int(np.median(ys)))
            cv2.circle(image, center, 24, (255, 255, 255), 4)
            cv2.circle(image, center, 20, (255, 32, 32), 3)
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
        Merge nearby proposal identities while retaining the latest bound surface.
        """
        valid_pairs = [
            (index, obj)
            for index, obj in enumerate(self.detected_objects)
            if obj.centroid is not None
        ]
        if not valid_pairs:
            return
        centroids = np.asarray([obj.centroid for _, obj in valid_pairs])
        clustering = DBSCAN(
            eps=distance_threshold, min_samples=1, metric="euclidean"
        ).fit_predict(centroids)
        merged_objects = []
        for cluster_id in np.unique(clustering):
            local_indices = np.where(clustering == cluster_id)[0]
            members = [valid_pairs[index][1] for index in local_indices]
            if not members:
                continue
            if len(members) == 1:
                merged_objects.append(members[0])
                continue
            # Never average instance geometry across time.  Select the newest,
            # strongest observation and retain older observations as evidence.
            merged_obj = max(
                members,
                key=lambda obj: (
                    int(obj.last_seen_step or -1),
                    float(obj.detection_score or 0.0),
                ),
            )
            merged_obj.observation_count = sum(
                int(getattr(obj, "observation_count", 1)) for obj in members
            )
            first_steps = [obj.first_seen_step for obj in members if obj.first_seen_step is not None]
            if first_steps:
                merged_obj.first_seen_step = min(first_steps)
            evidence = []
            for obj in members:
                evidence.extend(getattr(obj, "evidence_observations", []))
                if self.goal_object is not None and obj.id == self.goal_object.id:
                    self.goal_object = merged_obj
            merged_obj.evidence_observations = evidence
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
        self.ft_manager.navmesh_planner = getattr(self, "navmesh_planner", None)
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
