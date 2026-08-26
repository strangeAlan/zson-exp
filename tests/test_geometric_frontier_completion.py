import unittest
from unittest import mock

import numpy as np

from frontier.frontier import Frontier
from frontier.geometric_completion import GeometricFrontierCompletion
from frontier.manager import FrontierManager
from mapping.wavemap import WaveMapper
from nav.agent import NavigationAgent


class FakePlanner:
    def snap_point(self, point):
        return np.asarray(point, dtype=float)

    def isoccupied(self, point):
        return False

    def geodesic_distance(self, start, goal):
        return float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))


class OffsetSnapPlanner(FakePlanner):
    def snap_point(self, point):
        snapped = np.asarray(point, dtype=float).copy()
        snapped[0] += 0.4
        return snapped


class ReachedTransientPlanner:
    def __init__(self):
        self.solution = []

    def update_start_goal(self, start, goal):
        self.goal = goal
        return True

    def solve(self, **_kwargs):
        return True

    def interpolate_path(self):
        self.solution = []


class GeometryCompletionTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "geometry_frontier_min_component_cells": 2,
            "geometry_frontier_min_robot_distance": 0.1,
            "geometry_frontier_match_threshold": 1.8,
            "geometry_frontier_match_weights": [1.0, 2.0],
            "geometry_frontier_coverage_radius": 3.5,
            "geometry_frontier_coverage_half_angle_deg": 45.0,
        }
        self.completion = GeometricFrontierCompletion(self.params)
        observed = np.zeros((20, 20), dtype=bool)
        observed[4:12, 4:10] = True
        known_free = observed.copy()
        self.projection = {
            "known_free": known_free,
            "occupied": np.zeros_like(observed),
            "unknown": ~observed,
            "observed": observed,
            "resolution": 0.25,
            "origin": np.array([0.0, 0.0]),
        }
        self.planner = FakePlanner()

    def test_ray_mask_marks_only_observed_rays(self):
        mapper = WaveMapper.__new__(WaveMapper)
        mapper.params = {
            "min_range": 0.05,
            "max_range": 3.5,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
        }
        mapper.res = 0.5
        mapper.grid_min = -2.0
        mapper.grid_size = 8
        mapper.observed_ray_stride = 1
        mapper.observed_mask_2d = np.zeros((8, 8), dtype=bool)
        mapper._update_observed_mask(np.array([[1.0]], dtype=np.float32), np.eye(4))
        self.assertGreater(np.count_nonzero(mapper.observed_mask_2d), 0)
        self.assertLess(np.count_nonzero(mapper.observed_mask_2d), 8 * 8)

    def test_generates_free_unknown_boundary_candidate(self):
        candidates = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
        )
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.source == "geometry" for candidate in candidates))
        self.assertTrue(
            all(
                self.projection["known_free"][
                    tuple(
                        np.floor(
                            (candidate.pos3d[:2] - self.projection["origin"])
                            / self.projection["resolution"]
                        ).astype(int)
                    )
                ]
                for candidate in candidates
            )
        )

    def test_grounding_anchor_is_not_displaced_by_navmesh_snap(self):
        candidates = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=OffsetSnapPlanner(),
            unreachable_positions=[],
        )
        self.assertTrue(candidates)
        self.assertAlmostEqual(
            candidates[0].navigation_point[0] - candidates[0].evidence_anchor[0],
            0.4,
        )

    def test_direction_points_to_unknown_next_to_representative(self):
        candidates = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
        )
        self.assertTrue(candidates)
        candidate = candidates[0]
        forward = candidate.evidence_anchor[:2] + 0.2 * candidate.view_direction[:2]
        forward_cell = np.floor(
            (forward - self.projection["origin"])
            / self.projection["resolution"]
        ).astype(int)
        self.assertTrue(self.projection["unknown"][tuple(forward_cell)])

    def test_matching_suppresses_covered_geometry(self):
        geometric = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
        )
        visual = Frontier()
        visual.pos3d = geometric[0].pos3d.copy()
        visual.view_direction = geometric[0].view_direction.copy()
        self.assertNotIn(geometric[0], self.completion.unmatched(geometric, [visual]))

    def test_coverage_only_selects_best_geometry(self):
        geometric = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
        )
        override, stats = self.completion.select_completion(
            geometric,
            [],
            self.projection,
            np.array([0.5, 0.5, 0.0]),
            self.planner,
        )
        self.assertIsNotNone(override)
        self.assertEqual(stats.best_visual_coverage, 0.0)

    def test_visual_coverage_never_vetoes_geometry_ranking(self):
        geometric = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
        )
        visual = Frontier()
        visual.source = "visual"
        visual.pos3d = geometric[0].pos3d + np.array([10.0, 0.0, 0.0])
        visual.view_direction = -geometric[0].view_direction

        def fake_coverage(frontier, *_args):
            frontier.coverage = 100.0 if frontier.source == "visual" else 1.0
            return frontier.coverage

        with mock.patch.object(self.completion, "coverage", side_effect=fake_coverage):
            selected, stats = self.completion.select_completion(
                geometric,
                [visual],
                self.projection,
                np.array([0.5, 0.5, 0.0]),
                self.planner,
            )
        self.assertIsNotNone(selected)
        self.assertEqual(stats.best_visual_coverage, 100.0)
        self.assertEqual(stats.best_geometry_coverage, 1.0)

    def test_cooldown_suppresses_same_snapped_position(self):
        candidates = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
        )
        self.assertTrue(candidates)
        suppressed = self.completion.generate(
            self.projection,
            current_position=np.array([0.5, 0.5, 0.0]),
            nav_level=0.0,
            planner=self.planner,
            unreachable_positions=[],
            suppressed_positions=[candidates[0].pos3d],
            suppression_distance=0.5,
        )
        self.assertTrue(
            all(
                np.linalg.norm(item.pos3d[:2] - candidates[0].pos3d[:2]) >= 0.5
                for item in suppressed
            )
        )

    def test_transient_arrival_ignores_camera_height(self):
        manager = FrontierManager.__new__(FrontierManager)
        manager.params = {
            "geometry_frontier_arrival_distance": 0.5,
            "geometry_frontier_arrival_angle_deg": 35.0,
        }
        manager.planner = ReachedTransientPlanner()
        manager.max_planning_time = 1.0
        manager.planning_algo = "pointnav"
        manager.current_goal_ft_id = None
        manager.current_goal_pose = None
        manager.last_transient_plan_status = "idle"
        manager.last_transient_pose_error = None
        manager.blacklist_position = mock.Mock()
        manager.log = mock.Mock()

        frontier = Frontier()
        frontier.pos3d = np.array([2.0, 3.0, 0.0])
        frontier.view_direction = np.array([1.0, 0.0, 0.0])
        goal_pose = manager.get_frontier_pose(frontier)
        current_pose = goal_pose.copy()
        current_pose[:2, 3] += np.array([0.2, 0.1])
        current_pose[2, 3] += 0.8

        path = manager.plan_path_to_frontier(current_pose, frontier)
        self.assertIsNone(path)
        self.assertEqual(manager.last_transient_plan_status, "reached")
        self.assertLess(
            manager.last_transient_pose_error["translation_xy"], 0.5
        )
        manager.blacklist_position.assert_not_called()

    def test_transient_utility_matches_of_equation(self):
        manager = FrontierManager.__new__(FrontierManager)
        manager.prob_sharpness = 2.0
        manager.seeking_weight = 1.0
        manager.utility_g_factor = 2.0
        frontier = Frontier()
        frontier.pos3d = np.array([2.0, 0.0, 0.0])
        frontier.gain = 4.0
        frontier.u_gain = 4.0
        frontier.probability = 0.5
        manager.update_transient_frontier_utilities(
            [frontier], np.array([0.0, 0.0, 0.0])
        )
        self.assertAlmostEqual(frontier.utility, 0.5)

    def test_fixed_geometry_gain_mapping_uses_visual_quantiles(self):
        manager = FrontierManager.__new__(FrontierManager)
        manager.transient_gain_source_knots = np.array([0.1, 0.2, 0.4])
        manager.transient_gain_target_knots = np.array([1.0, 4.0, 9.0])
        gain, u_gain = manager.normalize_transient_gain(0.2, 36.0, 18.0)
        self.assertAlmostEqual(gain, 4.0)
        self.assertAlmostEqual(u_gain, 2.0)

    def test_geometry_candidate_grounds_to_aligned_visible_keyframe(self):
        agent = NavigationAgent.__new__(NavigationAgent)
        agent.geometry_grounding_margin = 0.1
        agent.geometry_grounding_min_distance = 0.5
        agent.geometry_grounding_max_distance = 3.5
        agent.geometry_grounding_min_alignment = 0.5
        agent.geometry_grounding_depth_tolerance = 0.35

        candidate = Frontier()
        candidate.pos3d = np.array([2.0, 0.0, 1.0])
        candidate.evidence_anchor = candidate.pos3d.copy()
        candidate.view_direction = np.array([1.0, 0.0, 0.0])
        keyframe_pose = FrontierManager.get_frontier_pose(candidate)
        keyframe_pose[:3, 3] = np.array([0.0, 0.0, 1.0])
        agent.geometry_keyframes = [
            {
                "step": 7,
                "rgb": np.zeros((320, 320, 3), dtype=np.uint8),
                "depth": np.full((320, 320), 3.0, dtype=np.float32),
                "K": np.array(
                    [[100.0, 0.0, 160.0], [0.0, 100.0, 160.0], [0, 0, 1]]
                ),
                "W_T_C": keyframe_pose,
            }
        ]
        self.assertTrue(agent._ground_geometry_candidate(candidate))
        self.assertEqual(candidate.evidence_keyframe_step, 7)
        self.assertLess(abs(candidate.evidence_pixel[0] - 160), 2)
        self.assertLess(abs(candidate.evidence_pixel[1] - 160), 2)


if __name__ == "__main__":
    unittest.main()
