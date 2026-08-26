import unittest

import numpy as np

from frontier.frontier import Frontier
from frontier.geometric_completion import GeometricFrontierCompletion
from mapping.wavemap import WaveMapper


class FakePlanner:
    def snap_point(self, point):
        return np.asarray(point, dtype=float)

    def isoccupied(self, point):
        return False

    def geodesic_distance(self, start, goal):
        return float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))


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

    def test_override_compares_against_all_visual_coverage(self):
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


if __name__ == "__main__":
    unittest.main()
