import unittest
from unittest import mock

import numpy as np

from frontier.manager import FrontierManager
from nav.agent import NavigationAgent
from nav.detected_object import DetectedObject
from vlm.utils import detect_bound_target_candidates


class _FakeNavmeshPlanner:
    min_dist2occ = 0.2

    @staticmethod
    def snap_point(point):
        result = np.asarray(point, dtype=float).copy()
        result[:2] = np.round(result[:2] / 0.1) * 0.1
        return result

    @staticmethod
    def geodesic_distance(start, goal):
        return float(np.linalg.norm(np.asarray(goal)[:2] - np.asarray(start)[:2]))


class TargetClosureTests(unittest.TestCase):
    def test_robust_surface_rejects_mask_edge_background(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[3:17, 3:17] = 1
        depth = np.zeros((20, 20), dtype=np.float32)
        depth[3:17, 3:17] = 2.0
        depth[3, 3:17] = 8.0
        depth[16, 3:17] = 8.0
        intrinsic = np.array(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]
        )
        obj = DetectedObject.from_mask(
            {"mask": mask, "image_index": 0, "label": "sofa"},
            depth,
            np.eye(4),
            intrinsic,
            step=4,
        )
        self.assertGreater(len(obj.surface_points), 0)
        self.assertAlmostEqual(float(obj.robust_position[2]), 2.0, places=4)
        self.assertGreater(float(obj.raw_centroid[2]), 2.0)

    def test_gt_overlap_is_diagnostic_only_and_exact(self):
        mask = np.zeros((8, 8), dtype=bool)
        target = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        target[4:7, 4:7] = True
        overlap = DetectedObject.mask_overlap(mask, target)
        self.assertEqual(overlap["intersection_pixels"], 4)
        self.assertAlmostEqual(overlap["precision"], 4 / 16)
        self.assertAlmostEqual(overlap["recall"], 4 / 9)

    def test_approach_viewpoints_are_reachable_and_face_surface(self):
        manager = FrontierManager(params={}, planner=_FakeNavmeshPlanner())
        obj = DetectedObject()
        obj.robust_position = np.array([2.0, 0.0, 1.0])
        obj.centroid = obj.robust_position.copy()
        obj.viewpoint = np.eye(4)
        current = np.eye(4)
        current[2, 3] = 1.0
        candidates = manager.generate_object_approach_viewpoints(obj, current)
        self.assertGreaterEqual(len(candidates), 3)
        for candidate in candidates:
            pose = candidate["pose"]
            direction = obj.robust_position - pose[:3, 3]
            direction /= np.linalg.norm(direction)
            self.assertGreater(float(np.dot(pose[:3, 2], direction)), 0.99)

    @mock.patch("vlm.utils.QwenClient.generate")
    def test_bound_prompt_scores_marked_instance(self, generate):
        generate.return_value = '[{"A": [0.9, "sofa"], "B": [0.1, "chair"]}]'
        success, scores, _ = detect_bound_target_candidates(
            np.zeros((32, 32, 3), dtype=np.uint8),
            ["A", "B"],
            "sofa",
            "qwen3-vl-8b-local",
        )
        self.assertTrue(success)
        self.assertEqual(scores["A"][0], 0.9)
        prompt = generate.call_args.kwargs["prompt"]
        self.assertIn("specific marked mask", prompt)
        self.assertIn("unmarked target elsewhere", prompt)

    def test_facing_gate(self):
        agent = NavigationAgent.__new__(NavigationAgent)
        pose = np.eye(4)
        pose[:3, 2] = np.array([1.0, 0.0, 0.0])
        facing, angle = agent._facing_target(pose, np.array([2.0, 0.0, 0.0]))
        self.assertTrue(facing)
        self.assertEqual(angle, 0.0)
        facing, angle = agent._facing_target(pose, np.array([0.0, 2.0, 0.0]))
        self.assertFalse(facing)
        self.assertAlmostEqual(angle, 90.0)


if __name__ == "__main__":
    unittest.main()
