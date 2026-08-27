import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from frontier.manager import FrontierManager
from nav.agent import NavigationAgent
from nav.target_closure import (
    RobustTargetObservation,
    SafeClosureState,
    local_refinement_candidates,
    robust_target_observation,
)
from vlm.utils import detect_bound_target_candidates


class _FakeNavmesh:
    @staticmethod
    def snap_point(point):
        point = np.asarray(point, dtype=float).copy()
        point[:2] = np.round(point[:2] / 0.1) * 0.1
        return point

    @staticmethod
    def geodesic_distance(start, goal):
        return float(np.linalg.norm(np.asarray(goal)[:2] - np.asarray(start)[:2]))


class TargetClosureSafeV1Tests(unittest.TestCase):
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
        observation = robust_target_observation(
            {"mask": mask, "image_index": 0, "label": "sofa"},
            depth,
            np.eye(4),
            intrinsic,
            np.zeros((20, 20, 3), dtype=np.uint8),
            "A",
            ["A"],
        )
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(float(observation.robust_position[2]), 2.0, places=4)
        self.assertGreater(float(observation.raw_centroid[2]), 2.0)

    def test_local_refinement_never_leaves_legacy_neighborhood(self):
        observation = RobustTargetObservation(
            mask=np.ones((2, 2), dtype=bool),
            image_index=0,
            viewpoint=np.eye(4),
            raw_centroid=np.array([2.0, 0.0, 1.0]),
            robust_position=np.array([2.0, 0.0, 1.0]),
            surface_points=np.array([[2.0, 0.0, 1.0]]),
            evidence_image=np.zeros((2, 2, 3), dtype=np.uint8),
            evidence_label="A",
            evidence_labels=["A"],
            detection_score=None,
        )
        current = np.eye(4)
        current[2, 3] = 1.0
        candidates = local_refinement_candidates(
            np.array([1.2, 0.0, 1.0]), observation, current, _FakeNavmesh()
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertLessEqual(candidate["correction_from_legacy"], 0.45 + 1e-9)
            direction = observation.robust_position - candidate["pose"][:3, 3]
            direction /= np.linalg.norm(direction)
            self.assertGreater(float(np.dot(candidate["pose"][:3, 2], direction)), 0.99)

    def test_manager_override_is_opt_in_after_lock(self):
        planner = mock.Mock()
        manager = FrontierManager(params={}, planner=planner)
        manager.object_lockin = SimpleNamespace(centroid=np.array([2.0, 0.0, 1.0]))
        current = np.eye(4)
        current[2, 3] = 1.0
        with mock.patch.object(
            manager, "get_object_free_point", return_value=np.array([1.0, 0.0, 1.0])
        ) as legacy:
            pose, _ = manager.get_goal_pose(current, use_graph=False)
            legacy.assert_called_once()
            self.assertTrue(np.allclose(pose[:3, 3], [1.0, 0.0, 1.0]))
        override = np.eye(4)
        override[:3, 3] = [1.2, 0.1, 1.0]
        manager.object_closure_pose = override
        with mock.patch.object(manager, "get_object_free_point") as legacy:
            pose, _ = manager.get_goal_pose(current, use_graph=False)
            legacy.assert_not_called()
            self.assertTrue(np.allclose(pose, override))

    def test_legacy_path_exhausted_stops_without_reobservation(self):
        agent = NavigationAgent.__new__(NavigationAgent)
        agent.safe_closure_state = SafeClosureState(
            object_id=1,
            accepted_step=10,
            raw_centroid=np.array([2.0, 0.0, 1.0]),
            legacy_endpoint_pose=np.eye(4),
        )
        agent.path_to_go = []
        agent.navigation_steps = 11
        agent.ft_manager = SimpleNamespace(
            object_lockin=SimpleNamespace(id=1, centroid=np.array([2.0, 0.0, 1.0]))
        )
        agent.target_diagnostics = {"termination_event": None}
        agent.success_threshold = 1.0
        agent.logging_file = "/tmp/unused-safe-v1-test.log"
        agent.log = mock.Mock()
        agent._reobserve_safe_target = mock.Mock()
        result = agent._advance_safe_target_closure(np.eye(4), np.ones((2, 2)), 2.0)
        self.assertEqual(result, "stop")
        agent._reobserve_safe_target.assert_not_called()
        self.assertEqual(
            agent.target_diagnostics["termination_event"]["stop_trigger"],
            "path_exhausted_legacy_compat",
        )

    def test_translation_arrival_finishes_before_pointnav_orientation(self):
        agent = NavigationAgent.__new__(NavigationAgent)
        endpoint = np.eye(4)
        state = SafeClosureState(
            object_id=1,
            accepted_step=10,
            raw_centroid=np.array([2.0, 0.0, 1.0]),
            legacy_endpoint_pose=endpoint,
            translation_arrival_cycles=1,
        )
        agent.safe_closure_state = state
        agent.path_to_go = [np.eye(4)]
        agent.navigation_steps = 12
        agent.target_closure_arrival_radius = 0.45
        agent.target_closure_arrival_cycles = 2
        agent.success_threshold = 1.0
        agent.ft_manager = SimpleNamespace(
            object_lockin=SimpleNamespace(id=1, centroid=np.array([2.0, 0.0, 1.0]))
        )
        agent.target_diagnostics = {"translation_arrival_events": []}
        agent._finish_translation_arrival = mock.Mock(return_value="stop")
        result = agent._advance_safe_target_closure(
            np.eye(4), np.ones((2, 2)), 2.0
        )
        self.assertEqual(result, "stop")
        agent._finish_translation_arrival.assert_called_once()
        self.assertEqual(len(agent.target_diagnostics["translation_arrival_events"]), 1)

    @mock.patch("vlm.utils.QwenClient.generate")
    def test_bound_prompt_is_candidate_specific_but_diagnostic(self, generate):
        generate.return_value = '[{"A": [0.1, "chair"]}]'
        success, scores, _ = detect_bound_target_candidates(
            np.zeros((32, 32, 3), dtype=np.uint8),
            ["A"],
            "sofa",
            "qwen3-vl-8b-local",
        )
        self.assertTrue(success)
        self.assertEqual(scores["A"][0], 0.1)
        prompt = generate.call_args.kwargs["prompt"]
        self.assertIn("specific marked mask", prompt)
        self.assertIn("unmarked target elsewhere", prompt)


if __name__ == "__main__":
    unittest.main()
