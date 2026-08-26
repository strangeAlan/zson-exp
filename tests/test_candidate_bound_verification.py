import unittest
from unittest import mock

import numpy as np
from types import SimpleNamespace

from nav.agent import NavigationAgent
from nav.detected_object import DetectedObject
from vlm.utils import detect_bound_target_candidates


class CandidateBoundVerificationTests(unittest.TestCase):
    def test_detected_object_preserves_bound_mask_and_image(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 3:6] = 1
        rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
        evidence = np.full((16, 16, 3), 64, dtype=np.uint8)
        obj = DetectedObject.from_mask(
            mask={
                "label": "chair",
                "mask": mask,
                "image_index": 0,
                "detection_score": 0.8,
            },
            depth=np.ones((8, 8), dtype=np.float32),
            viewpoint=np.eye(4),
            intrinsic_mat=np.array([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0, 0, 1]]),
            step=12,
            rgb=rgb,
            evidence_image=evidence,
            evidence_label="A",
            evidence_labels=["A", "B"],
        )
        self.assertTrue(np.array_equal(obj.mask, mask.astype(bool)))
        self.assertEqual(obj.best_evidence()["label"], "A")
        self.assertEqual(obj.best_evidence()["labels"], ["A", "B"])
        self.assertEqual(obj.to_dict()["evidence_observation_count"], 1)

    def test_candidate_som_labels_each_exact_mask(self):
        agent = NavigationAgent.__new__(NavigationAgent)
        agent.composition_dims = (1, 2)
        images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        first = np.zeros((64, 64), dtype=np.uint8)
        second = np.zeros((64, 64), dtype=np.uint8)
        first[20:40, 20:40] = 1
        second[15:35, 25:45] = 1
        marked, labels = agent._build_candidate_evidence(
            [
                {"mask": first, "image_index": 0},
                {"mask": second, "image_index": 1},
            ],
            images,
        )
        self.assertEqual(labels, ["A", "B"])
        self.assertGreater(np.count_nonzero(marked), 0)

    @mock.patch("vlm.utils.QwenClient.generate")
    def test_qwen_prompt_returns_per_candidate_probabilities(self, generate):
        generate.return_value = (
            '[{"A": [0.9, "marked sofa"], "B": [0.1, "marked chair"]}]'
        )
        success, scores, _ = detect_bound_target_candidates(
            rgb_image=np.zeros((32, 32, 3), dtype=np.uint8),
            labels=["A", "B"],
            target_object="sofa",
            vlm_model="qwen3-vl-8b-local",
        )
        self.assertTrue(success)
        self.assertEqual(scores["A"][0], 0.9)
        prompt = generate.call_args.kwargs["prompt"]
        self.assertIn("specific marked mask", prompt)
        self.assertIn("unmarked target elsewhere", prompt)

    def test_path_exhaustion_without_observation_releases_instead_of_stopping(self):
        locked = SimpleNamespace(
            id=7,
            centroid=np.array([2.0, 0.0, 0.0]),
            frontier=None,
            is_valid=True,
            verification_status="true_positive",
        )
        manager = SimpleNamespace(
            object_lockin=locked,
            current_goal_pose=np.eye(4),
            current_goal_ft_id=4,
            filter_frontiers=mock.Mock(),
        )
        agent = NavigationAgent.__new__(NavigationAgent)
        agent.ft_manager = manager
        agent.navigation_steps = 20
        agent.path_exhausted_recovery_attempts = 0
        agent.target_path_exhausted_max_retries = 1
        agent.termination_images = []
        agent.target_diagnostics = {"path_exhausted_recovery_events": []}
        agent.goal_object = None
        agent.path_to_go = []
        agent.log = mock.Mock()
        agent.logging_file = "unused"

        outcome = agent._recover_path_exhausted_object(
            W_T_C=np.eye(4), depth=np.ones((2, 2), dtype=np.float32)
        )
        self.assertEqual(outcome, "released")
        self.assertIsNone(manager.object_lockin)
        self.assertEqual(
            agent.target_diagnostics["path_exhausted_recovery_events"][0][
                "outcome"
            ],
            "released_no_observation",
        )


if __name__ == "__main__":
    unittest.main()
