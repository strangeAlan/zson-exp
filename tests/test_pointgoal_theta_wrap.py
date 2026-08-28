"""Static parity tests for the OpenFrontier PointGoal adapter."""

from __future__ import annotations

import math

import numpy as np
import pytest
from habitat.tasks.nav.nav import PointGoalSensor

from planner.pointnav_planner import PointnavPlanner
from utils.transform import to_habitat_position, to_habitat_rotation


class _PolarPointGoal2D:
    _goal_format = "POLAR"
    _dimensionality = 2


def _camera_pose(position: np.ndarray, right_axis_yaw: float) -> np.ndarray:
    """Build a level CV camera pose: +X right, +Y down, +Z forward."""

    cosine, sine = np.cos(right_axis_yaw), np.sin(right_axis_yaw)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.array(
        [
            [cosine, 0.0, -sine],
            [sine, 0.0, cosine],
            [0.0, -1.0, 0.0],
        ]
    )
    pose[:3, 3] = position
    return pose


def _official_pointgoal(pose: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return PointGoalSensor._compute_pointgoal(
        _PolarPointGoal2D(),
        to_habitat_position(pose[:3, 3]),
        to_habitat_rotation(pose[:3, :3]),
        to_habitat_position(goal),
    )


def _circular_difference(left: float, right: float) -> float:
    return float(np.arctan2(np.sin(left - right), np.cos(left - right)))


@pytest.mark.parametrize(
    "yaw",
    [-math.pi, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, math.pi],
)
@pytest.mark.parametrize(
    "offset",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.2, 2.3),
        (-2.4, 0.7),
        (-0.8, -1.9),
        (2.1, -0.6),
    ],
)
def test_rho_theta_matches_habitat_official_sensor(yaw, offset):
    position = np.array([0.37, -1.21, 1.53], dtype=float)
    pose = _camera_pose(position, yaw)
    goal = position + np.array([offset[0], offset[1], 0.0], dtype=float)

    rho, theta = PointnavPlanner.rho_theta(None, pose, goal)
    official_rho, official_theta = _official_pointgoal(pose, goal)

    assert rho == pytest.approx(float(official_rho), abs=1e-6)
    assert _circular_difference(theta, float(official_theta)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert -math.pi <= theta <= math.pi


def test_left_right_and_pi_boundary_semantics():
    pose = _camera_pose(np.zeros(3), right_axis_yaw=0.0)

    # At yaw=0 the camera faces world +Y, so world -X is left and +X right.
    _, left = PointnavPlanner.rho_theta(None, pose, np.array([-1.0, 0.0, 0.0]))
    _, right = PointnavPlanner.rho_theta(None, pose, np.array([1.0, 0.0, 0.0]))
    assert left == pytest.approx(math.pi / 2.0)
    assert right == pytest.approx(-math.pi / 2.0)

    epsilon = 1e-7
    _, behind_left = PointnavPlanner.rho_theta(
        None, pose, np.array([-epsilon, -1.0, 0.0])
    )
    _, behind_right = PointnavPlanner.rho_theta(
        None, pose, np.array([epsilon, -1.0, 0.0])
    )
    for theta, x_offset in ((behind_left, -epsilon), (behind_right, epsilon)):
        assert -math.pi <= theta <= math.pi
        official = _official_pointgoal(
            pose, np.array([x_offset, -1.0, 0.0])
        )[1]
        assert _circular_difference(theta, float(official)) == pytest.approx(
            0.0, abs=1e-6
        )

    # The representation crosses from +pi to -pi, but is continuous on S1.
    assert abs(abs(behind_left) - math.pi) < 1e-6
    assert abs(abs(behind_right) - math.pi) < 1e-6
    assert abs(_circular_difference(behind_left, behind_right)) < 1e-6


def test_branch_cut_never_emits_unwrapped_angles():
    position = np.array([0.0, 0.0, 1.5])
    goal_angles = np.linspace(-math.pi, math.pi, 257)
    yaws = np.linspace(-math.pi, math.pi, 65)
    for yaw in yaws:
        pose = _camera_pose(position, yaw)
        for angle in goal_angles:
            goal = position + np.array([np.cos(angle), np.sin(angle), 0.0])
            _, theta = PointnavPlanner.rho_theta(None, pose, goal)
            assert -math.pi <= theta <= math.pi
