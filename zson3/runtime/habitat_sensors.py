"""Narrow adapter around Habitat simulator sensor/private-state access."""

from __future__ import annotations

import numpy as np

from utils.transform import from_habitat_position, from_habitat_rotation


def camera_extrinsic(env, sensor_uuid: str = "rgb") -> np.ndarray:
    """Return world-to-camera transform in OpenFrontier coordinates."""

    sensor_state = env.sim.get_agent_state().sensor_states[sensor_uuid]
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = from_habitat_rotation(sensor_state.rotation)
    world_from_camera[:3, 3] = from_habitat_position(sensor_state.position)
    return np.linalg.inv(world_from_camera)


def pinhole_intrinsic(config, sensor_uuid: str = "rgb_sensor") -> np.ndarray:
    """Return the 3x3 pinhole matrix without constructing visualization types."""

    sensor = config.habitat.simulator.agents.main_agent.sim_sensors[sensor_uuid]
    width, height, hfov = sensor.width, sensor.height, sensor.hfov
    focal = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    return np.array(
        [[focal, 0.0, (width - 1.0) / 2.0],
         [0.0, focal, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
