import os
from pathlib import Path

import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from zson3.runtime.hm3d import build_hm3d_config
from zson3.runtime.datasets import build_objectnav_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("OPENFRONTIER_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser()
HABITAT_LAB_ROOT = Path(
    os.environ.get("HABITAT_LAB_ROOT", PROJECT_ROOT / "habitat-lab")
).expanduser()

HM3D_CONFIG_PATH = os.environ.get(
    "HABITAT_HM3D_CONFIG",
    str(
        HABITAT_LAB_ROOT
        / "habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    ),
)

MP3D_CONFIG_PATH = os.environ.get(
    "HABITAT_MP3D_CONFIG",
    str(
        HABITAT_LAB_ROOT
        / "habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
    ),
)

DATA_PATH = str(DATA_ROOT) + os.sep


def hm3d_config(path: str = HM3D_CONFIG_PATH, stage: str = "val", episodes=200):
    # Upstream passes a scene id through ``stage``. Keep the public call shape
    # while loading the standard HM3Dv1 val split and filtering content scenes.
    scene = None if stage == "val" else stage
    count = episodes if episodes is not None and episodes > 0 else None
    return build_hm3d_config(scene=scene, episodes=count, top_down_map=True)


def hm3d_v2_config(path: str = HM3D_CONFIG_PATH, stage: str = "val", episodes=200):
    scene = None if stage == "val" else stage
    count = episodes if episodes is not None and episodes > 0 else None
    return build_objectnav_config(
        "hm3dv2", scene=scene, episodes=count, top_down_map=True
    )


def mp3d_config(path: str = MP3D_CONFIG_PATH, stage: str = "val", episodes=200):
    scene = None if stage == "val" else stage
    count = episodes if episodes is not None and episodes > 0 else None
    return build_objectnav_config(
        "mp3d", scene=scene, episodes=count, top_down_map=True
    )


def ovon_config(path: str = HM3D_CONFIG_PATH, stage: str = "val_unseen", episodes=200):
    import ovon.dataset
    import ovon.task.simulator
    import ovon.task.sensors
    import ovon.measurements.nav
    from ovon.config import (
        NavmeshSettings,
        OVONDistanceToGoalConfig,
        ClipObjectGoalSensorConfig,
        OVONObjectGoalIDMeasurementConfig,
    )

    habitat_config = hm3d_config(path, stage, episodes)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.data_path = (
            DATA_PATH + "ovon/val_unseen/content/{split}.json.gz"
        )
        habitat_config.habitat.dataset.type = "OVON-v1"
        habitat_config.habitat.simulator.type = "OVONSim-v0"
        habitat_config.habitat.simulator.navmesh_settings = NavmeshSettings()
        habitat_config.habitat.task.lab_sensors.update(
            {"clip_objectgoal_sensor": ClipObjectGoalSensorConfig()}
        )

        habitat_config.habitat.task.measurements.success.success_distance = 0.25
        habitat_config.habitat.task.measurements.distance_to_goal.type = (
            "OVONDistanceToGoal"
        )

        if "objectgoal_sensor" in habitat_config.habitat.task.lab_sensors:
            del habitat_config.habitat.task.lab_sensors["objectgoal_sensor"]

        habitat_config.habitat.task.measurements.update(
            {"ovon_object_goal_id": OVONObjectGoalIDMeasurementConfig()}
        )

        habitat_config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    return habitat_config
