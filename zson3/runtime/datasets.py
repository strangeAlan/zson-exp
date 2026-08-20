"""Unified Habitat 0.3.3 ObjectNav dataset configuration.

Dataset layout differences belong here; OpenFrontier algorithm modules receive
the same observation/action contract for every supported benchmark.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    HabitatSimSemanticSensorConfig,
    TopDownMapMeasurementConfig,
)
from habitat.config.read_write import read_write


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NavigationProtocol:
    split: str = "val"
    max_episode_steps: int = 500
    forward_step_size: float = 0.25
    turn_angle: int = 30
    success_distance: float = 0.1
    rgb_height: int = 480
    rgb_width: int = 640
    depth_height: int = 480
    depth_width: int = 640
    horizontal_fov: int = 79
    min_depth: float = 0.5
    max_depth: float = 3.5
    normalize_depth: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_protocol(dataset: str) -> NavigationProtocol:
    """Return benchmark semantics without hiding dataset-specific differences."""

    spec = dataset_spec(dataset)
    if spec.family == "mp3d":
        # Preserve the public OpenFrontier MP3D evaluation contract.
        return NavigationProtocol(success_distance=1.0)
    return NavigationProtocol()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    version: str
    dataset_root: Path
    scenes_root: Path
    habitat_config_name: str
    scene_dataset: str | Path = "default"

    def paths(self, split: str) -> dict[str, Path | str]:
        return {
            "dataset_root": self.dataset_root,
            "dataset": self.dataset_root / split / f"{split}.json.gz",
            "content": self.dataset_root / split / "content",
            "scenes": self.scenes_root,
            "scene_dataset": self.scene_dataset,
            "habitat_config": _habitat_config_path(self.habitat_config_name),
        }


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def dataset_spec(name: str) -> DatasetSpec:
    key = name.lower().replace("-", "").replace("_", "")
    external = _path_from_env(
        "ZSON3_EXTERNAL_DATASETS_ROOT", PROJECT_ROOT / "data/external_datasets"
    )
    scenes = _path_from_env(
        "ZSON3_SCENE_DATASETS_ROOT", PROJECT_ROOT / "data/scene_datasets"
    )
    if key in {"hm3d", "hm3dv1"}:
        root = _path_from_env(
            "ZSON3_HM3DV1_ROOT",
            PROJECT_ROOT / "data/datasets/objectnav/hm3d/v1",
        )
        return DatasetSpec(
            name="hm3dv1",
            family="hm3d",
            version="v1",
            dataset_root=root,
            scenes_root=scenes,
            habitat_config_name="objectnav_hm3d.yaml",
            scene_dataset=scenes
            / "hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        )
    if key == "hm3dv2":
        root = _path_from_env(
            "ZSON3_HM3DV2_ROOT", external / "objectnav/hm3d/v2"
        )
        return DatasetSpec(
            name="hm3dv2",
            family="hm3d",
            version="v2",
            dataset_root=root,
            scenes_root=scenes,
            habitat_config_name="objectnav_hm3d.yaml",
            scene_dataset=scenes
            / "hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        )
    if key in {"mp3d", "mp3dv1"}:
        root = _path_from_env(
            "ZSON3_MP3D_ROOT", external / "objectnav/mp3d/v1"
        )
        mp3d_scenes = _path_from_env(
            "ZSON3_MP3D_SCENES_ROOT", external / "mp3d/v1/tasks"
        )
        return DatasetSpec(
            name="mp3d",
            family="mp3d",
            version="v1",
            dataset_root=root,
            scenes_root=mp3d_scenes,
            habitat_config_name="objectnav_mp3d.yaml",
        )
    raise ValueError(f"Unsupported ObjectNav dataset: {name!r}")


def _habitat_config_path(filename: str) -> Path:
    root = Path(habitat.__file__).resolve().parent
    return root / "config/benchmark/nav/objectnav" / filename


def validate_dataset(spec: DatasetSpec, split: str) -> dict[str, Path | str]:
    paths = spec.paths(split)
    required = ["dataset_root", "dataset", "content", "scenes", "habitat_config"]
    if paths["scene_dataset"] != "default":
        required.append("scene_dataset")
    missing = [f"{key}={paths[key]}" for key in required if not Path(paths[key]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{spec.name} runtime closure is incomplete:\n  " + "\n  ".join(missing)
        )
    return paths


def list_scenes(dataset: str, split: str = "val") -> list[str]:
    spec = dataset_spec(dataset)
    content = Path(validate_dataset(spec, split)["content"])
    return sorted(path.name.removesuffix(".json.gz") for path in content.glob("*.json.gz"))


def build_objectnav_config(
    dataset: str = "hm3dv1",
    *,
    scene: str | None = None,
    episodes: int | None = None,
    seed: int = 0,
    protocol: NavigationProtocol | None = None,
    top_down_map: bool = True,
):
    protocol = protocol or default_protocol(dataset)
    spec = dataset_spec(dataset)
    paths = validate_dataset(spec, protocol.split)
    config = habitat.get_config(str(paths["habitat_config"]))

    with read_write(config):
        config.habitat.seed = seed
        config.habitat.environment.max_episode_steps = protocol.max_episode_steps
        config.habitat.environment.iterator_options.shuffle = False
        if episodes is not None and episodes > 0:
            config.habitat.environment.iterator_options.num_episode_sample = episodes

        config.habitat.dataset.split = protocol.split
        config.habitat.dataset.data_path = str(
            spec.dataset_root / "{split}/{split}.json.gz"
        )
        config.habitat.dataset.scenes_dir = str(spec.scenes_root)
        config.habitat.dataset.content_scenes = [scene] if scene else ["*"]
        config.habitat.simulator.scene_dataset = str(spec.scene_dataset)
        config.habitat.simulator.forward_step_size = protocol.forward_step_size
        config.habitat.simulator.turn_angle = protocol.turn_angle

        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        for prefix in ("rgb", "depth"):
            sensor = sensors[f"{prefix}_sensor"]
            sensor.height = getattr(protocol, f"{prefix}_height")
            sensor.width = getattr(protocol, f"{prefix}_width")
            sensor.hfov = protocol.horizontal_fov
        sensors.depth_sensor.min_depth = protocol.min_depth
        sensors.depth_sensor.max_depth = protocol.max_depth
        sensors.depth_sensor.normalize_depth = protocol.normalize_depth
        # Evaluation-only target visibility instrumentation. OpenFrontier never
        # consumes this observation for decisions; it lets failure analysis
        # distinguish an unseen target from a detector/verifier miss.
        rgb_sensor = sensors.rgb_sensor
        sensors.update(
            {
                "semantic_sensor": HabitatSimSemanticSensorConfig(
                    height=protocol.rgb_height,
                    width=protocol.rgb_width,
                    hfov=protocol.horizontal_fov,
                    position=list(rgb_sensor.position),
                    orientation=list(rgb_sensor.orientation),
                )
            }
        )

        config.habitat.task.measurements.success.success_distance = protocol.success_distance
        config.habitat.task.measurements.update({"collisions": CollisionsMeasurementConfig()})
        if top_down_map:
            config.habitat.task.measurements.update(
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
                        fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                    )
                }
            )
    return config
