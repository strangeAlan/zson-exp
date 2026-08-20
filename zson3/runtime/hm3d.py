"""Habitat 0.3.3 configuration adapter for the frozen HM3Dv1 protocol.

This module owns simulator and dataset compatibility only.  It deliberately
does not import or modify OpenFrontier's frontier, mapping, VLM, or navigation
decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .datasets import (
    NavigationProtocol,
    build_objectnav_config,
    dataset_spec,
    list_scenes,
    validate_dataset,
)


@dataclass(frozen=True)
class HM3DProtocol(NavigationProtocol):
    """Evaluation constants that must not drift silently between runs."""

    dataset_version: str = "v1"


def hm3d_paths(split: str = "val"):
    spec = dataset_spec("hm3dv1")
    return spec.paths(split)


def validate_hm3d_paths(split: str = "val"):
    return validate_dataset(dataset_spec("hm3dv1"), split)


def list_hm3d_scenes(split: str = "val") -> list[str]:
    return list_scenes("hm3dv1", split)


def build_hm3d_config(
    *,
    scene: str | None = None,
    episodes: int | None = None,
    seed: int = 0,
    protocol: HM3DProtocol | None = None,
    top_down_map: bool = True,
):
    """Build a Habitat 0.3.3 config for HM3Dv1 validation.

    ``scene`` filters the standard ``val`` split through ``content_scenes``;
    it is not treated as a synthetic dataset split as in the upstream script.
    """

    return build_objectnav_config(
        "hm3dv1",
        scene=scene,
        episodes=episodes,
        seed=seed,
        protocol=protocol,
        top_down_map=top_down_map,
    )
