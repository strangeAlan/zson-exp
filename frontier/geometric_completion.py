"""Transient geometric proposal completion for OpenFrontier.

This module deliberately owns no persistent map or frontier manager. It turns
one accumulated Wavemap projection into ephemeral proposals at a normal
high-level frontier refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from frontier.frontier import Frontier
from utils.frontier_utils import ft_pos_direct_distance


@dataclass(frozen=True)
class CompletionStats:
    geometric_count: int
    unmatched_count: int
    best_visual_coverage: float
    best_geometry_coverage: float


class GeometricFrontierCompletion:
    """Generate and rank geometric completion proposals without changing OF semantics."""

    def __init__(self, params: dict):
        self.min_component_cells = int(
            params.get("geometry_frontier_min_component_cells", 4)
        )
        self.min_robot_distance = float(
            params.get("geometry_frontier_min_robot_distance", 0.75)
        )
        self.match_threshold = float(
            params.get("geometry_frontier_match_threshold", 1.8)
        )
        self.match_weights = tuple(
            params.get("geometry_frontier_match_weights", [1.0, 2.0])
        )
        self.coverage_radius = float(
            params.get(
                "geometry_frontier_coverage_radius",
                params.get("render_depth_range", 3.5),
            )
        )
        self.coverage_half_angle = np.deg2rad(
            float(params.get("geometry_frontier_coverage_half_angle_deg", 45.0))
        )
        self.blacklist_distance = float(
            params.get("geometry_frontier_blacklist_distance", 0.1)
        )

    @staticmethod
    def _grid_to_world(cells: np.ndarray, projection: dict) -> np.ndarray:
        origin = np.asarray(projection["origin"], dtype=float)
        resolution = float(projection["resolution"])
        return origin + (np.asarray(cells, dtype=float) + 0.5) * resolution

    def generate(
        self,
        projection: dict,
        current_position: Sequence[float],
        nav_level: float,
        planner,
        unreachable_positions: Iterable[Sequence[float]],
        suppressed_positions: Iterable[Sequence[float]] = (),
        suppression_distance: float = 0.5,
        bbox: Sequence[float] | None = None,
        occupancy_planner=None,
    ) -> list[Frontier]:
        """Generate one free-side representative for each valid boundary component."""
        known_free = np.asarray(projection["known_free"], dtype=bool)
        occupied = np.asarray(projection["occupied"], dtype=bool)
        unknown = np.asarray(projection["unknown"], dtype=bool)
        if not (known_free.shape == occupied.shape == unknown.shape):
            raise ValueError("Projection layers must share one 2D shape")

        structure = np.ones((3, 3), dtype=bool)
        unknown_neighbor = ndimage.binary_dilation(unknown, structure=structure)
        boundary = known_free & unknown_neighbor & ~occupied
        labels, count = ndimage.label(boundary, structure=structure)
        current = np.asarray(current_position, dtype=float)
        blocked = [np.asarray(item, dtype=float) for item in unreachable_positions]
        suppressed = [np.asarray(item, dtype=float) for item in suppressed_positions]
        candidates: list[Frontier] = []

        for label_id in range(1, count + 1):
            cells = np.argwhere(labels == label_id)
            if len(cells) < self.min_component_cells:
                continue
            world_cells = self._grid_to_world(cells, projection)
            centroid = world_cells.mean(axis=0)
            representative_index = int(
                np.argmin(np.linalg.norm(world_cells - centroid, axis=1))
            )
            representative = world_cells[representative_index]

            component_mask = labels == label_id
            component_adjacent_unknown = unknown & ndimage.binary_dilation(
                component_mask, structure=structure
            )

            # Direction is local to the chosen representative. A long connected
            # boundary can wrap around an observed region; averaging unknown
            # cells along the whole component can point sideways or backwards.
            representative_cell = cells[representative_index]
            representative_mask = np.zeros_like(boundary, dtype=bool)
            representative_mask[tuple(representative_cell)] = True
            adjacent_unknown = unknown & ndimage.binary_dilation(
                representative_mask, structure=structure
            )
            unknown_cells = np.argwhere(adjacent_unknown)
            if len(unknown_cells) == 0:
                continue
            unknown_centroid = self._grid_to_world(
                unknown_cells, projection
            ).mean(axis=0)
            direction_xy = unknown_centroid - representative
            direction_norm = float(np.linalg.norm(direction_xy))
            if direction_norm <= 1e-6:
                continue
            direction_xy /= direction_norm

            raw_position = np.array(
                [representative[0], representative[1], float(nav_level)], dtype=float
            )
            snapped = np.asarray(planner.snap_point(raw_position), dtype=float)
            if not np.all(np.isfinite(snapped)):
                continue
            if bbox is not None and not (
                bbox[0] <= snapped[0] <= bbox[1]
                and bbox[2] <= snapped[1] <= bbox[3]
            ):
                continue
            if np.linalg.norm(snapped[:2] - current[:2]) < self.min_robot_distance:
                continue
            if any(
                np.linalg.norm(snapped[:2] - position[:2]) < self.blacklist_distance
                for position in blocked
            ):
                continue
            if any(
                np.linalg.norm(snapped[:2] - position[:2]) < suppression_distance
                for position in suppressed
            ):
                continue
            occupancy = occupancy_planner or planner
            if occupancy.isoccupied(snapped):
                continue
            path_distance = planner.geodesic_distance(current, snapped)
            if not np.isfinite(path_distance):
                continue

            frontier = Frontier()
            frontier.source = "geometry"
            frontier.navigation_point = snapped.copy()
            frontier.pos3d = snapped
            # Preserve the actual free/unknown boundary for image grounding.
            # The snapped PointNav viewpoint may move away from the observed ray.
            frontier.evidence_anchor = raw_position.copy()
            frontier.view_direction = np.array(
                [direction_xy[0], direction_xy[1], 0.0], dtype=float
            )
            frontier.direct_angle = float(np.arctan2(direction_xy[1], direction_xy[0]))
            frontier.pixel_pos = np.array([-1.0, -1.0])
            frontier.gain = float(np.count_nonzero(component_adjacent_unknown)) * float(
                projection["resolution"]
            ) ** 2
            frontier.u_gain = frontier.gain
            frontier.justification = "Geometric Frontier Completion"
            frontier.label = "G"
            frontier._path_distance = path_distance
            candidates.append(frontier)

        return candidates

    def coverage(
        self,
        frontier: Frontier,
        projection: dict,
        current_position: Sequence[float],
        planner,
    ) -> float:
        """Unknown area in a common forward sector divided by navmesh distance."""
        unknown_cells = np.argwhere(np.asarray(projection["unknown"], dtype=bool))
        if len(unknown_cells) == 0:
            frontier.coverage = 0.0
            return 0.0
        unknown_xy = self._grid_to_world(unknown_cells, projection)
        candidate_xy = np.asarray(frontier.pos3d, dtype=float)[:2]
        offsets = unknown_xy - candidate_xy
        distances = np.linalg.norm(offsets, axis=1)
        direction = np.asarray(frontier.view_direction, dtype=float)[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            frontier.coverage = 0.0
            return 0.0
        direction /= norm
        cos_angle = np.sum(offsets * direction[None, :], axis=1) / np.maximum(
            distances, 1e-8
        )
        sector = (
            (distances > 0.0)
            & (distances <= self.coverage_radius)
            & (cos_angle >= np.cos(self.coverage_half_angle))
        )
        unknown_area = float(np.count_nonzero(sector)) * float(
            projection["resolution"]
        ) ** 2
        path_distance = getattr(frontier, "_path_distance", None)
        if path_distance is None:
            path_distance = planner.geodesic_distance(current_position, frontier.pos3d)
        value = (
            unknown_area / max(float(path_distance), 1e-3)
            if np.isfinite(path_distance)
            else 0.0
        )
        frontier.coverage = value
        return value

    def unmatched(
        self, geometric: Iterable[Frontier], visual: Iterable[Frontier]
    ) -> list[Frontier]:
        """Use FrontierManager's position+direction metric and scale for matching."""
        visual_features = [
            np.concatenate((np.asarray(ft.pos3d), np.asarray(ft.view_direction)))
            for ft in visual
        ]
        result = []
        for candidate in geometric:
            feature = np.concatenate(
                (np.asarray(candidate.pos3d), np.asarray(candidate.view_direction))
            )
            covered = any(
                ft_pos_direct_distance(
                    feature, visual_feature, weights=list(self.match_weights)
                )
                <= self.match_threshold
                for visual_feature in visual_features
            )
            if not covered:
                result.append(candidate)
        return result

    def select_completion(
        self,
        geometric: list[Frontier],
        visual: list[Frontier],
        projection: dict,
        current_position: Sequence[float],
        planner,
    ) -> tuple[Frontier | None, CompletionStats]:
        for frontier in visual:
            self.coverage(frontier, projection, current_position, planner)
        for frontier in geometric:
            self.coverage(frontier, projection, current_position, planner)
        unmatched = self.unmatched(geometric, visual)
        best_geometry = max(unmatched, key=lambda ft: ft.coverage, default=None)
        best_visual_coverage = max(
            (float(ft.coverage) for ft in visual), default=0.0
        )
        best_geometry_coverage = (
            float(best_geometry.coverage) if best_geometry is not None else 0.0
        )
        # Coverage only ranks unmatched geometric proposals. Whether geometry is
        # allowed to run is a semantic-first lifecycle decision owned by the
        # navigation agent; it is never compared against visual coverage here.
        return best_geometry, CompletionStats(
            geometric_count=len(geometric),
            unmatched_count=len(unmatched),
            best_visual_coverage=best_visual_coverage,
            best_geometry_coverage=best_geometry_coverage,
        )
