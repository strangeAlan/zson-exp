"""Evaluation metrics that preserve multiple ObjectNav success radii."""

from __future__ import annotations

from typing import Any, Mapping


def success_spl_at_distance(
    habitat_env: Any,
    metrics: Mapping[str, Any],
    *,
    success_distance: float,
) -> dict[str, float]:
    """Compute Habitat Success/SPL at another radius from the same trajectory."""

    distance = float(metrics.get("distance_to_goal", float("inf")))
    stop_called = bool(
        getattr(getattr(habitat_env, "task", None), "is_stop_called", False)
    )
    success = float(stop_called and distance < success_distance)

    measures = getattr(
        getattr(getattr(habitat_env, "task", None), "measurements", None),
        "measures",
        {},
    )
    spl_measure = measures.get("spl")
    shortest = getattr(spl_measure, "_start_end_episode_distance", None)
    travelled = getattr(spl_measure, "_agent_episode_distance", None)
    if shortest is None or travelled is None:
        spl = 0.0
    else:
        shortest = float(shortest)
        travelled = float(travelled)
        spl = success * shortest / max(shortest, travelled)

    return {"success": success, "spl": float(spl)}
