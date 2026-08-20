from types import SimpleNamespace

import pytest

from zson3.runtime.metrics import success_spl_at_distance


def _env(*, stop_called: bool, shortest: float, travelled: float):
    spl = SimpleNamespace(
        _start_end_episode_distance=shortest,
        _agent_episode_distance=travelled,
    )
    return SimpleNamespace(
        task=SimpleNamespace(
            is_stop_called=stop_called,
            measurements=SimpleNamespace(measures={"spl": spl}),
        )
    )


def test_success_and_spl_at_one_meter_reuse_habitat_path_length():
    metrics = success_spl_at_distance(
        _env(stop_called=True, shortest=4.0, travelled=5.0),
        {"distance_to_goal": 0.5},
        success_distance=1.0,
    )
    assert metrics["success_at_1m"] == 1.0
    assert metrics["spl_at_1m"] == pytest.approx(0.8)


def test_one_meter_success_still_requires_stop():
    metrics = success_spl_at_distance(
        _env(stop_called=False, shortest=4.0, travelled=5.0),
        {"distance_to_goal": 0.5},
        success_distance=1.0,
    )
    assert metrics == {"success_at_1m": 0.0, "spl_at_1m": 0.0}
