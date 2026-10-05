"""Tests for forecast.chunk_size: config validation and splitting a rollout into chunks."""

import types

import pytest
from omegaconf import OmegaConf

from weathergen.common.config import validate_forecast_policy_and_steps
from weathergen.train.trainer import Trainer


def _chunks(forecast: dict, output_idxs: list[int]) -> list[int]:
    batch = types.SimpleNamespace(get_output_idxs=lambda: output_idxs)
    return Trainer._get_forecast_chunks(None, OmegaConf.create(forecast), batch)


def test_unset_chunk_size_runs_all_steps_in_one_chunk():
    steps = [1, 2, 3, 4]
    assert _chunks({"num_steps": 4, "chunk_size": None}, steps) == [4]
    assert _chunks({"num_steps": 4}, steps) == [4]


def test_chunk_size_splits_steps_with_remainder():
    assert _chunks({"num_steps": 8, "chunk_size": 3}, list(range(1, 9))) == [3, 3, 2]


def test_chunk_size_equal_to_num_steps_is_one_chunk():
    assert _chunks({"num_steps": 4, "chunk_size": 4}, [1, 2, 3, 4]) == [4]


def test_masking_only_runs_one_model_pass():
    # offset 0, num_steps 0: the batch has the single output step 0, as on develop
    assert _chunks({"offset": 0, "num_steps": 0, "chunk_size": None}, [0]) == [1]
    assert _chunks({"offset": 0, "num_steps": 0, "chunk_size": 2}, [0]) == [1]


def test_list_num_steps_uses_the_steps_of_the_batch():
    # random policy: this batch drew 3 forecast steps out of [2, 4]
    forecast = {"num_steps": [2, 4], "policy": "random"}
    assert _chunks({**forecast, "chunk_size": None}, [1, 2, 3]) == [3]
    assert _chunks({**forecast, "chunk_size": 2}, [1, 2, 3]) == [2, 1]


@pytest.mark.parametrize(
    "forecast",
    [
        {"offset": 0, "num_steps": 0},
        {"offset": 1, "policy": "fixed", "num_steps": 4},
        {"offset": 1, "policy": "random", "num_steps": [2, 4]},
    ],
)
def test_validation_accepts_missing_chunk_size(forecast):
    forecast_cfg = OmegaConf.create(forecast)
    validate_forecast_policy_and_steps(forecast_cfg, "validation_config")
    assert forecast_cfg.chunk_size is None


def test_forecast_chunk_size_must_be_positive():
    # positive and unset (default config: null) chunk sizes are valid
    for chunk_size in (2, None):
        forecast_cfg = OmegaConf.create(
            {"num_steps": 4, "offset": 1, "policy": "fixed", "chunk_size": chunk_size}
        )
        validate_forecast_policy_and_steps(forecast_cfg, "training_config")

    forecast_cfg = OmegaConf.create(
        {"num_steps": 4, "offset": 1, "policy": "fixed", "chunk_size": 0}
    )
    with pytest.raises(AssertionError, match="chunk_size"):
        validate_forecast_policy_and_steps(forecast_cfg, "training_config")
