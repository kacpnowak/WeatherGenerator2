"""The long-rollout round-2 overlays change only the spin-up of the run they fork."""

from pathlib import Path

import pytest
from test_fesom_vars_r1_configs import BASE_FILES, CONFIG, _differing, _merged

from weathergen.datasets.multi_stream_data_sampler import (
    resolve_prefix_range,
    resolve_prefix_zero_fraction,
)

FORECAST = "training_config.forecast"
EXPECTED_DIFF = {
    "spin": {f"{FORECAST}.prefix_steps"},
    "mix": {f"{FORECAST}.prefix_steps", f"{FORECAST}.prefix_zero_fraction"},
}


@pytest.mark.parametrize("run", EXPECTED_DIFF)
def test_overlay_changes_only_the_spin_up(run):
    drop = CONFIG / "fesom_vars_r1_drop.yml"
    with_overlay = _merged(drop, CONFIG / f"fesom_vars_r2_{run}.yml")
    assert _differing(_merged(drop), with_overlay) == EXPECTED_DIFF[run]


@pytest.mark.parametrize(("run", "fraction"), [("spin", 0.0), ("mix", 0.25)])
def test_the_sampler_reads_the_overlay(run, fraction):
    flat = _merged(CONFIG / f"fesom_vars_r2_{run}.yml")
    forecast = {
        key.removeprefix(f"{FORECAST}."): value
        for key, value in flat.items()
        if key.startswith(f"{FORECAST}.")
    }
    assert resolve_prefix_range(forecast, "train") == (0, 365)
    assert resolve_prefix_zero_fraction(forecast, "train") == fraction


def test_base_files_exist():
    assert all(Path(f).exists() for f in BASE_FILES)
