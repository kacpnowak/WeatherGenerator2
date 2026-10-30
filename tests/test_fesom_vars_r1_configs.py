"""The long-rollout round-1 overlays: each run differs from its reference by one setting.

Merged on the stage-3 config of the fvars lineage and the base overlay (latent noise off);
the keys that differ with and without a run overlay must be exactly the ones it is meant to
change.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from weathergen.common import config

CONFIG = Path(__file__).resolve().parents[1] / "config"
BASE_FILES = [
    CONFIG / "config_fesom_vars.yml",
    CONFIG / "fesom_vars_stage3.yml",
    CONFIG / "fesom_vars_r1_base.yml",
]
SAVED_RUN = Path("/e/scratch/weatherai/shared_work/models/fvars002")
LOSS = "training_config.losses.physical.loss_fcts"

EXPECTED_DIFF = {
    "drop": {"fe_dropout_rate"},
    "spin": {"training_config.forecast.prefix_steps"},
    "ctrl": set(),
    "bias": {f"{LOSS}.mean_error_sq.weight"},
}


def _flatten(node, prefix: str = "") -> dict:
    """Dotted-key view of a nested dict; an empty dict is a leaf, lists are values."""
    if isinstance(node, dict) and node:
        out = {}
        for key, value in node.items():
            out |= _flatten(value, f"{prefix}.{key}" if prefix else str(key))
        return out
    return {prefix: node}


def _merged(*extra: Path) -> dict:
    cfg = config.load_merge_configs(None, None, None, None, *BASE_FILES, *extra)
    return _flatten(OmegaConf.to_container(cfg, resolve=True))


def _differing(flat_a: dict, flat_b: dict) -> set:
    return {
        k
        for k in flat_a.keys() | flat_b.keys()
        if flat_a.get(k, "<absent>") != flat_b.get(k, "<absent>")
    }


@pytest.mark.parametrize("run", EXPECTED_DIFF)
def test_overlay_changes_only_its_keys(run):
    with_overlay = _merged(CONFIG / f"fesom_vars_r1_{run}.yml")
    assert _differing(_merged(), with_overlay) == EXPECTED_DIFF[run]


def test_drop_overlay_switches_dropout_off():
    assert _merged()["fe_dropout_rate"] == 0.1
    assert _merged(CONFIG / "fesom_vars_r1_drop.yml")["fe_dropout_rate"] == 0.0


def test_spin_overlay_extends_the_spin_up_to_a_year():
    assert _merged()["training_config.forecast.prefix_steps"] == [0, 90]
    assert _merged(CONFIG / "fesom_vars_r1_spin.yml")["training_config.forecast.prefix_steps"] == [
        0,
        365,
    ]


def test_bias_overlay_adds_the_mean_error_next_to_mse_and_the_dynamic_loss():
    flat = _merged(CONFIG / "fesom_vars_r1_bias.yml")
    assert flat[f"{LOSS}.mse"] == {}
    assert flat[f"{LOSS}.dynamic_loss.window"] == 128 and flat[f"{LOSS}.dynamic_loss.L"] == 20.0
    assert flat[f"{LOSS}.mean_error_sq.weight"] > 0


def test_base_overlay_turns_the_noise_off():
    without = config.load_merge_configs(None, None, None, None, *BASE_FILES[:2])
    assert without.fe_impute_latent_noise_std == 1e-4
    assert _merged()["fe_impute_latent_noise_std"] == 0.0


@pytest.mark.skipif(not SAVED_RUN.is_dir(), reason="saved fvars002 run absent")
def test_saved_fvars002_config_gets_the_noise_off():
    # The path the real runs take: continue from the saved run, which holds 1e-4.
    saved = config.load_merge_configs(None, "fvars002", -1, None)
    assert saved.fe_impute_latent_noise_std == 1e-4
    merged = config.load_merge_configs(
        None,
        "fvars002",
        -1,
        None,
        str(CONFIG / "fesom_vars_stage3.yml"),
        str(CONFIG / "fesom_vars_r1_base.yml"),
        {},
    )
    assert merged.fe_impute_latent_noise_std == 0.0
