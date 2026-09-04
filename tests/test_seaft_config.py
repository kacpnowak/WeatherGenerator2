# tests/test_seaft_config.py
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "config/config_glorys_v3_seaft.yml"
PHASE_C = REPO / "config/config_glorys_v3_phaseC.yml"

ARCH_KEYS = [
    "ae_local_dim_embed", "ae_local_num_blocks", "ae_local_max_tokens_per_cell",
    "ae_global_dim_embed", "ae_global_num_blocks", "ae_aggregation_num_blocks",
    "fe_num_blocks", "decoder_type", "healpix_level", "embed_orientation",
]

# representative named_modules() names; the regex is applied with re.fullmatch
MODULES_FROZEN = [
    "encoder.ae_local_engine", "encoder.ae_local_global_engine", "encoder.ae_local_global_engine.ae_adapter",
    "encoder.ae_global_engine", "encoder.q_cells", "forecast_engine", "latent_heads", "latent_pre_norm",
    "StreamEmbedder_GLORYS3", "StreamEmbedder_ATMO",
]
MODULES_TRAINABLE = [
    "encoder.ae_aggregation_engine", "embed_target_coords_GLORYS3", "TargetPredictionEngine_GLORYS3",
    "EnsPredictionHead_GLORYS3", "target_token_engines.GLORYS3.latent_in_norm",
    "embed_target_coords", "target_token_engines", "pred_heads",
]


def _cfg():
    return yaml.safe_load(CFG.read_text())


def test_freeze_regex_leaves_only_aggregation_and_decoder_trainable():
    rx = _cfg()["freeze_modules"]
    for name in MODULES_FROZEN:
        assert re.fullmatch(rx, name), f"expected frozen: {name}"
    for name in MODULES_TRAINABLE:
        assert re.fullmatch(rx, name) is None, f"expected trainable: {name}"


def test_run_mechanics_and_schedule():
    c = _cfg()
    assert c["streams_directory"] == "./config/streams/glorys_v3_seaft/"
    assert c["general"]["istep"] == 0
    t = c["training_config"]
    assert t["num_mini_epochs"] == 11
    assert t["samples_per_mini_epoch"] == 4096
    assert str(t["start_date"]) == "2020-01-01T00:00" and str(t["end_date"]) == "2023-12-31T00:00"
    assert t["forecast"]["num_steps"] == 10 and t["forecast"]["pushforward"] is False
    lf = t["losses"]["physical"]["loss_fcts"]
    assert lf["dynamic_loss"] is None
    assert [k for k, v in lf.items() if v is not None] == ["mse"]
    lr = t["learning_rate_scheduling"]
    assert float(lr["lr_max"]) == 2e-5 and lr["num_steps_warmup"] == 128 and lr["num_steps_cooldown"] == 512
    assert lr["policy_decay"] == "cosine"
    v = c["validation_config"]
    assert str(v["start_date"]) == "2019-01-01T00:00" and str(v["end_date"]) == "2019-12-31T00:00"
    assert v["validate_with_ema"]["enabled"] is True
    assert c["train_logging"]["checkpoint"] == 64


def test_architecture_block_identical_to_phase_c():
    c, p = _cfg(), yaml.safe_load(PHASE_C.read_text())
    for k in ARCH_KEYS:
        assert c[k] == p[k], k
