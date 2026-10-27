"""Opt-in inference scaling of the per-step forcing injection (forecast.forcing_scale).

ForcingInjection is tested directly (all four modes); Model.forward is called unbound on a
stub ``self`` (as in test_land_cells.py) whose forcing module honours the `scale` argument.
"""

import math
import types

import pytest
import torch
from omegaconf import OmegaConf
from test_land_cells import (  # sibling test module
    LAND,
    LAND_TOK,
    MODEL_PARAMS,
    NUM_AUX,
    Z0,
    _batch,
    _fe,
    _rollout,
    _stub_model,
)

from weathergen.common import config
from weathergen.model.engines import LatentState
from weathergen.model.forcing import ForcingInjection
from weathergen.model.forcing_scale import ForcingScale
from weathergen.model.land_cells import LandCells
from weathergen.model.latent_filter import LatentFilter
from weathergen.model.model import Model, ModelOutput
from weathergen.train.trainer import Trainer


def _scaled_forcing(calls=None):
    """Additive forcing stub with the module's (tokens, cond, field, num_aux[, scale]) call."""

    def fn(tokens, condition, field, num_aux, *rest):
        if calls is not None:
            calls.append(len(rest))
        scale = rest[0] if rest else 1.0
        patch = tokens[:, num_aux:] + scale * field[None, :, None]
        return torch.cat([tokens[:, :num_aux], patch], dim=1), condition

    return fn


def _ref(num_steps, alpha, land_weight=1.0, freeze=False):
    """Reference recursion: z <- FE(z + alpha * w * f), land reset after each step if frozen."""
    w = torch.where(LAND, torch.tensor(land_weight, dtype=torch.float64), 1.0).double()
    batch = _batch(num_steps)
    z, out = Z0.clone(), []
    for n in range(num_steps):
        f = alpha * w * batch.forcing[n + 1]
        z = torch.cat([z[:, :NUM_AUX], z[:, NUM_AUX:] + f[None, :, None]], dim=1)
        z = _fe(z)
        if freeze:
            z[:, LAND_TOK] = Z0[:, LAND_TOK]
        out.append(z)
    return torch.stack(out)


# ── config ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, 1.0, 1])
def test_unset_or_one_disables_it(value):
    assert ForcingScale.from_config(value) is None


def test_config_is_parsed():
    cfg = OmegaConf.create({"forcing_scale": 0.5})
    fs = ForcingScale.from_config(cfg.get("forcing_scale"))
    assert fs.alpha == 0.5 and isinstance(fs.alpha, float)
    assert fs.as_dict() == {"alpha": 0.5}
    assert ForcingScale.from_config(0).alpha == 0.0
    assert ForcingScale.from_config(2).as_dict() == {"alpha": 2.0}


@pytest.mark.parametrize("value", [-0.5, math.nan, math.inf, "0.5", True, [0.5]])
def test_invalid_config_is_rejected(value):
    with pytest.raises(ValueError, match="forcing_scale"):
        ForcingScale.from_config(value)


# ── ForcingInjection ──────────────────────────────────────────────────────────────────


def _module(mode, dim_aux=0):
    torch.manual_seed(0)
    m = ForcingInjection(num_vars=2, dim_embed=8, mode=mode, dim_aux=dim_aux, num_heads=2)
    with torch.no_grad():
        m.gate.fill_(0.7)
    return m.eval()


def _inputs(dim_aux=0):
    torch.manual_seed(1)
    tokens = torch.randn(2, NUM_AUX + 6, 8)
    field = torch.cat([torch.randn(6, 2), torch.ones(6, 2)], dim=-1)
    condition = torch.randn(dim_aux) if dim_aux else torch.empty(0)
    return tokens, condition, field


@pytest.mark.parametrize(
    "mode, dim_aux", [("additive", 0), ("cross_attn", 0), ("global", 4), ("adaln_local", 4)]
)
def test_module_scale_one_is_bitwise_default(mode, dim_aux):
    m = _module(mode, dim_aux)
    tokens, condition, field = _inputs(dim_aux)
    t0, c0 = m(tokens, condition, field, NUM_AUX)
    t1, c1 = m(tokens, condition, field, NUM_AUX, 1.0)
    assert torch.equal(t0, t1) and torch.equal(c0, c1)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.25])
@pytest.mark.parametrize("mode", ["additive", "cross_attn"])
def test_module_scales_the_token_residual(mode, alpha):
    m = _module(mode)
    tokens, condition, field = _inputs()
    t1, _ = m(tokens, condition, field, NUM_AUX)
    ta, _ = m(tokens, condition, field, NUM_AUX, alpha)
    assert torch.equal(ta[:, :NUM_AUX], tokens[:, :NUM_AUX])
    torch.testing.assert_close(ta - tokens, alpha * (t1 - tokens), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.25])
def test_module_scales_the_global_conditioning(alpha):
    m = _module("global", 4)
    tokens, condition, field = _inputs(4)
    _, c1 = m(tokens, condition, field, NUM_AUX)
    ta, ca = m(tokens, condition, field, NUM_AUX, alpha)
    assert torch.equal(ta, tokens)
    torch.testing.assert_close(ca - condition, alpha * (c1 - condition), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.25])
def test_module_scales_the_local_conditioning(alpha):
    m = _module("adaln_local", 4)
    tokens, condition, field = _inputs(4)
    _, c1 = m(tokens, condition, field, NUM_AUX)
    _, ca = m(tokens, condition, field, NUM_AUX, alpha)
    base = condition.reshape(1, -1)
    torch.testing.assert_close(ca - base, alpha * (c1 - base), rtol=1e-5, atol=1e-6)


def test_module_scale_does_not_touch_the_gate_parameter():
    m = _module("additive")
    tokens, condition, field = _inputs()
    m(tokens, condition, field, NUM_AUX, 0.5)
    assert m.gate.item() == pytest.approx(0.7)


# ── Model.forward ─────────────────────────────────────────────────────────────────────


def test_default_call_is_unchanged():
    """Without the option the forcing module is called with its four original arguments."""
    calls = []
    model = _stub_model(forcing=_scaled_forcing(calls))
    plain, _ = _rollout(model, 4)
    again, _ = _rollout(model, 4, forcing_scale=None)
    assert torch.equal(plain, again)
    assert calls == [0] * 8
    torch.testing.assert_close(plain, _ref(4, 1.0), rtol=0, atol=0)


def test_trainer_config_one_is_the_default():
    """forcing_scale 1.0 from the config gives no model argument at all (exact default)."""
    model = _stub_model(forcing=_scaled_forcing())
    kw = {}
    fs = ForcingScale.from_config(1.0)
    if fs is not None:
        kw["forcing_scale"] = fs
    assert kw == {}
    plain, _ = _rollout(model, 4)
    again, _ = _rollout(model, 4, **kw)
    assert torch.equal(plain, again)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.75, 1.25])
def test_scaled_rollout_matches_the_reference(alpha):
    calls = []
    model = _stub_model(forcing=_scaled_forcing(calls))
    z, _ = _rollout(model, 6, forcing_scale=ForcingScale(alpha))
    assert calls == [1] * 6
    torch.testing.assert_close(z, _ref(6, alpha), rtol=0, atol=1e-15)
    plain, _ = _rollout(model, 6)
    assert not torch.allclose(z, plain)


@pytest.mark.parametrize("alpha", [0.5, 1.25])
def test_composes_with_no_forcing(alpha):
    model = _stub_model(forcing=_scaled_forcing())
    z, _ = _rollout(
        model, 6, forcing_scale=ForcingScale(alpha), land_cells=LandCells("no_forcing", LAND)
    )
    torch.testing.assert_close(z, _ref(6, alpha, land_weight=0.0), rtol=0, atol=1e-15)


@pytest.mark.parametrize("alpha", [0.5, 1.25])
def test_composes_with_freeze(alpha):
    model = _stub_model(forcing=_scaled_forcing())
    z, out = _rollout(
        model, 6, forcing_scale=ForcingScale(alpha), land_cells=LandCells("freeze", LAND)
    )
    for n in range(6):
        assert torch.equal(z[n][:, LAND_TOK], Z0[:, LAND_TOK])
    torch.testing.assert_close(z, _ref(6, alpha, freeze=True), rtol=0, atol=1e-15)
    # the freeze-only rollout is the alpha = 1 reference
    z1, _ = _rollout(model, 6, land_cells=LandCells("freeze", LAND))
    torch.testing.assert_close(z1, _ref(6, 1.0, freeze=True), rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [[2, 2, 2], [1, 5], [4, 2]])
@pytest.mark.parametrize("land", [None, "freeze", "no_forcing"])
@pytest.mark.parametrize("with_filter", [False, True])
def test_chunked_rollout_equals_one_chunk(chunks, land, with_filter):
    model = _stub_model(forcing=_scaled_forcing())
    kw = {"forcing_scale": ForcingScale(0.5)}
    if land is not None:
        kw["land_cells"] = LandCells(land, LAND)
    if with_filter:
        kw["latent_filter"] = LatentFilter("second_difference", 0.1)
    one, _ = _rollout(model, 6, **kw)
    chunked, _ = _rollout(model, 6, chunks=chunks, **kw)
    torch.testing.assert_close(chunked, one, rtol=0, atol=0)


def test_rejected_in_training_mode():
    model = _stub_model(forcing=_scaled_forcing())
    model.training = True
    with pytest.raises(ValueError, match="inference-time"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, forcing_scale=ForcingScale(0.5))


def test_needs_a_forcing_injection():
    model = _stub_model(forcing=_scaled_forcing())
    model.forcing_module = None
    with pytest.raises(ValueError, match="forcing injection"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, forcing_scale=ForcingScale(0.5))
    model = _stub_model(forcing=_scaled_forcing(), forcing_mode="none")
    with pytest.raises(ValueError, match="forcing injection"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, forcing_scale=ForcingScale(0.5))


# ── resumable rollout state ───────────────────────────────────────────────────────────


def _trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_path_results", lambda cf: tmp_path / cf.general.run_id)
    trainer = Trainer.__new__(Trainer)
    trainer.cf = types.SimpleNamespace(general=types.SimpleNamespace(run_id="abcd1234"))
    trainer.device = torch.device("cpu")
    return trainer


def test_rollout_state_without_forcing_scale_keeps_the_old_keys(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    trainer._save_rollout_state(LatentState(None, None, None, Z0), 30, 7)
    raw = torch.load(trainer._rollout_state_path(), weights_only=False)
    assert set(raw) == {"latent_state", "done_steps", "idx", "run_id"}
    Trainer._check_resume_setting(raw, "forcing_scale", None)
    Trainer._check_resume_setting(raw, "forcing_scale", ForcingScale.from_config(1.0))
    with pytest.raises(ValueError, match="forcing_scale mismatch"):
        Trainer._check_resume_setting(raw, "forcing_scale", ForcingScale(0.5))


def test_rollout_state_carries_the_setting(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    fs = ForcingScale(0.75)
    lc = LandCells("freeze", LAND)
    trainer._save_rollout_state(
        LatentState(None, None, None, Z0),
        30,
        7,
        land_cells=lc,
        land_cells_init=Z0[:, LAND_TOK],
        forcing_scale=fs,
    )
    state = trainer._load_rollout_state()
    assert state["forcing_scale"] == {"alpha": 0.75}
    Trainer._check_resume_setting(state, "forcing_scale", fs)
    Trainer._check_resume_setting(state, "forcing_scale", ForcingScale(0.75))
    Trainer._check_resume_setting(state, "land_cells", lc)
    for other in (None, ForcingScale(0.5), ForcingScale(1.25)):
        with pytest.raises(ValueError, match="forcing_scale mismatch"):
            Trainer._check_resume_setting(state, "forcing_scale", other)


def test_resumed_scaled_freeze_rollout_equals_one_chunk(monkeypatch, tmp_path):
    """Save after a chunk, reload, seed a ModelOutput as the trainer does, continue."""
    trainer = _trainer(monkeypatch, tmp_path)
    model = _stub_model(forcing=_scaled_forcing())
    kw = {"land_cells": LandCells("freeze", LAND), "forcing_scale": ForcingScale(0.5)}
    one, _ = _rollout(model, 6, **kw)
    batch = _batch(6)
    out = Model.forward(model, MODEL_PARAMS, batch, 3, **kw)
    trainer._save_rollout_state(
        out.get_last_latent_state(),
        3,
        0,
        land_cells=kw["land_cells"],
        land_cells_init=out.land_cells_init,
        forcing_scale=kw["forcing_scale"],
    )
    st = trainer._load_rollout_state()
    Trainer._check_resume_setting(st, "forcing_scale", kw["forcing_scale"])
    seed = ModelOutput(0, batch=batch, step_offset=st["done_steps"])
    seed.latent = [{"latent_state": st["latent_state"]}]
    seed.land_cells_init = st["land_cells_init"]
    cont = Model.forward(model, MODEL_PARAMS, seed, 3, **kw)
    z = torch.stack([cont.latent[i]["latent_state"].z_pre_norm for i in range(3)])
    torch.testing.assert_close(z, one[3:], rtol=0, atol=0)
