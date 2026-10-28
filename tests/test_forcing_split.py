"""Opt-in inference split of the additive forcing injection (forecast.forcing_split).

ForcingInjection is tested directly; Model.forward is called unbound on a stub ``self`` (as
in test_land_cells.py) whose forcing module honours the `scale` and `split` arguments.
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
from weathergen.model.forcing_split import ForcingSplit
from weathergen.model.land_cells import LandCells
from weathergen.model.latent_filter import LatentFilter
from weathergen.model.model import Model, ModelOutput
from weathergen.train.trainer import Trainer

NUM_CELLS, DIM = 4, 3
CONST = torch.tensor(
    [[0.2, -0.1, 0.05], [0.3, 0.0, -0.2], [-0.1, 0.4, 0.1], [0.05, 0.05, 0.3]],
    dtype=torch.float64,
)


def _split(a_c, a_a, const=CONST):
    return ForcingSplit(const_path="mem", const=const, alpha_const=a_c, alpha_anom=a_a)


def _split_forcing(calls=None):
    """Additive forcing stub with the module's (tokens, cond, field, num_aux[, scale[, split]])
    call; the injection of a cell is its field value on every feature."""

    def fn(tokens, condition, field, num_aux, *rest):
        if calls is not None:
            calls.append(len(rest))
        scale = rest[0] if rest else 1.0
        split = rest[1] if len(rest) > 1 else None
        inj = (scale * field)[:, None].expand(-1, tokens.shape[-1])
        if split is not None:
            inj = split.apply(inj)
        patch = tokens[:, num_aux:] + inj[None]
        return torch.cat([tokens[:, :num_aux], patch], dim=1), condition

    return fn


def _ref(num_steps, a_c, a_a, land_weight=1.0, freeze=False):
    """Reference: z <- FE(z + w * (a_c C + a_a (f - C))), land reset after each step if frozen."""
    w = torch.where(LAND, torch.tensor(land_weight, dtype=torch.float64), 1.0).double()
    batch = _batch(num_steps)
    z, out = Z0.clone(), []
    for n in range(num_steps):
        f = batch.forcing[n + 1][:, None].expand(-1, DIM)
        inj = w[:, None] * (a_c * CONST + a_a * (f - CONST))
        z = torch.cat([z[:, :NUM_AUX], z[:, NUM_AUX:] + inj[None]], dim=1)
        z = _fe(z)
        if freeze:
            z[:, LAND_TOK] = Z0[:, LAND_TOK]
        out.append(z)
    return torch.stack(out)


# ── config ────────────────────────────────────────────────────────────────────────────


def test_unset_disables_it():
    assert ForcingSplit.from_config(None) is None


@pytest.mark.parametrize("as_dict", [False, True])
def test_config_is_parsed(tmp_path, as_dict):
    path = tmp_path / "C.pt"
    torch.save({"const": CONST.float(), "tag": "x"} if as_dict else CONST.float(), path)
    cfg = OmegaConf.create(
        {"forcing_split": {"const_path": str(path), "alpha_const": 1, "alpha_anom": 0.5}}
    )
    fs = ForcingSplit.from_config(cfg.get("forcing_split"))
    assert fs.alpha_const == 1.0 and isinstance(fs.alpha_const, float) and fs.alpha_anom == 0.5
    assert fs.const.dtype == torch.float32 and torch.equal(fs.const, CONST.float())
    d = fs.as_dict()
    assert d["const_path"] == str(path) and d["const_shape"] == [NUM_CELLS, DIM]
    assert d["alpha_const"] == 1.0 and d["alpha_anom"] == 0.5
    assert isinstance(d["const_crc32"], int)
    # alphas default to 1
    fs1 = ForcingSplit.from_config({"const_path": str(path)})
    assert (fs1.alpha_const, fs1.alpha_anom) == (1.0, 1.0)


def test_crc_identifies_the_constant():
    assert (
        _split(1, 1).as_dict()["const_crc32"]
        == _split(1, 1, CONST.clone()).as_dict()["const_crc32"]
    )
    other = CONST.clone()
    other[0, 0] += 1e-3
    assert _split(1, 1).as_dict()["const_crc32"] != _split(1, 1, other).as_dict()["const_crc32"]


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"alpha_const": 1.0}, "const_path"),
        ({"const_path": "X", "alpha_anom": -0.5}, "alpha_anom"),
        ({"const_path": "X", "alpha_const": math.nan}, "alpha_const"),
        ({"const_path": "X", "alpha_const": math.inf}, "alpha_const"),
        ({"const_path": "X", "alpha_anom": "0.5"}, "alpha_anom"),
        ({"const_path": "X", "alpha_anom": True}, "alpha_anom"),
        ({"const_path": "X", "alpha": 0.5}, "unknown keys"),
        (0.5, "mapping"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, cfg, match):
    path = tmp_path / "X"
    torch.save(CONST, path)
    if isinstance(cfg, dict) and "const_path" in cfg:
        cfg = {**cfg, "const_path": str(path)}
    with pytest.raises(ValueError, match=match):
        ForcingSplit.from_config(cfg)


@pytest.mark.parametrize("bad", [torch.zeros(5), torch.zeros(2, 3, 4), {"other": CONST}])
def test_bad_constant_file_is_rejected(tmp_path, bad):
    torch.save(bad, tmp_path / "C.pt")
    with pytest.raises(ValueError, match="const_path"):
        ForcingSplit.from_config({"const_path": str(tmp_path / "C.pt")})


def test_non_finite_constant_is_rejected(tmp_path):
    c = CONST.clone()
    c[1, 1] = math.nan
    torch.save(c, tmp_path / "C.pt")
    with pytest.raises(ValueError, match="non-finite"):
        ForcingSplit.from_config({"const_path": str(tmp_path / "C.pt")})


def test_apply_and_shape_check():
    f = torch.randn(NUM_CELLS, DIM, dtype=torch.float64)
    torch.testing.assert_close(
        _split(0.8, 1.5).apply(f), 0.8 * CONST + 1.5 * (f - CONST), rtol=0, atol=1e-15
    )
    # (1, 1) equals the injection up to rounding
    torch.testing.assert_close(_split(1, 1).apply(f), f, rtol=0, atol=1e-15)
    # (0, 0) removes it
    assert torch.equal(_split(0, 0).apply(f), torch.zeros_like(f))
    with pytest.raises(ValueError, match="does not match"):
        _split(1, 1).apply(torch.zeros(NUM_CELLS + 1, DIM))
    # the constant follows the injection's dtype
    assert _split(1, 0.5).apply(f.float()).dtype == torch.float32


# ── ForcingInjection ──────────────────────────────────────────────────────────────────


def _module(mode, dim_aux=0, dim=8):
    torch.manual_seed(0)
    m = ForcingInjection(num_vars=2, dim_embed=dim, mode=mode, dim_aux=dim_aux, num_heads=2)
    with torch.no_grad():
        m.gate.fill_(-0.7)
    return m.eval()


def _inputs(dim_aux=0, cells=6, dim=8):
    torch.manual_seed(1)
    tokens = torch.randn(2, NUM_AUX + cells, dim)
    field = torch.cat([torch.randn(cells, 2), torch.ones(cells, 2)], dim=-1)
    condition = torch.randn(dim_aux) if dim_aux else torch.empty(0)
    return tokens, condition, field


@pytest.mark.parametrize("a_c, a_a", [(1.0, 0.5), (1.0, 1.5), (0.8, 1.0), (1.25, 1.0), (0, 0)])
def test_module_splits_the_additive_injection(a_c, a_a):
    m = _module("additive")
    tokens, condition, field = _inputs()
    torch.manual_seed(2)
    const = torch.randn(6, 8)
    t1, _ = m(tokens, condition, field, NUM_AUX)
    f = (t1 - tokens)[0, NUM_AUX:]
    split = ForcingSplit("mem", const, a_c, a_a)
    ts, cs = m(tokens, condition, field, NUM_AUX, 1.0, split)
    assert torch.equal(ts[:, :NUM_AUX], tokens[:, :NUM_AUX])
    assert torch.equal(cs, condition)
    expected = a_c * const + a_a * (f - const)
    torch.testing.assert_close(
        ts[:, NUM_AUX:] - tokens[:, NUM_AUX:], expected.expand(2, -1, -1), rtol=1e-5, atol=1e-6
    )
    assert m.gate.item() == pytest.approx(-0.7)


def test_module_split_none_is_bitwise_default():
    m = _module("additive")
    tokens, condition, field = _inputs()
    t0, _ = m(tokens, condition, field, NUM_AUX)
    t1, _ = m(tokens, condition, field, NUM_AUX, 1.0, None)
    assert torch.equal(t0, t1)


@pytest.mark.parametrize("mode, dim_aux", [("cross_attn", 0), ("global", 4), ("adaln_local", 4)])
def test_module_split_needs_additive(mode, dim_aux):
    m = _module(mode, dim_aux)
    tokens, condition, field = _inputs(dim_aux)
    with pytest.raises(ValueError, match="additive"):
        m(tokens, condition, field, NUM_AUX, 1.0, ForcingSplit("mem", torch.zeros(6, 8), 1, 1))


def test_module_rejects_a_constant_of_the_wrong_shape():
    m = _module("additive")
    tokens, condition, field = _inputs()
    with pytest.raises(ValueError, match="does not match"):
        m(tokens, condition, field, NUM_AUX, 1.0, ForcingSplit("mem", torch.zeros(5, 8), 1, 1))


# ── Model.forward ─────────────────────────────────────────────────────────────────────


def test_default_call_is_unchanged():
    """Without the option the forcing module is called with its four original arguments."""
    calls = []
    model = _stub_model(forcing=_split_forcing(calls))
    plain, _ = _rollout(model, 4)
    again, _ = _rollout(model, 4, forcing_split=None)
    assert torch.equal(plain, again)
    assert calls == [0] * 8
    torch.testing.assert_close(plain, _ref(4, 1.0, 1.0), rtol=0, atol=1e-15)


@pytest.mark.parametrize("a_c, a_a", [(1.0, 0.5), (1.0, 1.5), (0.8, 1.0), (1.25, 1.0)])
def test_split_rollout_matches_the_reference(a_c, a_a):
    calls = []
    model = _stub_model(forcing=_split_forcing(calls))
    z, _ = _rollout(model, 6, forcing_split=_split(a_c, a_a))
    assert calls == [2] * 6
    torch.testing.assert_close(z, _ref(6, a_c, a_a), rtol=0, atol=1e-14)
    plain, _ = _rollout(model, 6)
    assert not torch.allclose(z, plain)


def test_split_one_one_is_the_default_up_to_rounding():
    model = _stub_model(forcing=_split_forcing())
    z, _ = _rollout(model, 6, forcing_split=_split(1, 1))
    plain, _ = _rollout(model, 6)
    torch.testing.assert_close(z, plain, rtol=0, atol=1e-14)


@pytest.mark.parametrize("a_c, a_a", [(1.0, 0.5), (1.25, 1.0)])
def test_composes_with_no_forcing(a_c, a_a):
    model = _stub_model(forcing=_split_forcing())
    z, _ = _rollout(
        model, 6, forcing_split=_split(a_c, a_a), land_cells=LandCells("no_forcing", LAND)
    )
    torch.testing.assert_close(z, _ref(6, a_c, a_a, land_weight=0.0), rtol=0, atol=1e-14)


@pytest.mark.parametrize("a_c, a_a", [(1.0, 1.5), (0.8, 1.0)])
def test_composes_with_freeze(a_c, a_a):
    model = _stub_model(forcing=_split_forcing())
    z, _ = _rollout(model, 6, forcing_split=_split(a_c, a_a), land_cells=LandCells("freeze", LAND))
    for n in range(6):
        assert torch.equal(z[n][:, LAND_TOK], Z0[:, LAND_TOK])
    torch.testing.assert_close(z, _ref(6, a_c, a_a, freeze=True), rtol=0, atol=1e-14)


@pytest.mark.parametrize("chunks", [[2, 2, 2], [1, 5], [4, 2]])
@pytest.mark.parametrize("land", [None, "freeze", "no_forcing"])
@pytest.mark.parametrize("with_filter", [False, True])
def test_chunked_rollout_equals_one_chunk(chunks, land, with_filter):
    model = _stub_model(forcing=_split_forcing())
    kw = {"forcing_split": _split(0.8, 1.5)}
    if land is not None:
        kw["land_cells"] = LandCells(land, LAND)
    if with_filter:
        kw["latent_filter"] = LatentFilter("second_difference", 0.1)
    one, _ = _rollout(model, 6, **kw)
    chunked, _ = _rollout(model, 6, chunks=chunks, **kw)
    torch.testing.assert_close(chunked, one, rtol=0, atol=0)


def test_rejected_in_training_mode():
    model = _stub_model(forcing=_split_forcing())
    model.training = True
    with pytest.raises(ValueError, match="inference-time"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, forcing_split=_split(1, 0.5))


def test_rejected_together_with_forcing_scale():
    model = _stub_model(forcing=_split_forcing())
    with pytest.raises(ValueError, match="cannot be combined"):
        Model.forward(
            model,
            MODEL_PARAMS,
            _batch(2),
            2,
            forcing_split=_split(1, 0.5),
            forcing_scale=ForcingScale(0.5),
        )


@pytest.mark.parametrize("forcing_mode", ["cross_attn", "global", "none"])
def test_needs_the_additive_injection(forcing_mode):
    model = _stub_model(forcing=_split_forcing(), forcing_mode=forcing_mode)
    with pytest.raises(ValueError, match="additive"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, forcing_split=_split(1, 0.5))
    model = _stub_model(forcing=_split_forcing())
    model.forcing_module = None
    with pytest.raises(ValueError, match="additive"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, forcing_split=_split(1, 0.5))


# ── resumable rollout state ───────────────────────────────────────────────────────────


def _trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_path_results", lambda cf: tmp_path / cf.general.run_id)
    trainer = Trainer.__new__(Trainer)
    trainer.cf = types.SimpleNamespace(general=types.SimpleNamespace(run_id="abcd1234"))
    trainer.device = torch.device("cpu")
    return trainer


def test_rollout_state_without_forcing_split_keeps_the_old_keys(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    trainer._save_rollout_state(LatentState(None, None, None, Z0), 30, 7)
    raw = torch.load(trainer._rollout_state_path(), weights_only=False)
    assert set(raw) == {"latent_state", "done_steps", "idx", "run_id"}
    Trainer._check_resume_setting(raw, "forcing_split", None)
    with pytest.raises(ValueError, match="forcing_split mismatch"):
        Trainer._check_resume_setting(raw, "forcing_split", _split(1, 0.5))


def test_rollout_state_carries_the_setting(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    fs = _split(1.0, 1.5)
    lc = LandCells("freeze", LAND)
    trainer._save_rollout_state(
        LatentState(None, None, None, Z0),
        30,
        7,
        land_cells=lc,
        land_cells_init=Z0[:, LAND_TOK],
        forcing_split=fs,
    )
    state = trainer._load_rollout_state()
    assert state["forcing_split"] == fs.as_dict()
    assert "forcing_scale" not in state and "fe_noise" not in state
    Trainer._check_resume_setting(state, "forcing_split", _split(1.0, 1.5))
    Trainer._check_resume_setting(state, "land_cells", lc)
    other_const = CONST.clone()
    other_const[2, 1] = 0.0
    for other in (None, _split(1.0, 0.5), _split(1.25, 1.5), _split(1.0, 1.5, other_const)):
        with pytest.raises(ValueError, match="forcing_split mismatch"):
            Trainer._check_resume_setting(state, "forcing_split", other)


def test_resumed_split_freeze_rollout_equals_one_chunk(monkeypatch, tmp_path):
    """Save after a chunk, reload, seed a ModelOutput as the trainer does, continue."""
    trainer = _trainer(monkeypatch, tmp_path)
    model = _stub_model(forcing=_split_forcing())
    kw = {"land_cells": LandCells("freeze", LAND), "forcing_split": _split(1.0, 0.5)}
    one, _ = _rollout(model, 6, **kw)
    batch = _batch(6)
    out = Model.forward(model, MODEL_PARAMS, batch, 3, **kw)
    trainer._save_rollout_state(
        out.get_last_latent_state(),
        3,
        0,
        land_cells=kw["land_cells"],
        land_cells_init=out.land_cells_init,
        forcing_split=kw["forcing_split"],
    )
    st = trainer._load_rollout_state()
    Trainer._check_resume_setting(st, "forcing_split", kw["forcing_split"])
    seed = ModelOutput(0, batch=batch, step_offset=st["done_steps"])
    seed.latent = [{"latent_state": st["latent_state"]}]
    seed.land_cells_init = st["land_cells_init"]
    cont = Model.forward(model, MODEL_PARAMS, seed, 3, **kw)
    z = torch.stack([cont.latent[i]["latent_state"].z_pre_norm for i in range(3)])
    torch.testing.assert_close(z, one[3:], rtol=0, atol=0)
