"""Opt-in latent noise / dropout at the forecast-engine input in inference (forecast.fe_noise).

FeNoise is tested directly; Model.forward is called unbound on a stub ``self`` (as in
test_land_cells.py). For dropout the stub forecast engine has ``fe_blocks`` holding a
Dropout module.
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
    _forcing,
    _rollout,
    _stub_model,
)

from weathergen.common import config
from weathergen.model.engines import LatentState
from weathergen.model.fe_noise import FeNoise
from weathergen.model.land_cells import LandCells
from weathergen.model.latent_filter import LatentFilter
from weathergen.model.model import Model, ModelOutput
from weathergen.train.trainer import Trainer


def _noise(n, shape=Z0.shape, dtype=Z0.dtype, fn=None):
    """The noise FeNoise draws for rollout step n."""
    gen = torch.Generator()
    gen.manual_seed(fn.step_seed(n, 0))
    return torch.randn(shape, generator=gen, dtype=dtype)


def _ref(num_steps, fn, freeze=False, dropout_fe=None):
    """Reference: z <- FE(z + f_n + noise_n), land reset after each step if frozen."""
    batch = _batch(num_steps)
    z, out = Z0.clone(), []
    for n in range(num_steps):
        z, _ = _forcing(z, None, batch.forcing[n + 1], NUM_AUX)
        if fn.active(n):
            eps = _noise(n, fn=fn)
            if fn.mode == "train_like":
                z = z + eps * torch.norm(z) * fn.std
            else:
                # the RMS is computed in float32, as for bf16 latents
                rms = z.float().pow(2).mean(-1, keepdim=True).sqrt().to(z.dtype)
                z = z + eps * rms * fn.std
        z = _fe(z)
        if freeze:
            z[:, LAND_TOK] = Z0[:, LAND_TOK]
        out.append(z)
    return torch.stack(out)


class _DropoutFE:
    """Forecast-engine stub with fe_blocks (one Dropout) before the stub map."""

    def __init__(self, p=0.5):
        self.fe_blocks = torch.nn.ModuleList([torch.nn.Dropout(p)]).eval()
        self.seen_training = []

    def __call__(self, tokens, condition, coords=None):
        self.seen_training.append(self.fe_blocks[0].training)
        for b in self.fe_blocks:
            tokens = b(tokens)
        return _fe(tokens)


# ── config ────────────────────────────────────────────────────────────────────────────


def test_unset_disables_it():
    assert FeNoise.from_config(None) is None


def test_config_is_parsed():
    cfg = OmegaConf.create(
        {"fe_noise": {"mode": "train_like", "std": 1.0e-4, "dropout": True, "seed": 3}}
    )
    fn = FeNoise.from_config(cfg.get("fe_noise"))
    assert fn == FeNoise("train_like", 1e-4, True, 3, "all")
    assert fn.as_dict() == {
        "mode": "train_like",
        "std": 1e-4,
        "dropout": True,
        "seed": 3,
        "steps": "all",
    }
    fn = FeNoise.from_config({"mode": "token_rms", "std": 1, "steps": "first"})
    assert fn.std == 1.0 and isinstance(fn.std, float) and fn.dropout is False and fn.seed == 0
    # dropout alone is allowed
    assert FeNoise.from_config({"mode": "train_like", "std": 0, "dropout": True}).std == 0.0


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"mode": "white", "std": 0.1}, "mode"),
        ({"std": 0.1}, "mode and std"),
        ({"mode": "token_rms"}, "mode and std"),
        ({"mode": "token_rms", "std": -0.1}, "std"),
        ({"mode": "token_rms", "std": math.nan}, "std"),
        ({"mode": "token_rms", "std": "0.1"}, "std"),
        ({"mode": "token_rms", "std": True}, "std"),
        ({"mode": "token_rms", "std": 0.1, "dropout": 1}, "dropout"),
        ({"mode": "token_rms", "std": 0.1, "seed": 1.5}, "seed"),
        ({"mode": "token_rms", "std": 0.1, "seed": -1}, "seed"),
        ({"mode": "token_rms", "std": 0.1, "steps": "last"}, "steps"),
        ({"mode": "token_rms", "std": 0.1, "scale": 2}, "unknown keys"),
        ({"mode": "token_rms", "std": 0.0}, "does nothing"),
        (0.1, "mapping"),
    ],
)
def test_invalid_config_is_rejected(cfg, match):
    with pytest.raises(ValueError, match=match):
        FeNoise.from_config(cfg)


# ── FeNoise ───────────────────────────────────────────────────────────────────────────


def test_train_like_is_the_training_expression():
    fn = FeNoise("train_like", 1e-4, seed=5)
    z = torch.randn(1, 50, 16)
    out = fn.perturb(z, 7)
    torch.testing.assert_close(
        out, z + _noise(7, z.shape, z.dtype, fn) * torch.norm(z) * 1e-4, rtol=0, atol=0
    )


def test_token_rms_scales_with_each_token():
    fn = FeNoise("token_rms", 0.1, seed=1)
    torch.manual_seed(0)
    z = torch.randn(1, 400, 2048) * torch.logspace(-1, 1, 400)[None, :, None]
    z[0, 0] = 0.0  # a zero token stays zero
    d = fn.perturb(z, 0) - z
    rms = z.pow(2).mean(-1).sqrt()
    ratio = d.pow(2).mean(-1).sqrt()[0, 1:] / rms[0, 1:]
    assert torch.all((ratio > 0.09) & (ratio < 0.11))
    assert torch.equal(d[0, 0], torch.zeros(2048))


def test_noise_is_seeded_per_step():
    fn = FeNoise("token_rms", 0.1, seed=1)
    z = torch.ones(1, 5, 8)
    a, b = fn.perturb(z, 3), fn.perturb(z, 3)
    assert torch.equal(a, b)
    assert not torch.equal(fn.perturb(z, 3), fn.perturb(z, 4))
    assert not torch.equal(a, FeNoise("token_rms", 0.1, seed=2).perturb(z, 3))
    # the global RNG is not consumed
    state = torch.get_rng_state()
    fn.perturb(z, 0)
    assert torch.equal(state, torch.get_rng_state())


def test_steps_first_only_perturbs_step_zero():
    fn = FeNoise("token_rms", 0.1, steps="first")
    z = torch.ones(1, 5, 8)
    assert not torch.equal(fn.perturb(z, 0), z)
    for n in (1, 2, 30):
        assert torch.equal(fn.perturb(z, n), z)


def test_dropout_context_is_seeded_and_restores_state():
    fe = _DropoutFE()
    fn = FeNoise("train_like", 0.0, dropout=True, seed=4)
    z = torch.ones(1, 20, 8, dtype=torch.float64)
    state = torch.get_rng_state()
    with fn.dropout_context(fe, 2, z.device):
        assert fe.fe_blocks.training and fe.fe_blocks[0].training
        a = fe.fe_blocks[0](z)
    assert not fe.fe_blocks.training and not fe.fe_blocks[0].training
    assert torch.equal(state, torch.get_rng_state())
    assert (a == 0).any() and (a == 2).any()  # p = 0.5: zeros and 1 / (1 - p)
    with fn.dropout_context(fe, 2, z.device):
        b = fe.fe_blocks[0](z)
    assert torch.equal(a, b)
    with fn.dropout_context(fe, 3, z.device):
        c = fe.fe_blocks[0](z)
    assert not torch.equal(a, c)
    # no dropout: a no-op context
    with FeNoise("train_like", 1e-4).dropout_context(fe, 2, z.device):
        assert not fe.fe_blocks.training


def test_dropout_needs_fe_blocks():
    fn = FeNoise("train_like", 0.0, dropout=True)
    with pytest.raises(ValueError, match="fe_blocks"):
        with fn.dropout_context(object(), 0, torch.device("cpu")):
            pass


# ── Model.forward ─────────────────────────────────────────────────────────────────────


def test_default_call_is_unchanged():
    model = _stub_model()
    plain, _ = _rollout(model, 4)
    again, _ = _rollout(model, 4, fe_noise=None)
    assert torch.equal(plain, again)
    torch.testing.assert_close(
        plain, _ref(4, FeNoise("token_rms", 0.0, dropout=True)), rtol=0, atol=0
    )


@pytest.mark.parametrize("mode, std", [("train_like", 1e-4), ("token_rms", 0.03)])
@pytest.mark.parametrize("steps", ["all", "first"])
def test_noisy_rollout_matches_the_reference(mode, std, steps):
    fn = FeNoise(mode, std, seed=11, steps=steps)
    model = _stub_model()
    z, _ = _rollout(model, 6, fe_noise=fn)
    torch.testing.assert_close(z, _ref(6, fn), rtol=0, atol=0)
    plain, _ = _rollout(model, 6)
    assert not torch.allclose(z, plain)
    # reproducible
    again, _ = _rollout(model, 6, fe_noise=fn)
    assert torch.equal(z, again)


def test_composes_with_freeze():
    fn = FeNoise("token_rms", 0.1, seed=2)
    model = _stub_model()
    z, _ = _rollout(model, 6, fe_noise=fn, land_cells=LandCells("freeze", LAND))
    for n in range(6):
        assert torch.equal(z[n][:, LAND_TOK], Z0[:, LAND_TOK])
    torch.testing.assert_close(z, _ref(6, fn, freeze=True), rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [[2, 2, 2], [1, 5], [4, 2]])
@pytest.mark.parametrize("land", [None, "freeze", "no_forcing"])
@pytest.mark.parametrize("dropout", [False, True])
@pytest.mark.parametrize("with_filter", [False, True])
def test_chunked_rollout_equals_one_chunk(chunks, land, dropout, with_filter):
    model = _stub_model()
    if dropout:
        model.forecast_engine = _DropoutFE(0.3)
    kw = {"fe_noise": FeNoise("train_like", 1e-2, dropout=dropout, seed=9)}
    if land is not None:
        kw["land_cells"] = LandCells(land, LAND)
    if with_filter:
        kw["latent_filter"] = LatentFilter("second_difference", 0.1)
    one, _ = _rollout(model, 6, **kw)
    chunked, _ = _rollout(model, 6, chunks=chunks, **kw)
    torch.testing.assert_close(chunked, one, rtol=0, atol=0)


def test_first_step_only_when_chunked():
    """steps: first perturbs the rollout's first step, not the first step of later chunks."""
    model = _stub_model()
    fn = FeNoise("token_rms", 0.1, seed=3, steps="first")
    one, _ = _rollout(model, 6, fe_noise=fn)
    chunked, _ = _rollout(model, 6, chunks=[1, 1, 4], fe_noise=fn)
    torch.testing.assert_close(chunked, one, rtol=0, atol=0)
    torch.testing.assert_close(one, _ref(6, fn), rtol=0, atol=0)


def test_dropout_rollout_runs_the_blocks_in_train_mode():
    model = _stub_model()
    fe = _DropoutFE(0.3)
    model.forecast_engine = fe
    _rollout(model, 4, fe_noise=FeNoise("train_like", 1e-4, dropout=True, steps="first"))
    assert fe.seen_training == [True, False, False, False]
    assert not fe.fe_blocks[0].training
    fe.seen_training.clear()
    _rollout(model, 3, fe_noise=FeNoise("train_like", 1e-4, dropout=False))
    assert fe.seen_training == [False] * 3


def test_rejected_in_training_mode():
    model = _stub_model()
    model.training = True
    with pytest.raises(ValueError, match="inference-time"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, fe_noise=FeNoise("token_rms", 0.1))


# ── resumable rollout state ───────────────────────────────────────────────────────────


def _trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_path_results", lambda cf: tmp_path / cf.general.run_id)
    trainer = Trainer.__new__(Trainer)
    trainer.cf = types.SimpleNamespace(general=types.SimpleNamespace(run_id="abcd1234"))
    trainer.device = torch.device("cpu")
    return trainer


def test_rollout_state_without_fe_noise_keeps_the_old_keys(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    trainer._save_rollout_state(LatentState(None, None, None, Z0), 30, 7)
    raw = torch.load(trainer._rollout_state_path(), weights_only=False)
    assert set(raw) == {"latent_state", "done_steps", "idx", "run_id"}
    Trainer._check_resume_setting(raw, "fe_noise", None)
    with pytest.raises(ValueError, match="fe_noise mismatch"):
        Trainer._check_resume_setting(raw, "fe_noise", FeNoise("token_rms", 0.1))


def test_rollout_state_carries_the_setting(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    fn = FeNoise("token_rms", 0.03, seed=1)
    trainer._save_rollout_state(LatentState(None, None, None, Z0), 30, 7, fe_noise=fn)
    state = trainer._load_rollout_state()
    assert state["fe_noise"] == fn.as_dict()
    Trainer._check_resume_setting(state, "fe_noise", FeNoise("token_rms", 0.03, seed=1))
    for other in (
        None,
        FeNoise("token_rms", 0.03, seed=2),
        FeNoise("token_rms", 0.1, seed=1),
        FeNoise("train_like", 0.03, seed=1),
        FeNoise("token_rms", 0.03, seed=1, dropout=True),
        FeNoise("token_rms", 0.03, seed=1, steps="first"),
    ):
        with pytest.raises(ValueError, match="fe_noise mismatch"):
            Trainer._check_resume_setting(state, "fe_noise", other)


def test_resumed_noisy_freeze_rollout_equals_one_chunk(monkeypatch, tmp_path):
    """Save after a chunk, reload, seed a ModelOutput as the trainer does, continue."""
    trainer = _trainer(monkeypatch, tmp_path)
    model = _stub_model()
    model.forecast_engine = _DropoutFE(0.3)
    kw = {
        "land_cells": LandCells("freeze", LAND),
        "fe_noise": FeNoise("token_rms", 0.1, dropout=True, seed=4),
    }
    one, _ = _rollout(model, 6, **kw)
    batch = _batch(6)
    out = Model.forward(model, MODEL_PARAMS, batch, 3, **kw)
    trainer._save_rollout_state(
        out.get_last_latent_state(),
        3,
        0,
        land_cells=kw["land_cells"],
        land_cells_init=out.land_cells_init,
        fe_noise=kw["fe_noise"],
    )
    st = trainer._load_rollout_state()
    Trainer._check_resume_setting(st, "fe_noise", kw["fe_noise"])
    seed = ModelOutput(0, batch=batch, step_offset=st["done_steps"])
    seed.latent = [{"latent_state": st["latent_state"]}]
    seed.land_cells_init = st["land_cells_init"]
    cont = Model.forward(model, MODEL_PARAMS, seed, 3, **kw)
    z = torch.stack([cont.latent[i]["latent_state"].z_pre_norm for i in range(3)])
    torch.testing.assert_close(z, one[3:], rtol=0, atol=0)
