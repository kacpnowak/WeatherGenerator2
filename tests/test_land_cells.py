"""Opt-in treatment of the latent land cells in validation / inference rollouts
(forecast.land_cells: freeze | no_forcing).

Model.forward is called unbound on a stub ``self`` (as in test_latent_filter.py) with one aux
token and four cell tokens; the forcing is additive and the forecast engine mixes the cells,
so land tokens influence the ocean tokens.
"""

import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common import config
from weathergen.datasets.batch import BatchSamples
from weathergen.datasets.utils import coords_to_hpyidxs
from weathergen.model.engines import LatentState
from weathergen.model.land_cells import LandCells, land_cell_mask
from weathergen.model.latent_filter import LatentFilter
from weathergen.model.model import Model
from weathergen.train.trainer import Trainer

MODEL_PARAMS = SimpleNamespace(rope_coords=None)
NUM_AUX = 1
LAND = torch.tensor([False, True, False, True])  # 4 cells, cells 1 and 3 are land
LAND_TOK = torch.cat([torch.zeros(NUM_AUX, dtype=torch.bool), LAND])  # token-level mask
Z0 = (torch.arange(15, dtype=torch.float64).reshape(1, 5, 3) - 7.0) / 7.0
MESH = Path(
    "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
)


class _Batch(BatchSamples):
    def get_num_source_steps(self) -> int:
        return 1


def _batch(num_steps: int, offset: int = 1) -> _Batch:
    output_len = max(1, offset + num_steps)
    batch = _Batch(["stream"], 1, output_len, list(range(offset, output_len)))
    # per-step forcing: one value per cell token, broadcast over the token dim
    batch.forcing = [
        0.1 * (i + 1) * torch.arange(1, 5, dtype=torch.float64) for i in range(output_len)
    ]
    batch.conditions = [torch.empty(0) for _ in range(output_len)]
    return batch


def _fe(z):
    """Forecast engine stub: alternating self term plus a cell-mixing term."""
    return -0.9 * z + 0.3 * z.mean(dim=1, keepdim=True) + 0.05


def _forcing(tokens, condition, field, num_aux, cell_weight=None):
    w = (
        torch.ones(tokens.shape[1] - num_aux, dtype=tokens.dtype)
        if cell_weight is None
        else cell_weight
    )
    patch = tokens[:, num_aux:] + (w * field)[None, :, None]
    return torch.cat([tokens[:, :num_aux], patch], dim=1), condition


class _Forcing:
    """Forcing-module stub with an injection `mode`."""

    def __init__(self, fn, mode):
        self.fn, self.mode = fn, mode

    def __call__(self, *args):
        return self.fn(*args)


def _stub_model(forcing=_forcing, forcing_mode="additive"):
    def predict_decoders(model_params, output_step, batch_step, tokens, batch, output):
        output.add_physical_prediction(output_step, "stream", tokens.clone())
        return output

    def predict_latent(model_params, step, tokens, batch, output):
        output.add_latent_prediction(step, "latent_state", LatentState(None, None, None, tokens))
        return output

    return SimpleNamespace(
        encoder=lambda params, batch: (Z0.clone(), None),
        forecast_engine=lambda tokens, condition, coords=None: _fe(tokens),
        forcing_module=_Forcing(forcing, forcing_mode),
        predict_decoders=predict_decoders,
        predict_latent=predict_latent,
        cf=SimpleNamespace(training_config={}),
        training=False,
        compute_cos_sim_to_prev=False,
        num_aux_tokens=NUM_AUX,
        _warned_missing_forcing=False,
    )


def _rollout(model, num_steps, chunks=None, **kwargs):
    """Latents z_1..z_N of a rollout run in `chunks` (default: one chunk)."""
    batch = _batch(num_steps)
    chunks = chunks or [num_steps]
    assert sum(chunks) == num_steps
    source, latents = batch, []
    for n in chunks:
        source = Model.forward(model, MODEL_PARAMS, source, n, **kwargs)
        latents += [source.latent[i]["latent_state"].z_pre_norm for i in range(n)]
    return torch.stack(latents), source


def _lc(mode):
    return LandCells(mode, LAND)


# ── config and mask ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", [None, {}, {"mode": None}, {"mode": "none"}])
def test_unset_or_none_disables_it(cfg):
    cfg = None if cfg is None else OmegaConf.create(cfg)
    assert LandCells.from_config(cfg, land_mask=LAND.numpy()) is None


def test_config_is_parsed():
    cfg = OmegaConf.create({"mode": "freeze", "expected_num_land_cells": 2})
    lc = LandCells.from_config(cfg, land_mask=LAND.numpy())
    assert lc.mode == "freeze" and lc.needs_init and lc.num_land_cells == 2
    assert not _lc("no_forcing").needs_init
    d = lc.as_dict()
    assert d["mode"] == "freeze" and d["num_land_cells"] == 2 and d["num_cells"] == 4
    assert d == _lc("freeze").as_dict()
    assert d != LandCells("freeze", torch.tensor([True, False, False, True])).as_dict()


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"mode": "melt"}, "mode"),
        ({"mode": "freeze", "nu": 0.1}, "unknown"),
        ({"mode": "freeze", "expected_num_land_cells": 3}, "expected 3"),
    ],
)
def test_invalid_config_is_rejected(cfg, match):
    with pytest.raises(ValueError, match=match):
        LandCells.from_config(OmegaConf.create(cfg), land_mask=LAND.numpy())


def test_mask_from_a_mesh_file(tmp_path):
    xr = pytest.importorskip("xarray")
    # points in four of the twelve level-0 cells
    lat = np.array([60.0, 61.0, 0.0, -60.0, -45.0])
    lon = np.array([10.0, 12.0, 100.0, -100.0, -95.0])
    path = tmp_path / "mesh.nc"
    xr.Dataset({"lat": ("n", lat), "lon": ("n", lon)}).to_netcdf(path)
    streams = {
        "TEMP": {"type": "fesom", "mesh_path": str(path)},
        "IFS_ATMO": {"type": "forcing", "mesh_path": "/nonexistent.nc"},  # ignored
        "OTHER": {"type": "anemoi"},
    }
    land = land_cell_mask(streams, 0)
    ocean_cells = set(coords_to_hpyidxs(0, lat, lon).tolist())
    assert land.shape == (12,)
    assert set(np.flatnonzero(~land).tolist()) == ocean_cells
    with pytest.raises(ValueError, match="mesh_path"):
        land_cell_mask({"OTHER": {"type": "anemoi"}}, 0)


@pytest.mark.skipif(not MESH.exists(), reason="CORE2 mesh not available")
def test_core2_mesh_has_2863_land_cells_at_level_5():
    lc = LandCells.from_config(
        OmegaConf.create({"mode": "freeze", "expected_num_land_cells": 2863}),
        {"TEMP": {"type": "fesom", "mesh_path": str(MESH)}},
        5,
    )
    assert lc.land_mask.numel() == 12 * 4**5 and lc.num_land_cells == 2863


# ── Model.forward ─────────────────────────────────────────────────────────────────────


def test_default_rollout_is_unchanged():
    model = _stub_model()
    plain, out = _rollout(model, 6)
    again, out2 = _rollout(model, 6, land_cells=None)
    assert torch.equal(plain, again)
    assert out.land_cells_init is None and out2.land_cells_init is None
    # the stub's land tokens do move without the option
    assert not torch.allclose(
        plain[:, :, LAND_TOK], Z0[:, LAND_TOK].expand_as(plain[:, :, LAND_TOK])
    )


def test_freeze_resets_land_tokens_every_step():
    model = _stub_model()
    z, out = _rollout(model, 6, land_cells=_lc("freeze"))
    for n in range(6):
        assert torch.equal(z[n][:, LAND_TOK], Z0[:, LAND_TOK])
    torch.testing.assert_close(out.land_cells_init, Z0[:, LAND_TOK])
    # ocean tokens: the reference recursion with the land reset after each step
    ref = Z0.clone()
    batch = _batch(6)
    for n in range(6):
        ref, _ = _forcing(ref, None, batch.forcing[n + 1], NUM_AUX)
        ref = _fe(ref)
        ref[:, LAND_TOK] = Z0[:, LAND_TOK]
        torch.testing.assert_close(z[n], ref, rtol=0, atol=0)


def test_no_forcing_equals_forcing_masked_to_ocean():
    model = _stub_model()
    z, out = _rollout(model, 6, land_cells=_lc("no_forcing"))
    assert out.land_cells_init is None
    ocean_only = (~LAND).to(torch.float64)
    ref_model = _stub_model(forcing=lambda t, c, f, a: _forcing(t, c, f, a, cell_weight=ocean_only))
    ref, _ = _rollout(ref_model, 6)
    torch.testing.assert_close(z, ref, rtol=0, atol=0)
    plain, _ = _rollout(model, 6)
    assert not torch.allclose(z, plain)


@pytest.mark.parametrize("chunks", [[2, 2, 2], [1, 5], [4, 2], [1, 1, 1, 1, 1, 1]])
@pytest.mark.parametrize("mode", ["freeze", "no_forcing"])
@pytest.mark.parametrize("with_filter", [False, True])
def test_chunked_rollout_equals_one_chunk(chunks, mode, with_filter):
    model = _stub_model()
    kw = {"land_cells": _lc(mode)}
    if with_filter:
        kw["latent_filter"] = LatentFilter("second_difference", 0.1)
    one, _ = _rollout(model, 6, **kw)
    chunked, _ = _rollout(model, 6, chunks=chunks, **kw)
    torch.testing.assert_close(chunked, one, rtol=0, atol=0)


def test_freeze_continuation_needs_the_initial_land_tokens():
    model = _stub_model()
    _, out = _rollout(model, 2, land_cells=_lc("freeze"))
    out.land_cells_init = None
    with pytest.raises(ValueError, match="initial land tokens"):
        Model.forward(model, MODEL_PARAMS, out, 2, land_cells=_lc("freeze"))


def test_rejected_in_training_mode():
    model = _stub_model()
    model.training = True
    with pytest.raises(ValueError, match="inference-time"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, land_cells=_lc("freeze"))


@pytest.mark.parametrize("forcing_mode", ["global", "adaln_local"])
def test_no_forcing_needs_a_token_forcing_mode(forcing_mode):
    model = _stub_model(forcing_mode=forcing_mode)
    with pytest.raises(ValueError, match="no_forcing"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, land_cells=_lc("no_forcing"))
    # freeze does not depend on the forcing mode
    Model.forward(model, MODEL_PARAMS, _batch(2), 2, land_cells=_lc("freeze"))


def test_mask_size_must_match_the_latent():
    model = _stub_model()
    with pytest.raises(ValueError, match="cell tokens"):
        Model.forward(
            model,
            MODEL_PARAMS,
            _batch(2),
            2,
            land_cells=LandCells("freeze", torch.ones(5, dtype=bool)),
        )


# ── resumable rollout state ───────────────────────────────────────────────────────────


def _trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_path_results", lambda cf: tmp_path / cf.general.run_id)
    trainer = Trainer.__new__(Trainer)
    trainer.cf = types.SimpleNamespace(general=types.SimpleNamespace(run_id="abcd1234"))
    trainer.device = torch.device("cpu")
    return trainer


def test_rollout_state_without_land_cells_keeps_the_old_keys(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    trainer._save_rollout_state(LatentState(None, None, None, Z0), 30, 7)
    raw = torch.load(trainer._rollout_state_path(), weights_only=False)
    assert set(raw) == {"latent_state", "done_steps", "idx", "run_id"}
    Trainer._check_resume_setting(raw, "land_cells", None)
    with pytest.raises(ValueError, match="land_cells mismatch"):
        Trainer._check_resume_setting(raw, "land_cells", _lc("freeze"))


def test_rollout_state_carries_the_setting_and_initial_land_tokens(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    lc = _lc("freeze")
    init = Z0[:, LAND_TOK]
    trainer._save_rollout_state(
        LatentState(None, None, None, Z0), 30, 7, land_cells=lc, land_cells_init=init
    )
    state = trainer._load_rollout_state()
    assert state["land_cells"] == lc.as_dict()
    torch.testing.assert_close(state["land_cells_init"], init)
    Trainer._check_resume_setting(state, "land_cells", lc)
    Trainer._check_resume_setting(state, "latent_filter", None)
    for other in (None, _lc("no_forcing"), LandCells("freeze", torch.tensor([1, 0, 0, 1]))):
        with pytest.raises(ValueError, match="land_cells mismatch"):
            Trainer._check_resume_setting(state, "land_cells", other)


def test_resumed_freeze_rollout_equals_one_chunk(monkeypatch, tmp_path):
    """Save after a chunk, reload, seed a ModelOutput as the trainer does, continue."""
    from weathergen.model.model import ModelOutput

    trainer = _trainer(monkeypatch, tmp_path)
    model = _stub_model()
    lc = _lc("freeze")
    one, _ = _rollout(model, 6, land_cells=lc)
    batch = _batch(6)
    out = Model.forward(model, MODEL_PARAMS, batch, 3, land_cells=lc)
    trainer._save_rollout_state(
        out.get_last_latent_state(), 3, 0, land_cells=lc, land_cells_init=out.land_cells_init
    )
    st = trainer._load_rollout_state()
    seed = ModelOutput(0, batch=batch, step_offset=st["done_steps"])
    seed.latent = [{"latent_state": st["latent_state"]}]
    seed.land_cells_init = st["land_cells_init"]
    cont = Model.forward(model, MODEL_PARAMS, seed, 3, land_cells=lc)
    z = torch.stack([cont.latent[i]["latent_state"].z_pre_norm for i in range(3)])
    torch.testing.assert_close(z, one[3:], rtol=0, atol=0)
