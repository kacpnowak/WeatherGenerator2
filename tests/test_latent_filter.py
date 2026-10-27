"""Opt-in latent filter of validation / inference rollouts (forecast.latent_filter).

Model.forward is called unbound on a stub ``self`` (as in test_model_rollout_steps.py) whose
forecast engine is a simple affine map, so the filtered trajectories can be written down.
"""

import types
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common import config
from weathergen.datasets.batch import BatchSamples
from weathergen.model.engines import LatentState
from weathergen.model.latent_filter import LatentFilter
from weathergen.model.model import Model
from weathergen.train.trainer import Trainer

MODEL_PARAMS = SimpleNamespace(rope_coords=None)


class _Batch(BatchSamples):
    def get_num_source_steps(self) -> int:
        return 1


def _batch(num_steps: int, offset: int = 1) -> _Batch:
    output_len = max(1, offset + num_steps)
    batch = _Batch(["stream"], 1, output_len, list(range(offset, output_len)))
    batch.forcing = [torch.full((1,), float(i)) for i in range(output_len)]
    batch.conditions = [torch.empty(0) for _ in range(output_len)]
    return batch


def _stub_model(fe, z0):
    """Stub model: encoder returns z0 (1, 2, 3), the forecast engine is `fe`."""

    def predict_decoders(model_params, output_step, batch_step, tokens, batch, output):
        output.add_physical_prediction(output_step, "stream", tokens.clone())
        return output

    def predict_latent(model_params, step, tokens, batch, output):
        output.add_latent_prediction(step, "latent_state", LatentState(None, None, None, tokens))
        return output

    return SimpleNamespace(
        encoder=lambda params, batch: (z0.clone(), None),
        forecast_engine=lambda tokens, condition, coords=None: fe(tokens),
        forcing_module=lambda tokens, condition, field, num_aux: (tokens, condition),
        predict_decoders=predict_decoders,
        predict_latent=predict_latent,
        cf=SimpleNamespace(training_config={}),
        training=False,
        compute_cos_sim_to_prev=False,
        num_aux_tokens=0,
        _warned_missing_forcing=False,
    )


def _rollout(model, num_steps, chunks=None, latent_filter=None):
    """Latents z_1..z_N of a rollout run in `chunks` (default: one chunk)."""
    batch = _batch(num_steps)
    chunks = chunks or [num_steps]
    assert sum(chunks) == num_steps
    kwargs = {} if latent_filter is None else {"latent_filter": latent_filter}
    source, latents = batch, []
    for n in chunks:
        source = Model.forward(model, MODEL_PARAMS, source, n, **kwargs)
        latents += [source.latent[i]["latent_state"].z_pre_norm for i in range(n)]
    return torch.stack(latents)


Z0 = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3) / 7.0


# ── config ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", [None, {}, {"type": None}, {"type": "none", "nu": 0.1}])
def test_unset_or_none_disables_the_filter(cfg):
    cfg = None if cfg is None else OmegaConf.create(cfg)
    assert LatentFilter.from_config(cfg) is None


def test_config_is_parsed():
    f = LatentFilter.from_config(OmegaConf.create({"type": "second_difference", "nu": 0.25}))
    assert f == LatentFilter("second_difference", 0.25)
    assert f.needs_prev
    assert not LatentFilter("relax", 0.1).needs_prev
    assert f.as_dict() == {"type": "second_difference", "nu": 0.25}


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"type": "robert", "nu": 0.1}, "type"),
        ({"type": "relax"}, "nu"),
        ({"type": "relax", "nu": 1.0}, "nu"),
        ({"type": "relax", "nu": -0.1}, "nu"),
        ({"type": "relax", "nu": 0.1, "alpha": 0.5}, "unknown"),
    ],
)
def test_invalid_config_is_rejected(cfg, match):
    with pytest.raises(ValueError, match=match):
        LatentFilter.from_config(OmegaConf.create(cfg))


# ── formulas ──────────────────────────────────────────────────────────────────────────


def test_filter_formulas():
    z_new, z_cur, z_prev = torch.tensor(3.0), torch.tensor(1.0), torch.tensor(-2.0)
    assert LatentFilter("relax", 0.25)(z_new, z_cur, z_prev) == 0.75 * 3.0 + 0.25 * 1.0
    # second difference 3 - 2 + (-2) = -1
    assert LatentFilter("second_difference", 0.25)(z_new, z_cur, z_prev) == 3.0 + 0.25
    # no z_{n-1}: unfiltered
    assert LatentFilter("second_difference", 0.25)(z_new, z_cur, None) == 3.0


# ── Model.forward ─────────────────────────────────────────────────────────────────────


def test_no_filter_is_the_unfiltered_rollout():
    model = _stub_model(lambda z: -0.9 * z + 0.3, Z0)
    plain = _rollout(model, 5)
    batch = _batch(5)
    out = Model.forward(model, MODEL_PARAMS, batch, 5, latent_filter=None)
    latents = torch.stack([out.latent[i]["latent_state"].z_pre_norm for i in range(5)])
    assert torch.equal(plain, latents)
    assert out.latent_filter_prev is None


def test_relax_matches_the_closed_form():
    nu = 0.25
    model = _stub_model(lambda z: -z, Z0)  # eigenvalue -1: pure alternation
    z = _rollout(model, 6, latent_filter=LatentFilter("relax", nu))
    expected = torch.stack([(-(1 - 2 * nu)) ** n * Z0 for n in range(1, 7)])
    torch.testing.assert_close(z, expected)


def test_second_difference_damps_the_alternating_mode():
    model = _stub_model(lambda z: -z, Z0)
    plain = _rollout(model, 40)
    filtered = _rollout(model, 40, latent_filter=LatentFilter("second_difference", 0.25))
    assert torch.allclose(plain[-1].abs(), Z0.abs())  # unfiltered: no damping
    # lam = -1, nu = 0.25: a complex pair with |mu| = sqrt(nu) = 0.5 per step
    assert filtered[-1].norm() < 1e-9 * Z0.norm()
    assert filtered[-2].norm() > 0


def test_second_difference_leaves_a_linear_trend_unchanged():
    model = _stub_model(lambda z: z + 1.0, Z0)  # z_n = z_0 + n: zero second difference
    plain = _rollout(model, 6)
    filtered = _rollout(model, 6, latent_filter=LatentFilter("second_difference", 0.25))
    torch.testing.assert_close(filtered, plain)
    # relaxation slows it down instead
    relaxed = _rollout(model, 6, latent_filter=LatentFilter("relax", 0.25))
    torch.testing.assert_close(relaxed[-1], Z0 + 6 * 0.75)


def test_second_difference_first_step_is_unfiltered():
    model = _stub_model(lambda z: -2.0 * z + 1.0, Z0)
    plain = _rollout(model, 1)
    filtered = _rollout(model, 1, latent_filter=LatentFilter("second_difference", 0.25))
    torch.testing.assert_close(filtered, plain)


@pytest.mark.parametrize("chunks", [[2, 2, 2], [1, 5], [4, 2], [1, 1, 1, 1, 1, 1]])
@pytest.mark.parametrize("ftype", ["relax", "second_difference"])
def test_chunked_rollout_equals_one_chunk(chunks, ftype):
    model = _stub_model(lambda z: -0.95 * z.flip(-1) + 0.1, Z0)
    f = LatentFilter(ftype, 0.1)
    one = _rollout(model, 6, latent_filter=f)
    chunked = _rollout(model, 6, chunks=chunks, latent_filter=f)
    torch.testing.assert_close(chunked, one, rtol=0, atol=0)


def test_filter_is_rejected_in_training_mode():
    model = _stub_model(lambda z: z, Z0)
    model.training = True
    with pytest.raises(ValueError, match="inference-time"):
        Model.forward(model, MODEL_PARAMS, _batch(2), 2, latent_filter=LatentFilter("relax", 0.1))


# ── resumable rollout state ───────────────────────────────────────────────────────────


def _trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_path_results", lambda cf: tmp_path / cf.general.run_id)
    trainer = Trainer.__new__(Trainer)
    trainer.cf = types.SimpleNamespace(general=types.SimpleNamespace(run_id="abcd1234"))
    trainer.device = torch.device("cpu")
    return trainer


def test_rollout_state_without_filter_keeps_the_old_keys(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    trainer._save_rollout_state(LatentState(None, None, None, Z0), 30, 7)
    raw = torch.load(trainer._rollout_state_path(), weights_only=False)
    assert set(raw) == {"latent_state", "done_steps", "idx", "run_id"}
    state = trainer._load_rollout_state()
    assert state.get("latent_filter") is None and state.get("latent_filter_prev") is None


def test_rollout_state_carries_the_filter_and_previous_latent(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, tmp_path)
    f = LatentFilter("second_difference", 0.1)
    trainer._save_rollout_state(
        LatentState(None, None, None, Z0), 30, 7, latent_filter=f, latent_filter_prev=-Z0
    )
    state = trainer._load_rollout_state()
    assert state["latent_filter"] == f.as_dict()
    torch.testing.assert_close(state["latent_filter_prev"], -Z0)
    torch.testing.assert_close(state["latent_state"].z_pre_norm, Z0)
