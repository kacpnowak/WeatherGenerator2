# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Deep-SSL / temporal-JEPA plumbing: tap parsing, the fused student path in
Model.predict_latent, the per-level loss guard, the teacher time offset and the EMA reset.

Model.predict_latent and LossLatentSSLStudentTeacher.compute_loss are called unbound on stub
``self`` objects, as in test_model_rollout_steps.py.
"""

import functools
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.multi_stream_data_sampler import resolve_teacher_time_offset
from weathergen.model.ema import EMAModel
from weathergen.model.encoder import parse_deep_ssl_taps
from weathergen.model.model import Model, ModelOutput
from weathergen.train.loss_modules.loss_module_ssl import LossLatentSSLStudentTeacher

# ---------------------------------------------------------------------------
# tap_after parsing
# ---------------------------------------------------------------------------


def _deep(tap_after, enabled=True):
    return OmegaConf.create({"enabled": enabled, "tap_after": tap_after})


def test_taps_parsed_in_forward_order():
    assert parse_deep_ssl_taps(_deep(["local2global", "global:0", "global:1"]), 4) == (
        True,
        {0, 1},
    )
    assert parse_deep_ssl_taps(_deep(["global:2"]), 4) == (False, {2})
    assert parse_deep_ssl_taps(_deep(["local2global"]), 4) == (True, None)


def test_taps_disabled():
    assert parse_deep_ssl_taps(None, 4) == (False, None)
    assert parse_deep_ssl_taps(_deep(["global:0"], enabled=False), 4) == (False, None)
    assert parse_deep_ssl_taps(_deep([]), 4) == (False, None)


@pytest.mark.parametrize(
    "tap_after",
    [
        ["global:1", "global:0"],  # out of forward order: level weights would be misassigned
        ["global:0", "local2global"],
        ["global:0", "global:0"],  # duplicate: the engine would emit it only once
        ["local2global", "local2global"],
    ],
)
def test_taps_out_of_order_or_duplicate_rejected(tap_after):
    with pytest.raises(ValueError, match="forward order"):
        parse_deep_ssl_taps(_deep(tap_after), 4)


@pytest.mark.parametrize("tap", ["global:4", "global:-1"])
def test_taps_outside_global_engine_rejected(tap):
    with pytest.raises(ValueError, match="outside the global engine"):
        parse_deep_ssl_taps(_deep([tap]), 4)


def test_unknown_tap_rejected():
    with pytest.raises(ValueError, match="Unknown deep_ssl tap"):
        parse_deep_ssl_taps(_deep(["aggregation:0"]), 4)


# ---------------------------------------------------------------------------
# Model.predict_latent: student (fused) and teacher (per level) deep-SSL paths
# ---------------------------------------------------------------------------


class _CountingHead(torch.nn.Module):
    """Latent head: scale * patch tokens, counting its calls."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, latent_state):
        self.calls += 1
        return latent_state.patch_tokens * self.scale


def _stub_model(heads, fusion=None, projections=None):
    stub = SimpleNamespace(
        latent_pre_norm=lambda x: x + 100.0,
        latent_heads=heads,
        deep_ssl_fusion=fusion,
        deep_ssl_level_projections=projections,
        register_token_idxs=[],
        class_token_idxs=[],
        num_aux_tokens=0,
    )
    stub.tokens_to_latent_state = functools.partial(Model.tokens_to_latent_state, stub)
    return stub


def _levels():
    # three taps and the final tokens, (batch 1, 2 tokens, dim 3)
    taps = [torch.full((1, 2, 3), float(i)) for i in range(3)]
    return taps, torch.full((1, 2, 3), 10.0)


def test_student_fused_path_runs_predictor_once_and_exposes_final_level():
    head = _CountingHead(2.0)
    # fusion: mean over levels; per-level projection i multiplies by (i + 1)
    fusion = lambda levels: torch.stack(levels).mean(0)  # noqa: E731
    projections = {"JEPA": [functools.partial(torch.mul, other=i + 1.0) for i in range(4)]}
    stub = _stub_model({"JEPA": head}, fusion, projections)
    taps, tokens = _levels()

    out = Model.predict_latent(stub, None, 0, tokens, None, ModelOutput(1), taps)

    assert head.calls == 1, "the plain predictor pass on the final latent must be skipped"
    fused = (torch.stack(taps + [tokens]).mean(0) + 100.0) * 2.0
    level_preds = out.latent_deep[0]["JEPA"]
    assert len(level_preds) == 4
    for i, pred in enumerate(level_preds):
        torch.testing.assert_close(pred, fused * (i + 1))
    # the plain latent entry (collapse monitor) is the final-level prediction
    assert out.latent[0]["JEPA"] is level_preds[-1]
    assert "latent_state" in out.latent[0]


def test_student_without_taps_uses_plain_head():
    head = _CountingHead(2.0)
    stub = _stub_model({"JEPA": head}, lambda levels: levels[0], {"JEPA": [torch.nn.Identity()]})
    _, tokens = _levels()

    out = Model.predict_latent(stub, None, 0, tokens, None, ModelOutput(1), [])

    assert head.calls == 1
    torch.testing.assert_close(out.latent[0]["JEPA"], (tokens + 100.0) * 2.0)
    assert out.latent_deep is None


def test_teacher_per_level_reuses_final_plain_prediction():
    head = _CountingHead(1.0)
    stub = _stub_model({"JEPA": head})
    taps, tokens = _levels()

    out = Model.predict_latent(stub, None, 0, tokens, None, ModelOutput(1), taps)

    assert head.calls == 4, "one pass per tap plus the final latent, no duplicate"
    level_preds = out.latent_deep[0]["JEPA"]
    assert len(level_preds) == 4
    for level, pred in zip(taps + [tokens], level_preds, strict=True):
        torch.testing.assert_close(pred, level + 100.0)
    assert level_preds[-1] is out.latent[0]["JEPA"]


# ---------------------------------------------------------------------------
# loss: no silent fallback to the final level when deep SSL is enabled
# ---------------------------------------------------------------------------


def _loss_stub():
    return SimpleNamespace(level_weights=[0.5, 0.5], device="cpu", losses={}, context_loss_weight=0)


@pytest.mark.parametrize(
    "student_deep, teacher_deep, missing",
    [(None, {"JEPA": []}, "student"), ([{"JEPA": []}], None, "teacher")],
)
def test_loss_refuses_missing_deep_levels(student_deep, teacher_deep, missing):
    preds = SimpleNamespace(latent_deep=student_deep, latent=[{}])
    targets = SimpleNamespace(latent_deep=teacher_deep, latent={}, aux_outputs=None)
    with pytest.raises(ValueError, match=f"the {missing} produced no per-level"):
        LossLatentSSLStudentTeacher.compute_loss(_loss_stub(), preds, targets, (None,) * 4)


# ---------------------------------------------------------------------------
# teacher_time_offset
# ---------------------------------------------------------------------------


def _mode(training_mode, offset):
    return OmegaConf.create({"training_mode": training_mode, "teacher_time_offset": offset})


def test_teacher_offset_active_with_student_teacher_only():
    assert resolve_teacher_time_offset(_mode(["student_teacher"], 1)) == 1
    assert resolve_teacher_time_offset(_mode(["student_teacher"], 0)) == 0
    assert resolve_teacher_time_offset(OmegaConf.create({"training_mode": ["masking"]})) == 0
    # a single mode as a plain string
    assert resolve_teacher_time_offset(_mode("student_teacher", 1)) == 1


def test_teacher_offset_ignored_without_student_teacher():
    # an offset inherited from the pretraining config must not shift forecast targets
    assert resolve_teacher_time_offset(_mode(["masking"], 1)) == 0
    assert resolve_teacher_time_offset(_mode(["forecasting"], 2)) == 0


@pytest.mark.parametrize(
    "modes", [["masking", "student_teacher"], ["student_teacher", "forecasting"]]
)
def test_teacher_offset_with_a_second_mode_rejected(modes):
    with pytest.raises(NotImplementedError, match="alone"):
        resolve_teacher_time_offset(_mode(modes, 1))
    # without an offset the combination is fine
    assert resolve_teacher_time_offset(_mode(modes, 0)) == 0


# ---------------------------------------------------------------------------
# EMAModel.reset: non-persistent buffers are copied from the model
# ---------------------------------------------------------------------------


class _WithBuffers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3))
        self.register_buffer("persistent", torch.randn(3))
        self.register_buffer("transient", torch.randn(3), persistent=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EMAModel.reset allocates on cuda")
def test_ema_reset_initialises_non_persistent_buffers():
    model = _WithBuffers().cuda()
    with torch.device("meta"):
        empty = _WithBuffers()

    ema = EMAModel(model, empty)

    for name in ("weight", "persistent", "transient"):
        torch.testing.assert_close(getattr(ema.ema_model, name), getattr(model, name))
