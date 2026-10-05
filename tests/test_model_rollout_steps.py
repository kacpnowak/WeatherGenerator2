"""Rollout bookkeeping of Model.forward: forecast-engine applications, forcing and decodes.

Model.forward is called unbound on a stub ``self`` whose forecast engine counts its calls,
whose forcing module records the index of the field it receives and whose decoder records
the step it decodes and how many forecast-engine applications preceded it.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from weathergen.datasets.batch import BatchSamples
from weathergen.model.engines import LatentState
from weathergen.model.model import Model

NUM_TOKENS, DIM = 3, 4
MODEL_PARAMS = SimpleNamespace(rope_coords=None)


class _Batch(BatchSamples):
    def get_num_source_steps(self) -> int:
        return 1


def _batch(offset: int, num_steps: int) -> _Batch:
    """Batch laid out as ModelBatch does: output_len = offset + num_steps (at least 1)."""
    output_len = max(1, offset + num_steps)
    batch = _Batch(["stream"], 1, output_len, list(range(offset, output_len)))
    # every forcing field carries its own index, so the stub reports which one it consumed
    batch.forcing = [torch.full((1,), float(i)) for i in range(output_len)]
    batch.conditions = [torch.empty(0) for _ in range(output_len)]
    return batch


def _stub_model():
    log = SimpleNamespace(fe_calls=0, forcing=[], decodes=[])

    def forecast_engine(tokens, condition, coords=None):
        log.fe_calls += 1
        return tokens + 1.0

    def forcing_module(tokens, condition, field, num_aux):
        present = torch.is_tensor(field) and field.numel() > 0
        log.forcing.append(int(field.item()) if present else None)
        return tokens, condition

    def predict_decoders(model_params, output_step, batch_step, tokens, batch, output):
        log.decodes.append(
            {"output_step": output_step, "batch_step": batch_step, "fe_calls": log.fe_calls}
        )
        return output

    def predict_latent(model_params, step, tokens, batch, output):
        state = LatentState(None, None, None, tokens)
        output.add_latent_prediction(step, "latent_state", state)
        return output

    model = SimpleNamespace(
        encoder=lambda params, batch: (torch.zeros(len(batch), NUM_TOKENS, DIM), None),
        forecast_engine=forecast_engine,
        forcing_module=forcing_module,
        predict_decoders=predict_decoders,
        predict_latent=predict_latent,
        cf=SimpleNamespace(training_config={}),
        training=False,
        compute_cos_sim_to_prev=False,
        num_aux_tokens=0,
        _warned_missing_forcing=False,
    )
    return model, log


def _rollout_steps(batch, call: str) -> int:
    # training (and the teachers) pass the batch's output length, validation and inference
    # pass the number of forecast steps (one chunk covering all of them)
    return batch.get_output_len() if call == "train" else len(batch.get_output_idxs())


@pytest.mark.parametrize("call", ["train", "validation"])
def test_offset_one_applies_the_forecast_engine_once_per_forecast_step(call):
    model, log = _stub_model()
    batch = _batch(offset=1, num_steps=2)
    Model.forward(model, MODEL_PARAMS, batch, _rollout_steps(batch, call))

    assert log.fe_calls == 2
    assert log.forcing == [1, 2]
    assert [(d["batch_step"], d["fe_calls"]) for d in log.decodes] == [(1, 1), (2, 2)]


def test_full_range_call_stores_predictions_under_the_absolute_step():
    model, log = _stub_model()
    batch = _batch(offset=1, num_steps=2)
    output = Model.forward(model, MODEL_PARAMS, batch, batch.get_output_len())

    # the loss and the output writer read the full-range output at the batch's output indices
    assert [d["output_step"] for d in log.decodes] == [1, 2]
    assert len(output.physical) == batch.get_output_len()
    assert "latent_state" not in output.latent[0]
    assert all("latent_state" in output.latent[i] for i in batch.get_output_idxs())


def test_chunk_call_stores_predictions_relative_to_the_chunk():
    model, log = _stub_model()
    batch = _batch(offset=1, num_steps=2)
    Model.forward(model, MODEL_PARAMS, batch, len(batch.get_output_idxs()))

    assert [d["output_step"] for d in log.decodes] == [0, 1]


@pytest.mark.parametrize("call", ["train", "validation"])
def test_offset_zero_is_unchanged(call):
    model, log = _stub_model()
    batch = _batch(offset=0, num_steps=2)
    Model.forward(model, MODEL_PARAMS, batch, _rollout_steps(batch, call))

    assert log.fe_calls == 2
    assert log.forcing == [0, 1]
    assert [(d["output_step"], d["batch_step"], d["fe_calls"]) for d in log.decodes] == [
        (0, 0, 1),
        (1, 1, 2),
    ]


def test_chunked_rollout_continues_at_the_next_forecast_step():
    model, log = _stub_model()
    batch = _batch(offset=1, num_steps=4)
    first = Model.forward(model, MODEL_PARAMS, batch, 2)
    Model.forward(model, MODEL_PARAMS, first, 2)

    assert log.fe_calls == 4
    assert log.forcing == [1, 2, 3, 4]
    assert [(d["batch_step"], d["fe_calls"]) for d in log.decodes] == [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ]
    assert [d["output_step"] for d in log.decodes] == [0, 1, 0, 1]


def test_missing_forcing_field_warns_once_per_model(caplog):
    model, log = _stub_model()
    batch = _batch(offset=1, num_steps=2)
    batch.forcing = [[] for _ in batch.forcing]  # e.g. the target batch a teacher receives
    with caplog.at_level(logging.WARNING, logger="weathergen.model.model"):
        Model.forward(model, MODEL_PARAMS, batch, batch.get_output_len())
        Model.forward(model, MODEL_PARAMS, batch, batch.get_output_len())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "step 1" in warnings[0].getMessage()
    assert "without forcing" in warnings[0].getMessage()
    # no behaviour change otherwise: the steps still run, through the forcing module
    assert log.fe_calls == 4
    assert log.forcing == [None] * 4


def test_present_forcing_does_not_warn(caplog):
    model, _ = _stub_model()
    batch = _batch(offset=1, num_steps=2)
    with caplog.at_level(logging.WARNING, logger="weathergen.model.model"):
        Model.forward(model, MODEL_PARAMS, batch, batch.get_output_len())

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
