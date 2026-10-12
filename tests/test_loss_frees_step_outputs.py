"""The training loss must not keep a step's predictions and targets alive after it returns.

torch.tensor(<OmegaConf ListConfig>) used to leave a reference cycle (OmegaConf answers torch's
__cuda_array_interface__ / __dlpack__ probes with exceptions whose tracebacks refer to themselves)
that kept LossPhysical.compute_loss's frame, and with it the ModelOutput (and so the whole source
batch on the GPU) and the targets, alive until Python's cyclic garbage collector ran. The tests run
with the collector disabled, so anything that is not freed by reference counting stays alive.
"""

import gc
import weakref

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.batch import SampleMetaData
from weathergen.model.model import ModelOutput
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.target_and_aux_module_base import TargetAuxOutput
from weathergen.train.utils import TRAIN

CHANNELS = ["a", "b", "c"]
OFFSET, NUM_STEPS, NUM_POINTS = 1, 2, 5


@pytest.fixture
def no_cyclic_gc():
    gc.collect()
    gc.disable()
    yield
    gc.enable()
    gc.collect()


def _config():
    stream = {
        "name": "S",
        "type": "fesom",
        "train_target_channels": CHANNELS,
        "val_target_channels": CHANNELS,
        # a ListConfig once the config is an OmegaConf object, as in training
        "target_channel_weights": [1.0, 2.0, 0.5],
        "location_weight": "cosine_latitude",
        "tokenize_spacetime": True,
    }
    cf = OmegaConf.create({"streams": {"S": stream}})
    mode_cfg = OmegaConf.create(
        {
            "forecast": {"offset": OFFSET},
            "losses": {
                "physical": {
                    "type": "LossPhysical",
                    "loss_fcts": {"mse": {}, "dynamic_loss": {"window": 8, "L": 20.0}},
                }
            },
        }
    )
    return cf, mode_cfg


def _step_outputs():
    """Predictions, targets and batch metadata of one training step (one sample, one stream)."""
    output_len = OFFSET + NUM_STEPS
    output_idxs = list(range(OFFSET, output_len))
    preds = ModelOutput(output_len)
    targets = TargetAuxOutput(output_len, output_idxs)
    params = {"loss": ["mse"], "relationship": "independent"}
    target_meta = {"S": SampleMetaData(params={}, global_params={**params, "idx": 0})}
    source_meta = SampleMetaData(params={}, global_params={**params, "correspondence": 0})
    latent = torch.randn(4, requires_grad=True)
    for step in output_idxs:
        target = torch.randn(NUM_POINTS, len(CHANNELS))
        pred = latent.sum() * torch.ones(1, NUM_POINTS, len(CHANNELS))
        preds.add_physical_prediction(step, "S", torch.split(pred, [NUM_POINTS], dim=1))
        targets.add_physical_target(
            step,
            "S",
            {
                "target": [target],
                "target_times": [np.full(NUM_POINTS, np.datetime64("2000-01-01T00:00"))],
                "target_coords": [torch.rand(NUM_POINTS, 2) * 90.0],
                "target_metda_data": [target_meta],
                "is_spoof": [False],
                "is_forecast_query": [False],
                "idxs_inv": [torch.arange(NUM_POINTS)],
            },
        )
    metadata = (np.zeros(1, dtype=np.int32), [source_meta], [[0]], [target_meta["S"]])
    return preds, {"physical": targets}, metadata


def test_channel_weights_leave_no_reference_cycle(no_cyclic_gc):
    cf, mode_cfg = _config()
    loss_calculator = LossCalculator(cf, mode_cfg, TRAIN, device="cpu")
    loss_module = loss_calculator.loss_calculators["physical"][0][1]

    def caller():  # holds a local, like compute_loss holds the step's predictions
        local = torch.zeros(1)
        ref = weakref.ref(local)
        loss_module._get_weights("S", cf.streams.S)
        return ref

    assert caller()() is None, "the caller's frame was kept alive by a reference cycle"


def test_training_loss_frees_predictions_and_targets(no_cyclic_gc):
    cf, mode_cfg = _config()
    loss_calculator = LossCalculator(cf, mode_cfg, TRAIN, device="cpu")

    preds, targets_and_auxs, metadata = _step_outputs()
    preds_ref = weakref.ref(preds)
    targets_ref = weakref.ref(targets_and_auxs["physical"])

    loss = loss_calculator.compute_loss(
        preds=preds, targets_and_aux=targets_and_auxs, metadata=metadata
    )
    loss.backward()
    del preds, targets_and_auxs, metadata, loss

    assert preds_ref() is None, "the ModelOutput survived the step"
    assert targets_ref() is None, "the targets survived the step"
