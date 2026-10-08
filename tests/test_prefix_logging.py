"""The training spin-up length reaches the terminal log and the training metrics."""

import types

from weathergen.datasets.batch import BatchSamples
from weathergen.train.trainer import prefix_length
from weathergen.utils.train_logger import TrainLogger


def test_prefix_length():
    batch = BatchSamples(["s"], 1, 6, [1, 2, 3, 4, 5])
    assert prefix_length(batch) == 0
    batch.first_supervised_step = 4
    assert prefix_length(batch) == 3


def test_add_logs_writes_extra_training_metrics():
    logged = []
    stub = types.SimpleNamespace(log_metrics=lambda stage, metrics: logged.append(metrics))
    TrainLogger.add_logs(
        stub, "train", 8, {}, {}, avg_loss=[1.0], lr=1e-5,
        extra_metrics={"forecast_prefix_steps": 42.5},
    )
    assert logged[0]["forecast_prefix_steps"] == 42.5
    assert logged[0]["loss_avg_mean"] == 1.0
