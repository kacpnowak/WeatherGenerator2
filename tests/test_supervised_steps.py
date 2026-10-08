"""The first forecast step that carries predictions for a stream."""

from weathergen.train.loss_modules.supervised_steps import first_supervised_timestep


def test_first_step_with_predictions():
    preds = [{}, {}, {}, {"S": [1]}, {"S": [2]}]
    assert first_supervised_timestep(preds, "S") == 3


def test_without_spin_up_it_is_the_first_forecast_step():
    assert first_supervised_timestep([{}, {"S": [1]}, {"S": [2]}], "S") == 1


def test_stream_without_predictions():
    assert first_supervised_timestep([{}, {"T": [1]}], "S") is None
