import math

import torch

from weathergen.train.loss_modules.loss_functions import mse, rss

_NAN = float("nan")


def _pred(values: list[list[float]]) -> torch.Tensor:
    """Wrap a (num_points, num_channels) prediction in the ensemble dimension."""
    return torch.tensor(values).unsqueeze(0)


def test_mse_averages_over_valid_points_only():
    # Channel 0 is fully observed, channel 1 has a single valid point. Both have an
    # absolute error of 2.0 wherever the target is defined, so both must report 4.0.
    target = torch.tensor([[0.0, 0.0], [0.0, _NAN], [0.0, _NAN], [0.0, _NAN]])
    pred = _pred([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    _, loss_chs = mse(target, pred, weights_channels=None, weights_points=None)

    assert loss_chs[0].item() == 4.0
    assert loss_chs[1].item() == 4.0


def test_mse_with_point_weights_averages_over_valid_points_only():
    target = torch.tensor([[0.0, 0.0], [0.0, _NAN]])
    pred = _pred([[2.0, 2.0], [2.0, 2.0]])
    weights_points = torch.tensor([0.5, 0.5])

    _, loss_chs = mse(target, pred, weights_channels=None, weights_points=weights_points)

    assert loss_chs[0].item() == 2.0
    assert loss_chs[1].item() == 2.0


def test_mse_is_unchanged_when_no_target_is_missing():
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    pred = _pred([[1.0, 1.0], [0.0, 1.0]])

    _, loss_chs = mse(target, pred, weights_channels=None, weights_points=None)

    assert loss_chs[0].item() == 2.5
    assert loss_chs[1].item() == 2.0


def test_mse_of_channel_without_valid_points_is_exactly_zero():
    # DynamicLossEMA relies on this to recognise channels that carry no information.
    target = torch.tensor([[0.0, _NAN], [0.0, _NAN]])
    pred = _pred([[1.0, 1.0], [1.0, 1.0]])

    _, loss_chs = mse(target, pred, weights_channels=None, weights_points=None)

    assert loss_chs[1].item() == 0.0
    assert not math.isnan(loss_chs[1].item())


def test_rss_still_sums_over_points():
    target = torch.tensor([[0.0, 0.0], [0.0, _NAN]])
    pred = _pred([[2.0, 2.0], [2.0, 2.0]])

    _, loss_chs = rss(target, pred, weights_channels=None, weights_points=None)

    assert loss_chs[0].item() == 8.0
    assert loss_chs[1].item() == 4.0
