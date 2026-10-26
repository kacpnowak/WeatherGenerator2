"""The mean-error loss penalises a systematic offset, not the scatter around it."""

import math

import pytest
import torch

from weathergen.train.loss_modules import loss_functions
from weathergen.train.loss_modules.loss_functions import mean_error_sq


def test_uniform_offset_is_penalised():
    target = torch.randn(7, 2)
    pred = (target + 0.3)[None]
    loss, loss_chs = mean_error_sq(target, pred, None, None)
    assert loss_chs == pytest.approx([0.09, 0.09])
    assert loss.item() == pytest.approx(0.09)


def test_zero_mean_error_is_free():
    target = torch.zeros(4, 1)
    pred = torch.tensor([1.0, -1.0, 1.0, -1.0]).reshape(1, 4, 1)
    loss, loss_chs = mean_error_sq(target, pred, None, torch.ones(4))
    assert loss_chs.tolist() == [0.0]
    assert loss.item() == 0.0


def test_nan_targets_are_ignored():
    target = torch.tensor([[0.0], [float("nan")], [0.0]])
    pred = torch.tensor([[1.0], [100.0], [1.0]])[None]
    _, loss_chs = mean_error_sq(target, pred, None, None)
    # the NaN row is excluded from both the sum and the count
    assert loss_chs.tolist() == [1.0]


def test_all_nan_channel_contributes_zero():
    target = torch.tensor([[0.0, float("nan")], [0.0, float("nan")]])
    pred = torch.ones(1, 2, 2)
    loss, loss_chs = mean_error_sq(target, pred, None, None)
    assert loss_chs.tolist() == [1.0, 0.0]
    assert math.isfinite(loss.item())


def test_channel_weights_scale_the_mean():
    target = torch.zeros(3, 2)
    pred = torch.stack([torch.full((3,), 0.5), torch.full((3,), 2.0)], dim=-1)[None]
    loss, loss_chs = mean_error_sq(target, pred, torch.tensor([1.0, 3.0]), None)
    assert loss_chs.tolist() == pytest.approx([0.25, 4.0])
    assert loss.item() == pytest.approx((0.25 + 3 * 4.0) / 2)


def test_point_weights_and_ensemble_mean():
    target = torch.zeros(2, 1)
    # ensemble mean of the members is [1, 3]
    pred = torch.tensor([[[0.0], [2.0]], [[2.0], [4.0]]])
    _, loss_chs = mean_error_sq(target, pred, None, torch.tensor([1.0, 0.5]))
    # weighted mean (1 * 1 + 0.5 * 3) / (1 + 0.5) = 5 / 3
    assert loss_chs.tolist() == pytest.approx([(5 / 3) ** 2])


def test_uniform_offset_is_independent_of_point_weights():
    target = torch.randn(5, 2)
    pred = (target + 0.3)[None]
    weights = torch.tensor([0.1, 2.0, 0.7, 5.0, 0.3])
    _, loss_chs = mean_error_sq(target, pred, None, weights)
    assert loss_chs.tolist() == pytest.approx([0.09, 0.09], abs=1e-6)


def test_point_weights_normalise_over_valid_points_only():
    target = torch.tensor([[0.0], [float("nan")], [0.0]])
    pred = torch.tensor([[1.0], [100.0], [1.0]])[None]
    _, loss_chs = mean_error_sq(target, pred, None, torch.tensor([1.0, 50.0, 3.0]))
    assert loss_chs.tolist() == pytest.approx([1.0])


def test_all_nan_channel_with_point_weights_is_zero():
    target = torch.full((2, 1), float("nan"))
    loss, loss_chs = mean_error_sq(target, torch.ones(1, 2, 1), None, torch.tensor([1.0, 2.0]))
    assert loss_chs.tolist() == [0.0]
    assert loss.item() == 0.0


def test_selectable_by_name():
    name = "mean_error_sq"  # as in LossPhysical: looked up by the name from loss_fcts
    assert getattr(loss_functions, name) is mean_error_sq
