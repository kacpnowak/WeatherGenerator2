"""Training spin-up (forecast.prefix_steps): range parsing and the per-step draw."""

import pytest

from weathergen.datasets.multi_stream_data_sampler import (
    draw_prefix_steps,
    resolve_prefix_range,
    resolve_prefix_zero_fraction,
)


def test_training_reads_the_prefix_range():
    assert resolve_prefix_range({"prefix_steps": [0, 90]}, "train") == (0, 90)
    assert resolve_prefix_range({"prefix_steps": [5, 5]}, "train") == (5, 5)


@pytest.mark.parametrize("forecast_cfg", [{}, {"prefix_steps": None}])
def test_no_prefix_steps_means_no_prefix(forecast_cfg):
    assert resolve_prefix_range(forecast_cfg, "train") == (0, 0)


@pytest.mark.parametrize("stage", ["val", "test"])
def test_validation_and_test_ignore_the_inherited_prefix(stage):
    # validation/test configs inherit the training forecast settings
    assert resolve_prefix_range({"prefix_steps": [0, 90]}, stage) == (0, 0)


@pytest.mark.parametrize("bad", [[90, 0], [-1, 5], [5], [0, 1.5], 10, "0,90"])
def test_malformed_prefix_steps_are_rejected(bad):
    with pytest.raises(ValueError, match="prefix_steps"):
        resolve_prefix_range({"prefix_steps": bad}, "train")


def test_draw_depends_only_on_seed_mini_epoch_and_position():
    first = [draw_prefix_steps((0, 90), 7, 2, p) for p in range(50)]
    again = [draw_prefix_steps((0, 90), 7, 2, p) for p in range(50)]
    assert first == again
    assert all(0 <= p <= 90 for p in first)
    assert len(set(first)) > 10
    assert first != [draw_prefix_steps((0, 90), 7, 3, p) for p in range(50)]


def test_draw_covers_both_bounds():
    assert {draw_prefix_steps((5, 7), 1, 0, p) for p in range(200)} == {5, 6, 7}


def test_zero_range_always_draws_zero():
    assert {draw_prefix_steps((0, 0), 1, 0, p) for p in range(20)} == {0}


def test_zero_fraction_draws_that_share_of_zeros():
    draws = [draw_prefix_steps((0, 365), 7, 2, p, zero_fraction=0.25) for p in range(4000)]
    zeros = draws.count(0) / len(draws)
    assert 0.22 < zeros < 0.28
    assert max(draws) > 300


def test_zero_fraction_keeps_ranks_in_step():
    first = [draw_prefix_steps((0, 365), 7, 2, p, zero_fraction=0.25) for p in range(50)]
    assert first == [draw_prefix_steps((0, 365), 7, 2, p, zero_fraction=0.25) for p in range(50)]


def test_no_zero_fraction_keeps_the_old_draws():
    # runs started before the option must replay the same spin-ups on resume
    old = [draw_prefix_steps((0, 90), 7, 2, p) for p in range(50)]
    assert old == [draw_prefix_steps((0, 90), 7, 2, p, zero_fraction=0.0) for p in range(50)]


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.25"])
def test_malformed_zero_fraction_is_rejected(bad):
    with pytest.raises(ValueError, match="prefix_zero_fraction"):
        resolve_prefix_zero_fraction(
            {"prefix_steps": [0, 365], "prefix_zero_fraction": bad}, "train"
        )


def test_zero_fraction_only_applies_in_training():
    cfg = {"prefix_steps": [0, 365], "prefix_zero_fraction": 0.25}
    assert resolve_prefix_zero_fraction(cfg, "train") == 0.25
    assert resolve_prefix_zero_fraction(cfg, "val") == 0.0
    assert resolve_prefix_zero_fraction({}, "train") == 0.0
