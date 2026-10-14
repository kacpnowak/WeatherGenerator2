"""Pure functions of diag_forced_year.py."""

import numpy as np

from diag_forced_year import band_mask, variability_ratio, weighted_mean


def test_weighted_mean_ignores_zero_weight_points():
    x = np.array([1.0, 3.0, np.nan])
    w = np.array([1.0, 3.0, 0.0])
    assert weighted_mean(x, w) == 2.5


def test_band_mask_follows_the_paper_eval_rule():
    lat = np.array([-90.0, -30.0, 0.0, 29.99, 30.0, 89.9])
    assert band_mask(lat, (-30.0, 30.0)).tolist() == [False, True, True, True, False, False]
    assert band_mask(lat, None).all()


def test_perfect_forecast_has_ratio_one():
    rng = np.random.default_rng(0)
    truth, clim, w = rng.normal(size=100), rng.normal(size=100), rng.uniform(1, 2, 100)
    assert np.isclose(variability_ratio(truth, truth, clim, w), 1.0)


def test_ratio_measures_spread_not_bias():
    rng = np.random.default_rng(1)
    truth, clim, w = rng.normal(size=100), rng.normal(size=100), rng.uniform(1, 2, 100)
    pred = clim + 0.5 * (truth - clim) + 3.0  # half the anomaly, plus a uniform offset
    assert np.isclose(variability_ratio(pred, truth, clim, w), 0.5)
