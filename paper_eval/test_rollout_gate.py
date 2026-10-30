"""Tests for the rollout stability gate (synthetic data only).

OMP_NUM_THREADS=4 .venv/bin/python -m pytest paper_eval/test_rollout_gate.py -p no:cacheprovider
"""

import numpy as np
import pytest
import zarr

from paper_eval import rollout_gate as rg

LEADS = np.arange(10, 130, 10)


def test_healthy_series_passes():
    ok, msg = rg.gate_verdict(LEADS, np.linspace(0.3, 2.4, LEADS.size), True)
    assert ok
    assert "pass" in msg.lower()


def test_nan_fails_and_names_the_lead():
    rmse = np.linspace(0.3, 2.4, LEADS.size)
    rmse[4] = np.nan
    ok, msg = rg.gate_verdict(LEADS, rmse, True)
    assert not ok
    assert "lead 50" in msg


def test_inf_fails():
    rmse = np.ones(LEADS.size)
    rmse[2] = np.inf
    ok, msg = rg.gate_verdict(LEADS, rmse, True)
    assert not ok and "lead 30" in msg


def test_not_all_finite_fails_even_with_a_healthy_rmse():
    ok, msg = rg.gate_verdict(LEADS, np.ones(LEADS.size), False)
    assert not ok
    assert "finite" in msg


def test_exactly_the_threshold_fails():
    rmse = np.ones(LEADS.size)
    rmse[6] = 3.0
    ok, msg = rg.gate_verdict(LEADS, rmse, True)
    assert not ok and "lead 70" in msg


def test_just_below_the_threshold_passes():
    assert rg.gate_verdict(LEADS, np.full(LEADS.size, np.nextafter(3.0, 0.0)), True)[0]


def test_names_the_first_failing_lead():
    rmse = np.ones(LEADS.size)
    rmse[[3, 8]] = [5.0, 9.0]
    ok, msg = rg.gate_verdict(LEADS, rmse, True)
    assert not ok
    assert "lead 40" in msg and "lead 90" not in msg


def test_threshold_is_a_parameter():
    rmse = np.full(LEADS.size, 2.0)
    assert rg.gate_verdict(LEADS, rmse, True)[0]
    assert not rg.gate_verdict(LEADS, rmse, True, max_sst_rmse=1.5)[0]


def test_weighted_rmse_uses_wet_points_and_weights():
    pred = np.array([1.0, 2.0, 100.0, 4.0])
    truth = np.array([0.0, 0.0, 0.0, 0.0])
    wet = np.array([True, True, False, True])
    w = np.array([1.0, 3.0, 1.0, 0.0])
    # (1*1 + 3*4) / (1 + 3) = 3.25
    assert rg.weighted_rmse(pred, truth, wet, w) == pytest.approx(np.sqrt(3.25))


def test_weighted_rmse_ignores_nan_at_dry_points_but_not_at_wet_ones():
    pred = np.array([1.0, np.nan])
    truth = np.zeros(2)
    w = np.ones(2)
    assert rg.weighted_rmse(pred, truth, np.array([True, False]), w) == pytest.approx(1.0)
    assert np.isnan(rg.weighted_rmse(pred, truth, np.array([True, True]), w))


def test_select_leads_every_nth_and_the_last():
    assert rg.select_leads(range(1, 121), 10) == list(range(10, 121, 10))
    assert rg.select_leads(range(1, 125), 10) == [*range(10, 121, 10), 124]
    assert rg.select_leads([1, 2, 3], 10) == [3]


def test_truth_day_is_the_init_date_plus_the_lead():
    day = rg.truth_day(np.datetime64("2209-03-04T12:00"), 10)
    assert day == np.datetime64("2209-03-14", "D")


def _fake_archive(root, year, npts=3, ndays=4):
    """`<root>/<root.name>_<year>.zarr`: `data` (ndays*npts, cols), `dates` (ndays*npts, 1)."""
    g = zarr.open_group(str(root / f"{root.name}_{year}.zarr"), mode="w")
    cols = ["lon", "lat", "temp_2", "sss"]
    data = np.zeros((ndays * npts, len(cols)), dtype="f4")
    for d in range(ndays):
        data[d * npts : (d + 1) * npts, 2] = 10.0 * d + np.arange(npts)
    arr = g.create_array("data", shape=data.shape, dtype="f4")
    arr[:] = data
    arr.attrs["colnames"] = cols
    arr.attrs["nod2"] = npts
    days = np.datetime64(f"{year}-01-01T23:30", "ns") + np.repeat(
        np.arange(ndays), npts
    ) * np.timedelta64(1, "D")
    dates = g.create_array("dates", shape=(ndays * npts, 1), dtype="datetime64[ns]")
    dates[:] = days[:, None]


def test_truth_archive_reads_the_day_block_of_a_channel(tmp_path):
    root = tmp_path / "ocean_node"
    root.mkdir()
    _fake_archive(root, 2209)
    archive = rg.TruthArchive(root)
    np.testing.assert_array_equal(
        archive.field(np.datetime64("2209-01-03", "D"), "temp_2"), [20.0, 21.0, 22.0]
    )
    with pytest.raises(KeyError):
        archive.field(np.datetime64("2209-01-09", "D"), "temp_2")


def test_truth_archive_crosses_into_the_next_year(tmp_path):
    root = tmp_path / "ocean_node"
    root.mkdir()
    _fake_archive(root, 2209)
    _fake_archive(root, 2210)
    archive = rg.TruthArchive(root)
    day = rg.truth_day(np.datetime64("2209-12-30T00:00"), 3)
    assert day == np.datetime64("2210-01-02", "D")
    np.testing.assert_array_equal(archive.field(day, "temp_2"), [10.0, 11.0, 12.0])
