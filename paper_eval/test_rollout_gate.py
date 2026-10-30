"""Tests for the rollout stability gate (synthetic data only).

OMP_NUM_THREADS=4 .venv/bin/python -m pytest paper_eval/test_rollout_gate.py -p no:cacheprovider
"""

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from paper_eval import rollout_gate as rg
from weathergen.common.io import ItemKey, OutputDataset, OutputItem, TimeRange, ZipZarrIO

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


def test_short_series_fails_naming_the_shortfall():
    leads = np.arange(10, 50, 10)
    ok, msg = rg.gate_verdict(leads, np.ones(leads.size), True)
    assert not ok
    assert "lead 40" in msg and "80 short" in msg


def test_empty_series_fails():
    ok, msg = rg.gate_verdict(np.array([], dtype=int), np.array([]), True)
    assert not ok and "no leads" in msg


def test_series_reaching_exactly_min_lead_passes():
    leads = np.array([60, 120])
    assert rg.gate_verdict(leads, np.ones(2), True)[0]
    assert not rg.gate_verdict(np.array([60, 119]), np.ones(2), True)[0]
    assert rg.gate_verdict(np.array([60, 119]), np.ones(2), True, min_lead=119)[0]


def test_nonfinite_location_is_reported():
    ok, msg = rg.gate_verdict(LEADS, np.ones(LEADS.size), False, nonfinite="stream SALT, lead 7")
    assert not ok and "stream SALT, lead 7" in msg


# ── run_gate on a synthetic rollout, written the way the validation writer lays it out ──────
NPTS = 4
INIT = np.datetime64("2209-01-01T00:00", "ns")
STREAMS = {"TEMP": ["temp_2", "temp_7"], "SALT": ["salt_2"]}


def _truth_temp2(day: int) -> NDArray:
    return 10.0 + 0.01 * day + 0.1 * np.arange(NPTS)


def _write_truth(root, ndays=60):
    g = zarr.open_group(str(root / f"{root.name}_2209.zarr"), mode="w")
    cols = ["lon", "lat", "temp_2", "salt_2"]
    data = np.zeros((ndays * NPTS, len(cols)), dtype="f4")
    for d in range(ndays):
        data[d * NPTS : (d + 1) * NPTS, 2] = _truth_temp2(d)
    arr = g.create_array("data", shape=data.shape, dtype="f4")
    arr[:] = data
    arr.attrs["colnames"] = cols
    arr.attrs["nod2"] = NPTS
    stamps = np.datetime64("2209-01-01T23:30", "ns") + np.repeat(
        np.arange(ndays), NPTS
    ) * np.timedelta64(1, "D")
    dates = g.create_array("dates", shape=(ndays * NPTS, 1), dtype="datetime64[ns]")
    dates[:] = stamps[:, None]


def _dataset(name, key, channels, values):
    n = values.shape[0]
    return OutputDataset(
        name=name,
        item_key=key,
        source_interval=TimeRange(INIT, INIT + np.timedelta64(1, "D")),
        data=values.astype("f4")[:, :, None],
        times=np.full(n, INIT, dtype="datetime64[ns]"),
        coords=np.stack([np.arange(n, dtype="f8"), np.arange(n, dtype="f8")], axis=1),
        geoinfo=np.zeros((n, 0), dtype="f4"),
        channels=channels,
        geoinfo_channels=[],
    )


def _write_rollout(results, n_leads, nan_at=None):
    """One rank archive, one sample, forecast steps 0 (source only) .. n_leads. The SST
    prediction is the truth plus 0.5 at the three wet nodes; the dry node holds 99.
    `nan_at` = (stream, lead) puts a NaN into that prediction."""
    results.mkdir()
    with ZipZarrIO(results / "validation_chkpt00000_rank0000.zip", read_only=False) as zio:
        for stream, channels in STREAMS.items():
            for lead in range(n_leads + 1):
                key = ItemKey(0, lead, stream)
                if lead == 0:
                    src = _dataset("source", key, channels, np.zeros((NPTS, len(channels))))
                    zio.write_zarr(OutputItem(key, 1, source=src))
                    continue
                pred = np.zeros((NPTS, len(channels)))
                if stream == "TEMP":
                    pred[:, 0] = _truth_temp2(lead) + 0.5
                    pred[3, 0] = 99.0  # dry node: unconstrained
                if nan_at == (stream, lead):
                    pred[1, 0] = np.nan
                tar = _dataset("target", key, channels, np.zeros((NPTS, len(channels))))
                prd = _dataset("prediction", key, channels, pred)
                zio.write_zarr(OutputItem(key, 1, target=tar, prediction=prd))


def _fake_mesh():
    from paper_eval.meshtools import Mesh

    return Mesh(
        lon=np.arange(NPTS, dtype=float),
        lat=np.arange(NPTS, dtype=float),
        tri_elem=np.zeros((1, 3), dtype=int),
        nod_area=np.ones((48, NPTS)),
        elem_area=np.ones(1),
        n_wet_layers_nod=np.array([2, 2, 2, 0]),
        n_wet_layers_elem=np.ones(1, dtype=int),
        nz1=np.array([2.5, 7.5]),
    )


def _gate(tmp_path, monkeypatch, capsys, n_leads, nan_at=None, extra=()):
    archive = tmp_path / "ocean_node"
    archive.mkdir()
    _write_truth(archive)
    _write_rollout(tmp_path / "results", n_leads, nan_at)
    monkeypatch.setattr(rg, "load_mesh", lambda path: _fake_mesh())
    code = rg.main(
        [
            *("--results-dir", str(tmp_path / "results"), "--mini-epoch", "0"),
            *("--archive", str(archive), "--mesh", "unused.nc", "--every", "10"),
            *("--streams", "TEMP", "SALT", "--min-lead", "30", *extra),
        ]
    )
    return code, capsys.readouterr().out


def test_run_gate_passes_a_healthy_rollout(tmp_path, monkeypatch, capsys):
    code, out = _gate(tmp_path, monkeypatch, capsys, n_leads=30)
    assert code == 0, out
    assert "PASS" in out
    assert "lead   10  temp_2 RMSE   0.5000" in out and "lead   30" in out


def test_run_gate_fails_a_nan_naming_stream_and_lead(tmp_path, monkeypatch, capsys):
    # lead 7 is not a scored lead (every 10th): finiteness is checked at every lead
    code, out = _gate(tmp_path, monkeypatch, capsys, n_leads=30, nan_at=("SALT", 7))
    assert code == 1
    assert "FAIL" in out and "stream SALT, lead 7" in out


def test_run_gate_fails_a_nan_in_the_sst_at_a_scored_lead(tmp_path, monkeypatch, capsys):
    code, out = _gate(tmp_path, monkeypatch, capsys, n_leads=30, nan_at=("TEMP", 20))
    assert code == 1
    assert "FAIL" in out and "lead 20" in out


def test_run_gate_fails_a_rollout_that_stops_early(tmp_path, monkeypatch, capsys):
    code, out = _gate(tmp_path, monkeypatch, capsys, n_leads=18)
    assert code == 1
    assert "stops at lead 18" in out and "12 short" in out
