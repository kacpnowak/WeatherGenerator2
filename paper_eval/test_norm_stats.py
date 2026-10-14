"""Tests for the normalization statistics paper_eval uses to denormalize the stored
`source` (persistence baseline, snapshot initial conditions, selfcheck).

    OMP_NUM_THREADS=4 .venv/bin/python -m pytest paper_eval/test_norm_stats.py -p no:cacheprovider

No real data is needed: a tiny fake truth archive and run directory are built in
`tmp_path`. A run that sets the stream option `normalization_stats` (wet-point
statistics) must be denormalized with the JSON's values for the channels it lists
and with the truth archive's statistics for every other channel; a run without the
option must get exactly the archive statistics.
"""

import json
import warnings

import numpy as np
import pytest
import zarr

from paper_eval import compute_damped, scores, selfcheck, snapshots
from paper_eval.config import apply_normalization_overrides, model_config_path

STREAM = "FESOM_NODE"
# archive column order: lon, lat first, then the channels (means/std cover the channels only)
COLNAMES = ["lon", "lat", "ssh", "sst", "temp_2", "salt_2", "const"]
MEANS = [0.5, 10.0, 12.0, 34.0, 7.0]
STDS = [0.2, 3.0, 11.0, 0.9, 1e-7]  # "const": near-zero std -> treated as 1.0
# source channel order differs from the archive's, and omits `ssh`
SRC_CHANNELS = ["salt_2", "const", "sst", "temp_2"]
# archive stats in SRC_CHANNELS order (std rule applied)
ARCHIVE_MEAN = np.array([34.0, 7.0, 10.0, 12.0], dtype=np.float32)
ARCHIVE_STD = np.array([0.9, 1.0, 3.0, 11.0], dtype=np.float32)

WET = {
    "_meta": {"stream": "node", "years": [2000, 2208]},
    "temp_2": {"mean": 12.5, "std": 10.5},
    "salt_2": {"mean": 34.5, "std": 0.8},
    "temp_999": {"mean": -1.0, "std": 1.0},  # not a source channel: ignored
}


def _make_run(tmp_path, stream_cfg, *, stats=WET, write_model_json=True):
    """Fake truth archive + `models/<run>/model_<run>_chkpt00000.json`; returns cfg."""
    truth = tmp_path / "truth"
    truth.mkdir(parents=True)
    g = zarr.open_group(str(truth / "2000.zarr"), mode="w")
    data = g.create_array("data", shape=(2, len(COLNAMES)), dtype="f4")
    data.attrs["colnames"] = COLNAMES
    data.attrs["means"] = MEANS
    data.attrs["std"] = STDS

    run_id = "fakerun1"
    if write_model_json:
        run_dir = tmp_path / "models" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / f"model_{run_id}_chkpt00000.json").write_text(
            json.dumps({"streams": {STREAM: {"type": "fesom", **stream_cfg}}})
        )
    if stats is not None:
        (tmp_path / "wet.json").write_text(json.dumps(stats))
    return {
        "run_id": run_id,
        "mini_epoch": 0,
        "truth": {STREAM: str(truth)},
        "models_dir": str(tmp_path / "models"),
        "repo_root": str(tmp_path),
    }


def test_without_option_is_archive_statistics(tmp_path):
    cfg = _make_run(tmp_path, {})
    mean, std = scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)
    assert mean.dtype == std.dtype == np.float32
    np.testing.assert_array_equal(mean, ARCHIVE_MEAN)
    np.testing.assert_array_equal(std, ARCHIVE_STD)


def test_option_set_on_other_stream_only_is_archive_statistics(tmp_path):
    cfg = _make_run(tmp_path, {})
    model_json = model_config_path(cfg)
    streams = json.loads(model_json.read_text())["streams"]
    streams["FESOM_ELEM"] = {"type": "fesom", "normalization_stats": "wet.json"}
    model_json.write_text(json.dumps({"streams": streams}))
    mean, std = scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)
    np.testing.assert_array_equal(mean, ARCHIVE_MEAN)
    np.testing.assert_array_equal(std, ARCHIVE_STD)


def test_missing_model_config_warns_and_keeps_archive_statistics(tmp_path):
    cfg = _make_run(tmp_path, {}, write_model_json=False)
    with pytest.warns(UserWarning, match="no model config"):
        mean, std = scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)
    np.testing.assert_array_equal(mean, ARCHIVE_MEAN)
    np.testing.assert_array_equal(std, ARCHIVE_STD)


def test_option_set_overrides_listed_channels_only(tmp_path):
    cfg = _make_run(tmp_path, {"normalization_stats": "wet.json"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean, std = scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)
    assert mean.dtype == std.dtype == np.float32
    # SRC_CHANNELS = salt_2, const, sst, temp_2 -> salt_2 and temp_2 overridden
    np.testing.assert_array_equal(mean, np.array([34.5, 7.0, 10.0, 12.5], dtype=np.float32))
    np.testing.assert_array_equal(std, np.array([0.8, 1.0, 3.0, 10.5], dtype=np.float32))


def test_override_applied_after_near_zero_std_rule(tmp_path):
    # the reader maps archive std <= 1e-5 to 1.0 BEFORE applying overrides, so a small
    # (but valid, > 0) override std on such a channel is kept as is
    stats = {"const": {"mean": 3.0, "std": 1e-6}}
    cfg = _make_run(tmp_path, {"normalization_stats": "wet.json"}, stats=stats)
    mean, std = scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)
    assert (mean[1], std[1]) == (np.float32(3.0), np.float32(1e-6))
    np.testing.assert_array_equal(np.delete(std, 1), np.delete(ARCHIVE_STD, 1))


def test_invalid_override_raises(tmp_path):
    cfg = _make_run(
        tmp_path, {"normalization_stats": "wet.json"}, stats={"sst": {"mean": 1.0, "std": 0.0}}
    )
    with pytest.raises(ValueError, match="std > 0"):
        scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)


def test_missing_stats_file_raises(tmp_path):
    cfg = _make_run(tmp_path, {"normalization_stats": "nope.json"}, stats=None)
    with pytest.raises(FileNotFoundError, match="nope.json"):
        scores._load_norm_stats(cfg, STREAM, SRC_CHANNELS)


def test_selfcheck_norm_stats_matches(tmp_path):
    plain = _make_run(tmp_path / "plain", {})
    wet = _make_run(tmp_path / "wet", {"normalization_stats": "wet.json"})
    for name, mean, std in zip(SRC_CHANNELS, ARCHIVE_MEAN, ARCHIVE_STD, strict=True):
        assert selfcheck._norm_stats(plain, STREAM, name) == pytest.approx(
            (float(mean), float(std)), rel=1e-6
        )
    assert selfcheck._norm_stats(wet, STREAM, "temp_2") == (12.5, 10.5)
    assert selfcheck._norm_stats(wet, STREAM, "salt_2") == (34.5, 0.8)
    assert selfcheck._norm_stats(wet, STREAM, "sst") == (10.0, 3.0)
    assert selfcheck._norm_stats(wet, STREAM, "const") == (7.0, 1.0)


def test_snapshots_and_damped_use_the_same_statistics(tmp_path, monkeypatch):
    # snapshots.py calls scores' function directly
    assert snapshots._load_norm_stats is scores._load_norm_stats
    # compute_damped.py goes through scores.init_worker with the whole cfg (so the
    # run id / mini epoch the lookup needs reach `_load_norm_stats`)
    seen = {}
    monkeypatch.setattr(scores, "init_worker", lambda c: seen.update(c))
    cfg = {**_make_run(tmp_path, {"normalization_stats": "wet.json"}), "regions": {"global": None}}
    compute_damped.init_worker(cfg, ["sst"])
    for key in ("run_id", "mini_epoch", "truth", "models_dir", "repo_root"):
        assert seen[key] == cfg[key]


def test_matches_reader_implementation():
    reader = pytest.importorskip("weathergen.readers_extra.data_reader_fesom")
    mean = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    std = np.array([1.0, 1.0, 4.0, 5.0, 6.0])
    names = ["lon", "lat", "a", "b", "c"]
    overrides = {"_meta": {}, "b": {"mean": -2.0, "std": 0.5}, "zzz": {"mean": 1.0, "std": 1.0}}
    want = reader.apply_normalization_overrides(names, mean, std, overrides)
    got = apply_normalization_overrides(names, mean, std, overrides)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    for bad in ({"a": {"mean": 0.0, "std": -1.0}}, {"a": {"mean": np.nan, "std": 1.0}}):
        with pytest.raises(ValueError):
            reader.apply_normalization_overrides(names, mean, std, bad)
        with pytest.raises(ValueError):
            apply_normalization_overrides(names, mean, std, bad)
