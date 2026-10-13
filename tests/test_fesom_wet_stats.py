# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Wet-point normalisation statistics and dry-point handling of the FESOM reader.

The unit tests need no data. The real-data checks are skipped where the AWI-CM3 archive is absent.
"""

import functools
import json
from pathlib import Path

import numpy as np
import pytest

import weathergen.readers_extra.data_reader_fesom as fesom
from weathergen.datasets.data_reader_base import TimeWindowHandler

DATA_ROOT = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_reader(stream_info, mean, std, dtype=np.float64):
    """A reader that is already 'initialised': no files are opened."""
    reader = fesom.DataReaderFesom.__new__(fesom.DataReaderFesom)
    reader._initialized = True
    reader._stream_info = stream_info
    reader.source_idx = np.arange(len(mean))
    reader.target_idx = np.arange(len(mean))
    reader.src_mean_vec = np.asarray(mean, dtype=dtype)
    reader.src_stdev_vec = np.asarray(std, dtype=dtype)
    reader.trg_mean_vec = np.asarray(mean, dtype=dtype)
    reader.trg_stdev_vec = np.asarray(std, dtype=dtype)
    return reader


# --- apply_normalization_overrides -------------------------------------------------------------


def test_overrides_replace_only_the_listed_channels():
    colnames = ["lon", "lat", "sst", "salt_450", "temp_2"]
    mean = np.array([0.0, 0.0, 10.0, 23.87, 18.0])
    std = np.array([1.0, 1.0, 5.0, 15.0, 6.0])
    overrides = {"salt_450": {"mean": 34.74, "std": 0.82}}

    new_mean, new_std = fesom.apply_normalization_overrides(colnames, mean, std, overrides)

    np.testing.assert_array_equal(new_mean, [0.0, 0.0, 10.0, 34.74, 18.0])
    np.testing.assert_array_equal(new_std, [1.0, 1.0, 5.0, 0.82, 6.0])


def test_overrides_ignore_meta_and_unknown_channels():
    colnames = ["lon", "lat", "salt_450"]
    mean = np.array([0.0, 0.0, 23.87])
    std = np.array([1.0, 1.0, 15.0])
    overrides = {
        "_meta": {"years": [2000, 2208], "step": 73},
        "temp_9999": {"mean": 3.0, "std": 2.0},
        "salt_450": {"mean": 34.74, "std": 0.82},
    }

    new_mean, new_std = fesom.apply_normalization_overrides(colnames, mean, std, overrides)

    np.testing.assert_array_equal(new_mean, [0.0, 0.0, 34.74])
    np.testing.assert_array_equal(new_std, [1.0, 1.0, 0.82])


def test_overrides_return_copies_and_leave_inputs_unchanged():
    colnames = ["lon", "lat", "salt_450"]
    mean = np.array([0.0, 0.0, 23.87])
    std = np.array([1.0, 1.0, 15.0])

    new_mean, new_std = fesom.apply_normalization_overrides(
        colnames, mean, std, {"salt_450": {"mean": 34.74, "std": 0.82}}
    )
    assert new_mean is not mean and new_std is not std
    np.testing.assert_array_equal(mean, [0.0, 0.0, 23.87])
    np.testing.assert_array_equal(std, [1.0, 1.0, 15.0])

    # also copies when nothing is overridden
    same_mean, same_std = fesom.apply_normalization_overrides(colnames, mean, std, {})
    assert same_mean is not mean and same_std is not std
    np.testing.assert_array_equal(same_mean, mean)
    np.testing.assert_array_equal(same_std, std)


@pytest.mark.parametrize("bad_std", [0.0, -1.0])
def test_overrides_reject_non_positive_std(bad_std):
    colnames = ["lon", "lat", "salt_450"]
    mean = np.array([0.0, 0.0, 23.87])
    std = np.array([1.0, 1.0, 15.0])

    with pytest.raises(ValueError):
        fesom.apply_normalization_overrides(
            colnames, mean, std, {"salt_450": {"mean": 34.74, "std": bad_std}}
        )


# --- source_fill_value -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_source_fill_value_becomes_zero_after_normalisation(dtype):
    reader = _stub_reader({"name": "t", "source_fill_value": 0.0}, [2.0, 4.0], [2.0, 0.5])
    source = np.array([[0.0, 5.0], [3.0, 0.0]], dtype=dtype)

    out = reader.normalize_source_channels(source)

    assert out.dtype == dtype
    # dry points: exactly 0, i.e. the channel mean after normalisation
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0
    # wet points: (x - mean) / std as before
    np.testing.assert_allclose(out[0, 1], (5.0 - 4.0) / 0.5)
    np.testing.assert_allclose(out[1, 0], (3.0 - 2.0) / 2.0)


def test_source_fill_value_float32_with_float64_statistics_is_exactly_zero():
    # the real statistics are float64 and the data float32: the dry points must still be exact 0
    reader = _stub_reader(
        {"name": "t", "source_fill_value": 0.0}, [34.74123456789, 0.1], [0.82123456789, 3.0]
    )
    source = np.zeros((5, 2), dtype=np.float32)

    out = reader.normalize_source_channels(source)

    assert (out == 0.0).all()


def test_without_source_fill_value_zeros_are_normalised_as_before():
    reader = _stub_reader({"name": "t"}, [2.0, 4.0], [2.0, 0.5])
    source = np.array([[0.0, 5.0], [3.0, 0.0]])

    out = reader.normalize_source_channels(source)

    np.testing.assert_allclose(out, [[(0.0 - 2.0) / 2.0, 2.0], [0.5, (0.0 - 4.0) / 0.5]])


def test_source_fill_value_leaves_nan_alone():
    reader = _stub_reader({"name": "t", "source_fill_value": 0.0}, [2.0], [2.0])
    out = reader.normalize_source_channels(np.array([[np.nan], [0.0], [4.0]]))

    assert np.isnan(out[0, 0]) and out[1, 0] == 0.0 and out[2, 0] == 1.0


# --- target_fill_value (regression) ------------------------------------------------------------


def test_target_fill_value_still_gives_nan():
    reader = _stub_reader({"name": "t", "target_fill_value": 0.0}, [2.0, 4.0], [2.0, 0.5])
    target = np.array([[0.0, 5.0], [3.0, 0.0]], dtype=np.float32)

    out = reader.normalize_target_channels(target)

    assert np.isnan(out[0, 0]) and np.isnan(out[1, 1])
    np.testing.assert_allclose(out[0, 1], 2.0)
    np.testing.assert_allclose(out[1, 0], 0.5)


# --- real data ---------------------------------------------------------------------------------

needs_archive = pytest.mark.skipif(
    not (DATA_ROOT / "ocean_node").is_dir() or not (DATA_ROOT / "ocean_elem").is_dir(),
    reason="AWI-CM3 ocean data not available",
)


def _real_reader(stream: str, stream_info_extra: dict):
    info = {
        "name": f"FESOM_{stream.upper()}",
        "type": "fesom",
        "filenames": [f"ocean_{stream}"],
        "source": ["salt_450"] if stream == "node" else ["u_450"],
        "target": ["salt_450"] if stream == "node" else ["u_450"],
        **stream_info_extra,
    }
    tw = TimeWindowHandler(
        np.datetime64("2001-01-01T00:00"),
        np.datetime64("2001-01-31T00:00"),
        np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    reader = fesom.DataReaderFesom(
        tw_handler=tw, filename=DATA_ROOT / f"ocean_{stream}", stream_info=info, stage="train"
    )
    reader._lazy_init()
    return reader


@functools.cache
def _plain_reader(stream: str):
    return _real_reader(stream, {})


@needs_archive
@pytest.mark.parametrize(
    ("stream", "channel"),
    [("node", "salt_450"), ("elem", "u_450")],
)
def test_reader_uses_the_statistics_from_the_normalization_stats_file(tmp_path, stream, channel):
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps({"_meta": {"years": [2000, 2001]}, channel: {"mean": 34.5, "std": 0.75}})
    )

    plain = _plain_reader(stream)
    forced = _real_reader(stream, {"normalization_stats": str(stats)})

    # the archive statistics differ (they include the dry zeros), the file's win, for sources
    # and targets alike
    assert plain.src_mean_vec[0] != 34.5
    np.testing.assert_allclose(forced.src_mean_vec, [34.5])
    np.testing.assert_allclose(forced.src_stdev_vec, [0.75])
    np.testing.assert_allclose(forced.trg_mean_vec, [34.5])
    np.testing.assert_allclose(forced.trg_stdev_vec, [0.75])
    # without the option nothing changes
    assert plain.src_stdev_vec[0] != 0.75


@needs_archive
@pytest.mark.parametrize(
    ("stream", "channel"),
    [("node", "salt_450"), ("elem", "u_450")],
)
def test_reader_with_the_committed_wet_statistics(stream, channel):
    stats = REPO_ROOT / "config" / "streams" / "fesom" / f"wet_stats_{stream}.json"
    if not stats.is_file():
        pytest.skip(f"{stats.name} has not been generated yet")
    expected = json.loads(stats.read_text())[channel]

    reader = _real_reader(stream, {"normalization_stats": str(stats)})

    np.testing.assert_allclose(reader.src_mean_vec, [expected["mean"]])
    np.testing.assert_allclose(reader.src_stdev_vec, [expected["std"]])
    np.testing.assert_allclose(reader.trg_mean_vec, [expected["mean"]])
    np.testing.assert_allclose(reader.trg_stdev_vec, [expected["std"]])
