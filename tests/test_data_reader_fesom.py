# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the FESOM reader's channel normalization."""

import numpy as np
import pytest

from weathergen.readers_extra.data_reader_fesom import DataReaderFesom


def _reader(stream_info: dict) -> DataReaderFesom:
    """A reader with only the state that (de)normalization touches.

    The full constructor needs Zarr stores on disk; normalization is independent of
    them, so the lazy-init flag is set and the channel statistics are injected.
    """
    reader = DataReaderFesom.__new__(DataReaderFesom)
    reader._initialized = True
    reader._stream_info = stream_info
    reader.source_idx = [0, 1]
    reader.target_idx = [0, 1]
    reader.source_mean = np.array([0.0, 10.0], dtype=np.float32)
    reader.source_stdev = np.array([1.0, 2.0], dtype=np.float32)
    reader.target_mean = np.array([0.0, 10.0], dtype=np.float32)
    reader.target_stdev = np.array([1.0, 2.0], dtype=np.float32)
    return reader


def test_target_fill_value_becomes_nan():
    # FESOM stores below-bathymetry nodes as 0.0; those must not be trained on.
    reader = _reader({"name": "FESOM_NODES", "target_fill_value": 0.0})
    target = np.array([[1.0, 12.0], [0.0, 0.0]], dtype=np.float32)

    out = reader.normalize_target_channels(target)

    assert out[0].tolist() == [1.0, 1.0]
    assert np.isnan(out[1]).all()


def test_targets_are_untouched_without_target_fill_value():
    # IFS_ATMO and friends have genuine zeros, so the conversion must be opt-in.
    reader = _reader({"name": "IFS_ATMO"})
    target = np.array([[0.0, 0.0]], dtype=np.float32)

    out = reader.normalize_target_channels(target)

    assert not np.isnan(out).any()
    assert out[0].tolist() == [0.0, -5.0]


def test_sources_keep_the_fill_value_as_a_normal_number():
    # The fill is masked out of the loss only; the model still gets it as input.
    reader = _reader({"name": "FESOM_NODES", "target_fill_value": 0.0})
    source = np.array([[0.0, 0.0]], dtype=np.float32)

    out = reader.normalize_source_channels(source)

    assert not np.isnan(out).any()
    assert out[0].tolist() == [0.0, -5.0]


@pytest.mark.parametrize("fill", [-9999.0, 0.0])
def test_any_fill_value_is_honoured(fill):
    reader = _reader({"name": "FESOM_NODES", "target_fill_value": fill})
    target = np.array([[fill, 12.0]], dtype=np.float32)

    out = reader.normalize_target_channels(target)

    assert np.isnan(out[0, 0])
    assert out[0, 1] == 1.0
