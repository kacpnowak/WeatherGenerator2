"""Tests for forcing-stream handling in MultiStreamDataSampler (no data access)."""

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


def test_fesom_streams_resolve_to_fesom_reader():
    # develop moved the fesom reader into readers_extra; the sampler must find it there
    cls = MultiStreamDataSampler._get_dataset_class(None, {"type": "fesom"}, "FESOM_NODES")
    assert cls.__name__ == "DataReaderFesom"
    assert cls.__module__ == "weathergen.readers_extra.data_reader_fesom"


def test_latent_output_uses_chunk_relative_step_index():
    import types

    import torch

    from weathergen.utils.validation_io import get_latent_output

    requested = []

    def get_latent_prediction(i):
        requested.append(i)
        return {"head": torch.zeros(1, 3)}

    sample = types.SimpleNamespace(streams_data={"s": types.SimpleNamespace(sample_idx=0)})
    batch = types.SimpleNamespace(
        get_output_idxs=lambda: [1, 2, 3, 4],
        get_source_samples=lambda: types.SimpleNamespace(get_samples=lambda: [sample]),
    )
    model_output = types.SimpleNamespace(get_latent_prediction=get_latent_prediction)

    latents = get_latent_output(batch, model_output, timestep_idxs=[3, 4], fstep_offset=3)

    assert requested == [0, 1]
    assert len(latents) == 2 and latents[0][0]["head"].shape == (3,)


import types

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import ReaderData
from weathergen.model.forcing import ForcingUnavailableError

_COORDS = np.array([[80.0, 10.0], [80.0, 10.5], [-80.0, 10.0], [-80.0, 10.5]], dtype=np.float32)


class _FakeReader:
    """Fixed 4-point grid; every value in window `idx` equals `idx`."""

    source_channels = ["a", "b"]

    def __init__(self, tw_handler=None, stream_info=None, stage=None, filename=None):
        self.snapshots = 4
        self.empty_windows = set()
        self.short_windows = set()
        self.requested = []
        self.forcing_cell_idx = None
        self.forcing_stream_name = "F"
        self.forcing_time_reduce = "mean"
        self.forcing_expected_snapshots = None

    def get_source(self, idx):
        idx = int(idx)
        self.requested.append(idx)
        if idx in self.empty_windows:
            return ReaderData.empty(num_data_fields=2, num_geo_fields=0)
        n = self.snapshots - (1 if idx in self.short_windows else 0)
        p = len(_COORDS)
        times = np.repeat(
            np.datetime64("2001-01-01T00") + np.arange(n) * np.timedelta64(6, "h"), p
        )
        return ReaderData(
            coords=np.tile(_COORDS, (n, 1)),
            geoinfos=np.zeros((n * p, 0), dtype=np.float32),
            data=np.full((n * p, 2), float(idx), dtype=np.float32),
            datetimes=times,
        )

    def normalize_source_channels(self, data):
        return data


def _sampler(time_step="1D", step="1D", window="1D"):
    stub = types.SimpleNamespace(
        time_step=np.timedelta64(pd_td(time_step)),
        step_timedelta=np.timedelta64(pd_td(step)),
        len_timedelta=np.timedelta64(pd_td(window)),
        healpix_level=0,
        num_healpix_cells=12,
        time_window_handler=types.SimpleNamespace(window=lambda idx: f"window {idx}"),
        _stage="train",
    )
    stub._forcing_snapshots_per_window = lambda name, info: (
        MultiStreamDataSampler._forcing_snapshots_per_window(stub, name, info)
    )
    return stub


def pd_td(text):
    import pandas as pd

    return pd.to_timedelta(text).to_timedelta64()


class _Batch:
    def __init__(self, num_steps):
        self._samples = types.SimpleNamespace(forcing=[[] for _ in range(num_steps)])

    def get_source_samples(self):
        return self._samples


def test_forcing_is_read_at_each_steps_target_window():
    reader, batch = _FakeReader(), _Batch(3)
    MultiStreamDataSampler._build_forcing_data(_sampler(), batch, reader, 10, 3)
    # same window index the targets of absolute step i use: base + i * time_step / step
    assert reader.requested == [10, 11, 12]
    forcing = batch.get_source_samples().forcing
    assert [f.shape for f in forcing] == [(12, 4)] * 3
    north = int(reader.forcing_cell_idx[0])
    assert [float(f[north, 0]) for f in forcing] == [10.0, 11.0, 12.0]
    assert len(reader.forcing_cell_idx) == 4  # built lazily and cached on the reader


def test_forcing_window_index_follows_time_step():
    reader, batch = _FakeReader(), _Batch(3)
    MultiStreamDataSampler._build_forcing_data(
        _sampler(time_step="1D", step="6h"), batch, reader, 10, 3
    )
    assert reader.requested == [10, 14, 18]


def test_forcing_step_range_builds_only_that_chunk():
    reader, batch = _FakeReader(), _Batch(3)
    MultiStreamDataSampler._build_forcing_data(
        _sampler(), batch, reader, 10, 3, step_range=(1, 3)
    )
    assert reader.requested == [11, 12]
    assert len(batch.get_source_samples().forcing[0]) == 0


def test_missing_forcing_raises_with_stream_and_window():
    reader, batch = _FakeReader(), _Batch(3)
    reader.empty_windows = {11}
    with pytest.raises(ForcingUnavailableError, match="'F'.*window 11.*step 1"):
        MultiStreamDataSampler._build_forcing_data(_sampler(), batch, reader, 10, 3)


def test_stack_builds_wide_field_and_rejects_short_window():
    reader, batch = _FakeReader(), _Batch(2)
    reader.forcing_time_reduce = "stack"
    reader.forcing_expected_snapshots = 4
    MultiStreamDataSampler._build_forcing_data(_sampler(), batch, reader, 10, 2)
    assert batch.get_source_samples().forcing[0].shape == (12, 16)

    reader.short_windows = {10}
    with pytest.raises(ForcingUnavailableError, match="3 snapshots, expected 4"):
        MultiStreamDataSampler._build_forcing_data(_sampler(), _Batch(2), reader, 10, 2)


def test_snapshots_per_window():
    stub = _sampler(window="1D")
    f = MultiStreamDataSampler._forcing_snapshots_per_window
    assert f(stub, "F", {"frequency": "06:00:00"}) == 4
    assert f(stub, "F", {"frequency": np.timedelta64(12, "h")}) == 2


def test_snapshots_per_window_rejects_non_divisible_frequency():
    stub = _sampler(window="1D")
    f = MultiStreamDataSampler._forcing_snapshots_per_window
    with pytest.raises(ValueError, match="does not divide"):
        f(stub, "F", {"frequency": "05:00:00"})
    with pytest.raises(ValueError, match="needs 'frequency'"):
        f(stub, "F", {})


def _init_forcing(streams, tmp_path):
    stub = _sampler()
    seen = []

    def get_dataset_class(info, name):
        seen.append(dict(info))
        return _FakeReader

    stub._get_dataset_class = get_dataset_class
    for info in streams.values():
        info.setdefault("filenames", [str(tmp_path)])
    cf = types.SimpleNamespace(streams=streams, data_paths=[])
    out = MultiStreamDataSampler._init_forcing_streams(stub, cf)
    return out, seen


def test_reader_key_selects_reader_and_registers_channels(tmp_path):
    streams = {
        "OCEAN": {"type": "fesom"},
        "IFS_ATMO": {"type": "forcing", "reader": "fesom", "injection": {"mode": "additive"}},
    }
    out, seen = _init_forcing(streams, tmp_path)
    assert list(out) == ["IFS_ATMO"]
    assert seen == [{"type": "fesom"}]
    assert streams["IFS_ATMO"]["forcing_num_vars"] == 2
    assert list(streams["IFS_ATMO"]["train_source_channels"]) == ["a", "b"]
    reader = out["IFS_ATMO"].readers[0]
    assert reader.forcing_time_reduce == "mean" and reader.forcing_expected_snapshots is None
    assert reader.forcing_cell_idx is None and reader.forcing_stream_name == "IFS_ATMO"


def test_forcing_stream_without_reader_key_uses_its_own_type(tmp_path):
    streams = {"SST": {"type": "forcing"}}
    _, seen = _init_forcing(streams, tmp_path)
    assert seen[0]["type"] == "forcing"


def test_stack_multiplies_forcing_num_vars(tmp_path):
    streams = {
        "IFS_ATMO": {
            "type": "forcing",
            "reader": "fesom",
            "frequency": "06:00:00",
            "injection": {"mode": "additive", "time_reduce": "stack"},
        }
    }
    out, _ = _init_forcing(streams, tmp_path)
    assert streams["IFS_ATMO"]["forcing_num_vars"] == 8
    assert out["IFS_ATMO"].readers[0].forcing_expected_snapshots == 4


def test_second_forcing_stream_is_rejected(tmp_path):
    streams = {"A": {"type": "forcing"}, "B": {"type": "forcing"}}
    with pytest.raises(ValueError, match="Only one forcing stream"):
        _init_forcing(streams, tmp_path)


def test_unknown_time_reduce_is_rejected(tmp_path):
    streams = {"A": {"type": "forcing", "injection": {"time_reduce": "median"}}}
    with pytest.raises(ValueError, match="time_reduce"):
        _init_forcing(streams, tmp_path)


def test_iter_skips_sample_without_forcing():
    good = types.SimpleNamespace(
        sources_empty=lambda: False, is_nan=lambda: False, targets_empty=lambda: False
    )
    tried = []

    def get_batch(idx, num_forecast_steps):
        tried.append(int(idx))
        if idx == 0:
            raise ForcingUnavailableError("forcing stream 'F': no data")
        return good

    stub = types.SimpleNamespace(
        worker_workset=lambda: (0, 2),
        len=2,
        reset=lambda: (np.array([0, 1, 2]), np.array([1, 1])),
        batch_size=1,
        mode_cfg={"training_mode": ["forecast"]},
        _lazy_targets=False,
        _get_batch=get_batch,
    )
    batches = list(MultiStreamDataSampler.__iter__(stub))
    assert tried == [0, 1, 2]
    assert batches == [good, good]
