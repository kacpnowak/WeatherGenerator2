"""Tests for forcing-stream handling in MultiStreamDataSampler (no data access)."""

import types

import numpy as np
import pytest

from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.data_reader_base import ReaderData, TimeWindowHandler
from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    forcing_time_window_handler,
)
from weathergen.model.forcing import ForcingUnavailableError


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


_COORDS = np.array([[80.0, 10.0], [80.0, 10.5], [-80.0, 10.0], [-80.0, 10.5]], dtype=np.float32)


class _FakeReader:
    """Fixed 4-point grid; every value in window `idx` equals `idx`."""

    source_channels = ["a", "b"]

    def __init__(self, tw_handler=None, stream_info=None, stage=None, filename=None):
        self.tw_handler = tw_handler
        self.snapshots = 4
        self.empty_windows = set()
        self.short_windows = set()
        self.requested = []
        self.forcing_cell_idx = None
        self.forcing_stream_name = "F"
        self.forcing_time_reduce = "mean"
        self.forcing_expected_snapshots = None
        self.forcing_tw_handler = types.SimpleNamespace(window=lambda idx: f"window {idx}")

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
        time_window_handler=types.SimpleNamespace(window=lambda idx: f"sampler window {idx}"),
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
    # the message names the window the forcing was read over, not the sampler's window
    with pytest.raises(ForcingUnavailableError, match="'F', window 11.*step 1"):
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


def _init_forcing(streams, tmp_path, tw_handler=None):
    stub = _sampler()
    if tw_handler is not None:
        stub.time_window_handler = tw_handler
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


def _eager_sampler(output_offset):
    """Stub `self` that drives `_get_batch` down to its forcing call, without data streams."""
    stub = _sampler()
    stub.output_offset = output_offset
    stub.mode_cfg = {"training_mode": ["masking"], "model_input": {"random": {}}}
    no_masks = types.SimpleNamespace(masks=[])
    stub.streams_datasets = {"S": types.SimpleNamespace(info={}, readers=[])}
    stub._get_source_target_masks = lambda mode: ({"S": (no_masks, no_masks, [])}, 1, 1)
    stub._get_output_length = lambda n: MultiStreamDataSampler._get_output_length(stub, n)
    stub._get_data_windows = lambda *args: (None, None)
    stub.tokenizer = types.SimpleNamespace(get_tokens_windows=lambda *args: None)
    stub.condition_datasets = {}
    reader = _FakeReader()
    stub.forcing_datasets = {"F": types.SimpleNamespace(readers=[reader])}
    stub._build_forcing_data = lambda *args, **kwargs: (
        MultiStreamDataSampler._build_forcing_data(stub, *args, **kwargs)
    )
    stub._preprocess_model_batch = lambda batch, *args: batch
    return stub, reader


@pytest.mark.parametrize("output_offset, requested", [(1, [11, 12]), (0, [10, 11])])
def test_eager_batch_reads_forcing_only_for_the_forecast_steps(output_offset, requested):
    stub, reader = _eager_sampler(output_offset)
    batch = MultiStreamDataSampler._get_batch(stub, 10, 2)

    # the model consumes the forcing of the forecast steps output_idxs only
    assert reader.requested == requested
    forcing = batch.get_source_samples().forcing
    assert [len(forcing[i]) > 0 for i in range(len(forcing))] == [
        i in batch.get_output_idxs() for i in range(len(forcing))
    ]


def test_eager_batch_does_not_need_the_source_windows_forcing():
    stub, reader = _eager_sampler(1)
    reader.empty_windows = {10}  # only the source window lacks forcing
    MultiStreamDataSampler._get_batch(stub, 10, 2)
    assert reader.requested == [11, 12]


def test_release_target_chunk_frees_the_chunks_forcing():
    stub = types.SimpleNamespace(num_healpix_cells=12, streams_datasets={})
    batch = ModelBatch([], 0, 0, 1, 5)
    forcing = batch.get_source_samples().forcing
    forcing[:] = [np.full((12, 4), float(i)) for i in range(5)]

    MultiStreamDataSampler._release_target_chunk(stub, batch, (1, 3))

    assert [len(f) for f in forcing] == [12, 0, 0, 12, 12]


def _iter_stub(get_batch, perms):
    return types.SimpleNamespace(
        worker_workset=lambda: (0, 2),
        len=2,
        reset=lambda: (np.array(perms), np.array([1, 1])),
        batch_size=1,
        mode_cfg={"training_mode": ["forecast"]},
        _lazy_targets=False,
        _get_batch=get_batch,
    )


def test_iter_stops_when_no_sample_has_forcing():
    tried = []

    def get_batch(idx, num_forecast_steps):
        tried.append(int(idx))
        raise ForcingUnavailableError(f"forcing stream 'F': window {idx}: no data")

    stub = _iter_stub(get_batch, [0, 1, 2])
    with pytest.raises(RuntimeError, match="window 2: no data"):
        list(MultiStreamDataSampler.__iter__(stub))
    assert tried == [0, 1, 2]


def test_iter_forcing_skip_count_resets_after_a_batch():
    good = types.SimpleNamespace(
        sources_empty=lambda: False, is_nan=lambda: False, targets_empty=lambda: False
    )
    tried = []

    def get_batch(idx, num_forecast_steps):
        tried.append(int(idx))
        if idx != 1:
            raise ForcingUnavailableError("forcing stream 'F': no data")
        return good

    # 0 skipped, 1 yielded; 2 and 0 skipped, 1 yielded: three skips, but never three in a row
    stub = _iter_stub(get_batch, [0, 1, 2])
    batches = list(MultiStreamDataSampler.__iter__(stub))
    assert tried == [0, 1, 2, 0, 1]
    assert batches == [good, good]


_DAILY = TimeWindowHandler(
    np.datetime64("2001-01-01T00:00"),
    np.datetime64("2001-02-01T00:00"),
    np.timedelta64(1, "D"),
    np.timedelta64(1, "D"),
)


def test_forcing_window_is_the_samplers_without_window_shift():
    assert forcing_time_window_handler(_DAILY, {}) is _DAILY


@pytest.mark.parametrize("shift", ["06:00:00", np.timedelta64(6, "h")])
def test_window_shift_moves_every_forcing_window(shift):
    shifted = forcing_time_window_handler(_DAILY, {"window_shift": shift})
    # ocean day 2001-01-04 is forced by IFS at 06, 12, 18 and 00 of the next day
    window = shifted.window(3)
    assert window.start == np.datetime64("2001-01-04T06:00")
    assert window.end == np.datetime64("2001-01-05T06:00")
    assert shifted.t_window_len == _DAILY.t_window_len
    assert shifted.t_window_step == _DAILY.t_window_step
    assert _DAILY.window(3).start == np.datetime64("2001-01-04T00:00")  # sampler unchanged


def test_forcing_reader_is_built_with_the_shifted_window(tmp_path):
    streams = {"IFS_ATMO": {"type": "forcing", "reader": "fesom", "window_shift": "06:00:00"}}
    out, _ = _init_forcing(streams, tmp_path, tw_handler=_DAILY)
    reader = out["IFS_ATMO"].readers[0]
    assert reader.tw_handler is reader.forcing_tw_handler
    assert reader.tw_handler.window(0).start == np.datetime64("2001-01-01T06:00")


def test_forcing_reader_uses_the_samplers_window_without_shift(tmp_path):
    streams = {"IFS_ATMO": {"type": "forcing", "reader": "fesom"}}
    out, _ = _init_forcing(streams, tmp_path, tw_handler=_DAILY)
    reader = out["IFS_ATMO"].readers[0]
    assert reader.tw_handler is _DAILY and reader.forcing_tw_handler is _DAILY
