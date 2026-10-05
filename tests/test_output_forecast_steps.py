"""Forecast-step layout of the written output store (step-0 source item, forecast offset)."""

import numpy as np
import pytest

from weathergen.common.io import (
    IOReaderData,
    OutputBatchData,
    TimeRange,
    zarrio_reader,
    zarrio_writer,
)

CHANNELS = ["a", "b"]
NUM_POINTS = 3
T0 = np.datetime64("2020-01-01T00:00", "ns")


def _batch_data(forecast_offset: int, steps: list[int]) -> OutputBatchData:
    """One sample, one stream, as write_output builds it for the forecast steps `steps`."""
    source = IOReaderData(
        coords=np.zeros((NUM_POINTS, 2), dtype=np.float32),
        geoinfos=np.zeros((NUM_POINTS, 0), dtype=np.float32),
        data=np.ones((NUM_POINTS, len(CHANNELS)), dtype=np.float32),
        datetimes=np.full(NUM_POINTS, T0),
    )
    targets = [[np.full((NUM_POINTS, len(CHANNELS)), s, dtype=np.float32)] for s in steps]
    return OutputBatchData(
        sources=[[source]],
        source_intervals=[TimeRange(T0, T0 + np.timedelta64(1, "D"))],
        targets=targets,
        predictions=[[t[0][None]] for t in targets],
        targets_coords=[[np.zeros((NUM_POINTS, 2), dtype=np.float32)] for _ in steps],
        targets_times=[[np.full(NUM_POINTS, T0 + np.timedelta64(s, "D"))] for s in steps],
        targets_lens=[[[NUM_POINTS]] for _ in steps],
        streams={"S": 0},
        target_channels=[CHANNELS],
        source_channels=[CHANNELS],
        geoinfo_channels=[[]],
        forecast_offset=forecast_offset,
        forecast_steps_override=steps,
    )


def _items(data):
    return {item.key.forecast_step: item for item in data.items()}


@pytest.mark.parametrize("steps", [[1, 2], [1, 2, 3]], ids=["unchunked", "first_chunk"])
def test_offset_one_store_starts_with_the_step_zero_source_item(steps):
    items = _items(_batch_data(1, steps))

    assert sorted(items) == [0, *steps]
    assert items[0].source is not None and items[0].target is None
    for step in steps:
        assert items[step].source is None
        assert float(items[step].target.data[0, 0]) == step  # each step reads its own data


def test_later_chunk_has_no_step_zero_item():
    items = _items(_batch_data(1, [3, 4]))
    assert sorted(items) == [3, 4]


def test_offset_zero_step_zero_holds_source_and_target():
    items = _items(_batch_data(0, [0, 1]))
    assert sorted(items) == [0, 1]
    assert items[0].source is not None and items[0].target is not None


def test_store_of_an_offset_one_run_infers_offset_one(tmp_path):
    # the default store type (zip) lists the forecast steps in the order they were written
    path = tmp_path / "out.zip"
    with zarrio_writer(path) as zio:
        for item in _batch_data(1, [1, 2]).items():
            zio.write_zarr(item)

    with zarrio_reader(path) as zio:
        assert zio.forecast_offset == 1
        assert [int(s) for s in zio.forecast_steps] == [1, 2]
        assert zio.get_data(0, "S", 0).source is not None
