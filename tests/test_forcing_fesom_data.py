"""Real-data check of the IFS forcing path. Skipped where the AWI-CM3 data is absent."""

from pathlib import Path

import numpy as np
import pytest

from weathergen.common.config import load_streams
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.forcing import count_snapshots, forcing_field_from_window
from weathergen.readers_extra.registry import get_extra_reader

DATA_ROOT = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
STREAMS = Path(__file__).resolve().parents[1] / "config" / "streams" / "fesom_forced"
HEALPIX_LEVEL, NUM_CELLS, NUM_POINTS = 5, 12 * 4**5, 76800

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "atmos_all").is_dir(), reason="AWI-CM3 IFS data not available"
)


@pytest.fixture(scope="module")
def reader_and_windows():
    info = load_streams(STREAMS)["IFS_ATMO"]
    if "name" not in info:
        info["name"] = "IFS_ATMO"
    windows = TimeWindowHandler(
        np.datetime64("2001-01-01T00:00"),
        np.datetime64("2001-01-31T00:00"),
        np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    reader = get_extra_reader(info["reader"])(
        tw_handler=windows, filename=DATA_ROOT / "atmos_all", stream_info=info, stage="train"
    )
    return reader, windows


def test_selected_channels(reader_and_windows):
    reader, _ = reader_and_windows
    assert sorted(reader.source_channels) == sorted(
        ["10u", "10v", "2d", "2t", "cp", "lsp", "msl", "tsr", "tsrc"]
    )


@pytest.mark.parametrize("idx", [0, 3])
def test_daily_window_holds_four_snapshots_inside_the_window(reader_and_windows, idx):
    reader, windows = reader_and_windows
    rdata = reader.get_source(idx)
    assert count_snapshots(rdata.datetimes) == 4
    assert rdata.data.shape == (4 * NUM_POINTS, 9)
    window = windows.window(idx)
    times = np.asarray(rdata.datetimes)
    assert (times >= window.start).all() and (times < window.end).all()


def test_field_covers_every_latent_cell(reader_and_windows):
    reader, _ = reader_and_windows
    rdata = reader.get_source(0)
    values = np.asarray(reader.normalize_source_channels(rdata.data))
    field, cell_idx = forcing_field_from_window(
        values, rdata.datetimes, rdata.coords, None, HEALPIX_LEVEL, NUM_CELLS, "mean"
    )
    assert field.shape == (NUM_CELLS, 18)
    assert np.isfinite(field).all()
    counts = np.bincount(cell_idx.numpy(), minlength=NUM_CELLS)
    assert counts.min() >= 1 and counts.sum() == NUM_POINTS
    assert (field[:, 9:] == 1.0).all()  # every cell fully valid
    stacked, _ = forcing_field_from_window(
        values, rdata.datetimes, rdata.coords, cell_idx, HEALPIX_LEVEL, NUM_CELLS, "stack", 4
    )
    assert stacked.shape == (NUM_CELLS, 72)
