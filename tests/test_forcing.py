"""Tests for the pure forcing helpers in weathergen.model.forcing."""

import numpy as np
import pytest
import torch

from weathergen.model.forcing import (
    ForcingUnavailableError,
    build_forcing_cell_index,
    count_snapshots,
    forcing_field_from_window,
    reduce_forcing_time,
    scatter_to_cells,
)

# four points: two share a HEALPix cell near the north pole, two near the south pole
COORDS = np.array([[80.0, 10.0], [80.0, 10.5], [-80.0, 10.0], [-80.0, 10.5]], dtype=np.float32)


def _window(num_snapshots=4, num_points=4, num_vars=2):
    """Time-major window; value = 100 * snapshot + 10 * point + var."""
    s, p, v = np.meshgrid(
        np.arange(num_snapshots), np.arange(num_points), np.arange(num_vars), indexing="ij"
    )
    values = (100 * s + 10 * p + v).astype(np.float32).reshape(-1, num_vars)
    times = np.repeat(
        np.datetime64("2001-01-01T00") + np.arange(num_snapshots) * np.timedelta64(6, "h"),
        num_points,
    )
    coords = np.tile(COORDS[:num_points], (num_snapshots, 1))
    return values, times, coords


def test_count_snapshots():
    _, times, _ = _window()
    assert count_snapshots(times) == 4
    assert count_snapshots(np.array([], dtype="datetime64[ns]")) == 0


def test_reduce_mean_last_stack():
    values, _, _ = _window()
    mean = reduce_forcing_time(values, 4, "mean")
    last = reduce_forcing_time(values, 4, "last")
    stack = reduce_forcing_time(values, 4, "stack")
    assert mean.shape == (4, 2) and last.shape == (4, 2) and stack.shape == (4, 8)
    # point 1, var 0: snapshots give 10, 110, 210, 310
    assert mean[1, 0] == pytest.approx(160.0)
    assert last[1, 0] == pytest.approx(310.0)
    np.testing.assert_allclose(stack[1], [10, 11, 110, 111, 210, 211, 310, 311])
    assert mean.dtype == values.dtype


def test_mean_ignores_nan_snapshots():
    values, _, _ = _window()
    values[0, 0] = np.nan  # snapshot 0, point 0, var 0
    mean = reduce_forcing_time(values, 4, "mean")
    assert mean[0, 0] == pytest.approx(200.0)  # mean of 100, 200, 300


def test_unknown_mode_and_ragged_rows():
    values, _, _ = _window()
    with pytest.raises(ValueError, match="time_reduce"):
        reduce_forcing_time(values, 4, "median")
    with pytest.raises(ForcingUnavailableError):
        reduce_forcing_time(values[:-1], 4, "mean")


def test_scatter_to_cells_mean_nan_and_validity():
    values = torch.tensor([[[1.0], [3.0], [float("nan")], [5.0]]])
    cell_idx = torch.tensor([0, 0, 1, 1])
    cell_values, cell_valid = scatter_to_cells(values, cell_idx, 3)
    np.testing.assert_allclose(cell_values[0, :, 0].numpy(), [2.0, 5.0, 0.0])
    np.testing.assert_allclose(cell_valid[0, :, 0].numpy(), [1.0, 0.5, 0.0])


def test_field_from_window_builds_and_reuses_cell_index():
    values, times, coords = _window()
    field, cell_idx = forcing_field_from_window(values, times, coords, None, 0, 12, "mean")
    assert field.shape == (12, 4)  # 12 cells, [2 values | 2 validity]
    expected_idx = build_forcing_cell_index(COORDS[:, 0], COORDS[:, 1], 0)
    assert torch.equal(cell_idx, expected_idx)
    north = int(cell_idx[0])
    assert int(cell_idx[1]) == north
    # mean over time of points 0 and 1, var 0: (150 + 160) / 2
    assert field[north, 0] == pytest.approx(155.0)
    assert field[north, 2] == pytest.approx(1.0)
    # the returned index is reusable without coordinates
    field2, _ = forcing_field_from_window(values, times, None, cell_idx, 0, 12, "mean")
    np.testing.assert_allclose(field, field2)


def test_stack_field_width():
    values, times, coords = _window()
    field, _ = forcing_field_from_window(
        values, times, coords, None, 0, 12, "stack", expected_snapshots=4
    )
    assert field.shape == (12, 16)  # 2 vars x 4 snapshots, values and validity


def test_stack_with_incomplete_window_is_unavailable():
    values, times, coords = _window(num_snapshots=3)
    with pytest.raises(ForcingUnavailableError, match="3 snapshots, expected 4"):
        forcing_field_from_window(
            values, times, coords, None, 0, 12, "stack", expected_snapshots=4
        )


def test_empty_window_is_unavailable():
    empty = np.zeros((0, 2), dtype=np.float32)
    with pytest.raises(ForcingUnavailableError, match="no forcing data"):
        forcing_field_from_window(
            empty, np.array([], dtype="datetime64[ns]"), np.zeros((0, 2)), None, 0, 12, "mean"
        )


def test_all_nan_point_gives_finite_field_with_zero_validity():
    values, times, coords = _window()
    values[0::4, 0] = np.nan  # point 0, var 0, every snapshot
    values[1::4, 0] = np.nan  # point 1, var 0, every snapshot -> whole north cell missing
    field, cell_idx = forcing_field_from_window(values, times, coords, None, 0, 12, "mean")
    north = int(cell_idx[0])
    assert np.isfinite(field).all()
    assert field[north, 0] == 0.0 and field[north, 2] == 0.0
    assert field[north, 3] == pytest.approx(1.0)  # var 1 is still fully valid


def test_grid_size_mismatch_with_cached_index():
    values, times, coords = _window()
    with pytest.raises(ForcingUnavailableError, match="cell index"):
        forcing_field_from_window(values, times, coords, torch.tensor([0, 1, 2]), 0, 12, "mean")
