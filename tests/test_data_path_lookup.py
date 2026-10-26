"""Stream and forcing files resolve to the data path that has them (tmp dirs, no real data)."""

import pytest

from weathergen.datasets.multi_stream_data_sampler import resolve_data_file


def test_picks_the_data_path_that_has_the_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    (b / "ocean_node").mkdir(parents=True)
    assert resolve_data_file("ocean_node", [a, b], "x") == b / "ocean_node"


def test_first_existing_wins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "ocean_node").mkdir(parents=True)
    (b / "ocean_node").mkdir(parents=True)
    assert resolve_data_file("ocean_node", [a, b], "x") == a / "ocean_node"


def test_existing_full_path_is_used_as_is(tmp_path):
    a = tmp_path / "a"
    (a / "ocean_node").mkdir(parents=True)
    full = tmp_path / "elsewhere" / "ocean_node"
    full.mkdir(parents=True)
    assert resolve_data_file(full, [a], "x") == full


def test_missing_everywhere_raises_with_candidates(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with pytest.raises(FileNotFoundError, match="forcing stream 'IFS_ATMO'") as err:
        resolve_data_file("atmos_all", [a, b], "forcing stream 'IFS_ATMO'")
    assert str(a / "atmos_all") in str(err.value)
    assert str(b / "atmos_all") in str(err.value)
