"""Trimming a rollout zip back to the steps its rollout_state.pt records."""

import zipfile

import pytest

from weathergen.utils.trim_rollout_store import trim_rollout_store


def _store(path, leads):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("zarr.json", "{}")
        z.writestr("0/zarr.json", "{}")
        z.writestr("0/TEMP/zarr.json", "{}")
        for lead in leads:
            z.writestr(f"0/TEMP/{lead}/zarr.json", "{}")
            z.writestr(f"0/TEMP/{lead}/prediction/data/c/0/0/0", f"lead {lead}")


def test_leads_after_the_saved_state_are_removed(tmp_path):
    path = tmp_path / "validation_chkpt00000_rank0000.zip"
    _store(path, range(0, 41))
    assert trim_rollout_store(path, 30) == 10
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert z.read("0/TEMP/30/prediction/data/c/0/0/0") == b"lead 30"
    assert "0/TEMP/31/zarr.json" not in names
    assert "zarr.json" in names and "0/TEMP/zarr.json" in names
    assert len(names) == 3 + 31 * 2
    assert not list(tmp_path.glob("*.tmp"))


def test_a_consistent_store_is_left_untouched(tmp_path):
    path = tmp_path / "validation_chkpt00000_rank0000.zip"
    _store(path, range(0, 31))
    before = path.stat().st_mtime_ns
    assert trim_rollout_store(path, 30) == 0
    assert path.stat().st_mtime_ns == before


def test_an_unreadable_store_is_refused(tmp_path):
    path = tmp_path / "validation_chkpt00000_rank0000.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        trim_rollout_store(path, 30)
