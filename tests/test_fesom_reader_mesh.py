"""FESOM reader with the mesh-based dry mask and bathymetry geoinfo (real data)."""

from pathlib import Path

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.fesom_mesh import bathymetry, dry_mask, read_mesh_entity
from weathergen.readers_extra.registry import get_extra_reader

ROOT = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
pytestmark = pytest.mark.skipif(not (ROOT / "ocean_node").is_dir(), reason="AWI-CM3 data absent")
CHANNELS = ["temp_2", "temp_450", "salt_95", "salt_450"]


def _reader(days=1, **keys):
    info = {"name": "T", "type": "fesom", "stream_id": 2150, "filenames": ["ocean_node"],
            "source": CHANNELS, "target": CHANNELS, "loss_weight": 1.0, **keys}
    tw = TimeWindowHandler(np.datetime64("2100-01-01T00:00"), np.datetime64("2100-01-20T00:00"),
                           np.timedelta64(days, "D"), np.timedelta64(days, "D"))
    return get_extra_reader("fesom")(tw_handler=tw, filename=ROOT / "ocean_node",
                                     stream_info=info, stage="train")


MESH_KEYS = {"mesh_path": MESH, "wet_mask_from_mesh": True, "bathymetry_geoinfo": True}


def test_geoinfo_size_is_known_before_any_read():
    assert _reader(**MESH_KEYS).get_geoinfo_size() == 1
    assert _reader().get_geoinfo_size() == 0


def test_mesh_keys_need_a_mesh_path():
    with pytest.raises(ValueError, match="mesh_path"):
        _reader(wet_mask_from_mesh=True)


@pytest.mark.parametrize("days", [1, 2])
def test_dry_targets_are_nan_and_wet_targets_untouched(days):
    plain, masked = _reader(days), _reader(days, **MESH_KEYS)
    raw, got = plain.get_target(1), masked.get_target(1)
    n_steps = raw.data.shape[0] // 126858
    dry = dry_mask(CHANNELS, read_mesh_entity(MESH, 126858), n_steps)
    assert np.isnan(got.data[dry]).all()
    np.testing.assert_array_equal(got.data[~dry], raw.data[~dry])
    np.testing.assert_array_equal(got.coords, raw.coords)


@pytest.mark.parametrize("days", [1, 2])
def test_dry_inputs_become_zero_after_normalisation(days):
    reader = _reader(days, **MESH_KEYS)
    src = reader.get_source(1)
    n_steps = src.data.shape[0] // 126858
    dry = dry_mask(CHANNELS, read_mesh_entity(MESH, 126858), n_steps)
    norm = reader.normalize_source_channels(src.data.copy())
    assert np.isfinite(norm).all()
    np.testing.assert_allclose(norm[dry], 0.0, atol=1e-6)


def test_bathymetry_geoinfo_matches_the_mesh_and_normalises():
    reader = _reader(**MESH_KEYS)
    src = reader.get_source(1)
    assert src.geoinfos.shape == (src.data.shape[0], 1)
    raw = bathymetry(read_mesh_entity(MESH, 126858))
    np.testing.assert_allclose(src.geoinfos[:126858, 0], raw.astype(np.float32))
    g = reader.normalize_geoinfos(src.geoinfos.astype(np.float64).copy())
    assert abs(g.mean()) < 1e-3 and abs(g.std() - 1.0) < 1e-3


def test_without_mesh_keys_the_reader_is_unchanged():
    a, b = _reader(), _reader()
    np.testing.assert_array_equal(a.get_source(1).data, b.get_source(1).data)
    assert a.get_source(1).geoinfos.shape[1] == 0
