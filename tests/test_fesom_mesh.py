"""FESOM mesh helpers: wet layers, channel levels, dry mask, bathymetry."""

from pathlib import Path

import numpy as np
import pytest

from weathergen.readers_extra.fesom_mesh import (
    MeshEntity,
    bathymetry,
    channel_level,
    dry_mask,
    read_mesh_entity,
)

MESH = Path(
    "/e/data1/climateai/hclimrep/data/ifs-fesom/fesom.mesh.diag.nc"
)
NZ1 = np.array([2.5, 7.5, 15.0, 25.0, 450.0, 2035.0])


@pytest.mark.parametrize(
    "name, level",
    [
        ("temp_2", 0),
        ("salt_7", 1),
        ("u_15", 2),
        ("v_450", 4),
        ("temp_2035", 5),
        ("ssh", None),
        ("a_ice", None),
        ("m_ice", None),
        ("MLD2", None),
        ("sst", None),
    ],
)
def test_channel_level(name, level):
    assert channel_level(name, NZ1) == level


def test_dry_mask_follows_wet_layers_and_tiles_per_step():
    mesh = MeshEntity(np.array([1, 5, 0]), np.array([5.0, 3000.0, 1.0]), NZ1)
    mask = dry_mask(["temp_2", "temp_450", "ssh"], mesh, n_steps=2)
    expected_day = np.array([[False, True, False], [False, False, False], [True, True, False]])
    np.testing.assert_array_equal(mask, np.concatenate([expected_day, expected_day]))


def test_bathymetry_is_log1p_bottom_depth():
    mesh = MeshEntity(np.array([1, 5]), np.array([0.0, 1000.0]), NZ1)
    np.testing.assert_allclose(bathymetry(mesh), np.log1p([0.0, 1000.0]))


@pytest.mark.skipif(not MESH.is_file(), reason="FESOM mesh not available")
@pytest.mark.parametrize("n_points, wet_450", [(126858, 0.73), (244659, 0.73)])
def test_real_mesh_entities(n_points, wet_450):
    mesh = read_mesh_entity(MESH, n_points)
    assert mesh.n_wet_layers.shape == mesh.bottom_depth.shape == (n_points,)
    assert (mesh.bottom_depth > 0).all()
    k = channel_level("temp_450", mesh.nz1)
    assert abs(np.mean(mesh.n_wet_layers > k) - wet_450) < 0.01


@pytest.mark.skipif(not MESH.is_file(), reason="FESOM mesh not available")
def test_unknown_point_count_is_rejected():
    with pytest.raises(ValueError, match="neither"):
        read_mesh_entity(MESH, 12345)
