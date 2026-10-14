"""The per-variable FESOM stream set."""

from pathlib import Path

import pytest

from weathergen.common.config import load_streams

STREAMS = Path(__file__).resolve().parents[1] / "config" / "streams" / "fesom_vars"
DEPTHS = [2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210, 255, 310, 375,
          450, 535, 630, 735, 850, 975, 1110, 1255, 1415, 1600, 1810, 2035]
EXPECTED = {
    "TEMP": (2150, "ocean_node", [f"temp_{d}" for d in DEPTHS], "wet_stats_node.json"),
    "SALT": (2151, "ocean_node", [f"salt_{d}" for d in DEPTHS], "wet_stats_node.json"),
    "U": (2152, "ocean_elem", [f"u_{d}" for d in DEPTHS], "wet_stats_elem.json"),
    "V": (2153, "ocean_elem", [f"v_{d}" for d in DEPTHS], "wet_stats_elem.json"),
    "SURFACE": (2154, "ocean_node", ["ssh", "a_ice", "m_ice"], None),
}
ARCHIVE = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")


def test_stream_names():
    assert set(load_streams(STREAMS)) == set(EXPECTED) | {"IFS_ATMO"}


@pytest.mark.parametrize("name", EXPECTED)
def test_ocean_stream(name):
    s = load_streams(STREAMS)[name]
    stream_id, archive, channels, stats = EXPECTED[name]
    assert s["stream_id"] == stream_id and list(s["filenames"]) == [archive]
    assert list(s["source"]) == channels and list(s["target"]) == channels
    assert s["loss_weight"] == 1.0
    assert s["wet_mask_from_mesh"] is True and s["bathymetry_geoinfo"] is True
    assert s["mesh_path"] == "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
    if stats:
        assert s["normalization_stats"] == f"config/streams/fesom/{stats}"
    else:
        assert "normalization_stats" not in s
    assert "target_fill_value" not in s and "source_fill_value" not in s


def test_no_duplicate_surface_channels():
    for s in load_streams(STREAMS).values():
        assert not {"sst", "sss"} & set(s.get("source") or [])


def test_forcing_is_the_atmospheric_state_set():
    link = STREAMS / "ifs_forcing.yml"
    assert link.is_symlink() and link.resolve() == (STREAMS.parent / "fesom_forced_atm" / "ifs_forcing.yml").resolve()


@pytest.mark.skipif(not ARCHIVE.is_dir(), reason="AWI-CM3 data absent")
@pytest.mark.parametrize("name", EXPECTED)
def test_channels_exist_in_the_archive(name):
    import zarr

    _, archive, channels, _ = EXPECTED[name]
    cols = zarr.open_group(ARCHIVE / archive / f"{archive}_2100.zarr", mode="r")["data"].attrs["colnames"]
    assert set(channels) <= set(cols)
