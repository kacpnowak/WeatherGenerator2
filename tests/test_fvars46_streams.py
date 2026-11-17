# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pin the 46-level FESOM stream files (a typo in a level list once silently dropped a level)."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STREAMS = ROOT / "config" / "streams"

DEPTHS = [
    2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210, 255, 310, 375, 450,
    535, 630, 735, 850, 975, 1110, 1255, 1415, 1600, 1810, 2035, 2275, 2525, 2775, 3025,
    3275, 3525, 3775, 4025, 4275, 4525, 4775, 5025, 5275, 5525, 5825,
]  # fmt: skip

LEVANTE_MESH = (
    "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
)
JUPITER_MESH = "/e/data1/climateai/hclimrep/data/ifs-fesom/fesom.mesh.diag.nc"

DIRS = ["fvars46_levante", "fesom_all_levels", "fesom_all_levels_forced"]
# file, stream name, channel prefix, wet-statistics file, stream id
LEVELLED = [
    ("temp", "TEMP", "temp", "wet_stats_node.json", 2150),
    ("salt", "SALT", "salt", "wet_stats_node.json", 2151),
    ("u", "U", "u", "wet_stats_elem.json", 2152),
    ("v", "V", "v", "wet_stats_elem.json", 2153),
]
LEVANTE_FILES = ["temp", "salt", "u", "v", "surface", "ifs_forcing"]
SIZE_CHANGED = {"TEMP", "SALT", "U", "V"}


def load(directory, name):
    """Return the single stream config (name, dict) of streams/<directory>/<name>.yml."""
    with open(STREAMS / directory / f"{name}.yml") as f:
        content = yaml.safe_load(f)
    assert len(content) == 1
    return next(iter(content.items()))


@pytest.mark.parametrize("directory", DIRS)
@pytest.mark.parametrize(("fname", "stream", "prefix", "stats", "stream_id"), LEVELLED)
def test_levels(directory, fname, stream, prefix, stats, stream_id):
    name, cfg = load(directory, fname)
    assert name == stream
    expected = [f"{prefix}_{d}" for d in DEPTHS]
    assert len(expected) == 46
    assert cfg["source"] == expected
    assert cfg["target"] == expected
    assert not any(c.endswith("_6125") for c in cfg["source"] + cfg["target"])
    assert cfg["stream_id"] == stream_id
    assert cfg["normalization_stats"] == f"config/streams/fesom/{stats}"
    with open(ROOT / cfg["normalization_stats"]) as f:
        known = json.load(f)
    missing = [c for c in expected if c not in known]
    assert not missing, f"no wet statistics for {missing}"


def test_stream_ids_distinct():
    ids = {}
    for fname in LEVANTE_FILES:
        name, cfg = load("fvars46_levante", fname)
        ids[name] = cfg["stream_id"]
    assert ids == {
        "TEMP": 2150,
        "SALT": 2151,
        "U": 2152,
        "V": 2153,
        "SURFACE": 2154,
        "IFS_ATMO": 2139,
    }
    assert len(set(ids.values())) == len(ids)


@pytest.mark.parametrize("fname", LEVANTE_FILES)
def test_levante_sizes_and_mesh(fname):
    name, cfg = load("fvars46_levante", fname)
    if name in SIZE_CHANGED:
        assert cfg["embed"]["dim_embed"] == 512
        assert cfg["target_readout"]["num_layers"] == 4
    elif name == "SURFACE":
        assert cfg["embed"]["dim_embed"] == 256
        assert cfg["target_readout"]["num_layers"] == 2
    if name != "IFS_ATMO":
        assert cfg["mesh_path"] == LEVANTE_MESH
        assert cfg["max_num_targets"] == 32768
    else:
        assert "mesh_path" not in cfg


@pytest.mark.parametrize("directory", ["fesom_all_levels", "fesom_all_levels_forced"])
@pytest.mark.parametrize("fname", ["u", "v"])
def test_jupiter_mesh_and_sizes(directory, fname):
    _, cfg = load(directory, fname)
    assert cfg["mesh_path"] == JUPITER_MESH
    assert cfg["embed"]["dim_embed"] == 256
    assert cfg["target_readout"]["num_layers"] == 2


@pytest.mark.parametrize("fname", LEVANTE_FILES)
def test_levante_matches_fesom_vars(fname):
    """Apart from levels, mesh path and the two size keys, nothing differs from fesom_vars."""
    name, new = load("fvars46_levante", fname)
    old_name, old = load("fesom_vars", fname)
    assert name == old_name
    ignored = {"source", "target", "mesh_path"}
    for cfg in (new, old):
        for key in ignored:
            cfg.pop(key, None)
        if name in SIZE_CHANGED:
            cfg["embed"] = {k: v for k, v in cfg["embed"].items() if k != "dim_embed"}
            cfg["target_readout"] = {
                k: v for k, v in cfg["target_readout"].items() if k != "num_layers"
            }
    assert new == old
