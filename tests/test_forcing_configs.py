"""The forced IFS-FESOM stream sets and overwrites describe IFS_ATMO as a forcing."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from weathergen.common.config import load_merge_configs, load_streams, parse_timedelta
from weathergen.utils.utils import is_stream_fe_only

CONFIG = Path(__file__).resolve().parents[1] / "config"
STREAMS = CONFIG / "streams"
IFS_CHANNELS = ["10u", "10v", "2d", "2t", "cp", "lsp", "msl", "tsr", "tsrc"]
FORCED_SETS = {"fesom_forced": "additive", "fesom_forced_adaln": "adaln_local"}


def _ocean_stream_names():
    # the forced sets share the ocean stream files of the unforced set, whatever they are called
    return set(load_streams(STREAMS / "fesom").keys()) - {"IFS_ATMO"}


def _check_ifs_forcing(ifs, mode):
    assert ifs["type"] == "forcing" and ifs["reader"] == "fesom"
    assert ifs["injection"]["mode"] == mode
    assert ifs["injection"]["time_reduce"] == "mean"
    assert list(ifs["source"]) == IFS_CHANNELS
    assert parse_timedelta(ifs["frequency"]) == np.timedelta64(6, "h")
    assert is_stream_fe_only(ifs)


def _check_forced_set(name, mode):
    streams = load_streams(STREAMS / name)
    _check_ifs_forcing(streams["IFS_ATMO"], mode)
    ocean = {n: s for n, s in streams.items() if not is_stream_fe_only(s)}
    assert len(ocean) == 2
    assert all(s["type"] == "fesom" for s in ocean.values())
    assert set(ocean) == _ocean_stream_names()
    return streams


def test_fesom_forced_stream_set():
    streams = _check_forced_set("fesom_forced", "additive")
    assert set(streams.keys()) == _ocean_stream_names() | {"IFS_ATMO"}


def test_fesom_forced_adaln_stream_set():
    streams = _check_forced_set("fesom_forced_adaln", "adaln_local")
    assert set(streams.keys()) == _ocean_stream_names() | {"IFS_ATMO", "Forecast_conditions"}
    assert streams["Forecast_conditions"]["type"] == "condition"
    assert is_stream_fe_only(streams["Forecast_conditions"])


@pytest.mark.parametrize("name", FORCED_SETS)
def test_fe_only_streams_have_empty_target_channel_lists(name):
    # the dynamic loss reads train_target_channels of every stream, FE-only ones included
    fe_only = [s for s in load_streams(STREAMS / name).values() if is_stream_fe_only(s)]
    assert fe_only
    for stream in fe_only:
        for key in ("train_target_channels", "val_target_channels"):
            assert key in stream, f"{stream['name']} has no {key}"
            assert list(stream[key]) == [], f"{stream['name']}.{key} is not empty"


def test_ocean_streams_are_shared_with_the_unforced_set():
    for name in FORCED_SETS:
        for fname in ("fesom.yml", "fesom_elem.yml"):
            link = STREAMS / name / fname
            assert link.is_symlink()
            assert link.resolve() == (STREAMS / "fesom" / fname).resolve()


# a saved ocean run: ocean streams under the names the run was trained with, and IFS_ATMO as
# the old diagnostic data stream with targets
_OCEAN_STREAM = {
    "type": "fesom",
    "filenames": ["ocean_node"],
    "source": ["sst", "sss"],
    "target": ["sst", "sss"],
    "train_target_channels": ["sst", "sss"],
    "embed": {"net": "transformer", "num_tokens": 1, "dim_embed": 256},
}
_RUN_STREAMS = {
    "FESOM_NODE": {**_OCEAN_STREAM, "name": "FESOM_NODE", "stream_id": 2},
    "FESOM_ELEM": {**_OCEAN_STREAM, "name": "FESOM_ELEM", "stream_id": 1},
    "IFS_ATMO": {
        "type": "fesom",
        "name": "IFS_ATMO",
        "stream_id": 2139,
        "diagnostic": True,
        "filenames": ["atmos_all"],
        "loss_weight": 1.0,
        "source": ["2t"],
        "source_exclude": ["t"],
        "target": ["^"],
        "target_exclude": ["msl", "tsr", "tsrc", "_"],
        "train_target_channels": ["2t"],
        "val_target_channels": ["2t"],
    },
}


def _merge_into_run(tmp_path, overwrite):
    base = OmegaConf.create(
        {"streams_directory": "./config/streams/fesom/", "streams": _RUN_STREAMS}
    )
    private = tmp_path / "private.yml"
    private.write_text("data_paths: []\n")
    return load_merge_configs(private, None, None, base, CONFIG / overwrite)


def _stream_file_entry(name, stream_name):
    entry = OmegaConf.to_container(OmegaConf.load(STREAMS / name / f"{stream_name}.yml"))
    return entry[next(iter(entry))]


@pytest.mark.parametrize(
    "overwrite, stream_set",
    [
        ("fesom_forced_overwrite.yml", "fesom_forced"),
        ("fesom_forced_adaln_overwrite.yml", "fesom_forced_adaln"),
    ],
)
def test_overwrite_turns_the_runs_ifs_stream_into_a_forcing(tmp_path, overwrite, stream_set):
    cf = _merge_into_run(tmp_path, overwrite)
    streams = OmegaConf.to_container(cf.streams)

    # the run's ocean streams (and so their checkpoint weights) are kept as they are
    assert cf.streams_directory == "./config/streams/fesom/"
    for name in ("FESOM_NODE", "FESOM_ELEM"):
        assert streams[name] == _RUN_STREAMS[name]

    ifs = cf.streams["IFS_ATMO"]
    _check_ifs_forcing(ifs, FORCED_SETS[stream_set])
    assert not ifs.get("diagnostic", False)
    assert list(ifs["source_exclude"]) == []
    assert list(ifs["train_target_channels"]) == [] and list(ifs["val_target_channels"]) == []
    assert ifs["loss_weight"] == 0.0
    # the overwrite carries the stream file's IFS_ATMO entry
    for key, value in _stream_file_entry(stream_set, "ifs_forcing").items():
        if key != "frequency":
            assert streams["IFS_ATMO"][key] == value, key

    expected = {"FESOM_NODE", "FESOM_ELEM", "IFS_ATMO"}
    if stream_set == "fesom_forced_adaln":
        expected.add("Forecast_conditions")
        condition = streams["Forecast_conditions"]
        assert condition["type"] == "condition" and condition["name"] == "Forecast_conditions"
        assert is_stream_fe_only(condition)
        assert {**_stream_file_entry(stream_set, "forecast"), "name": "Forecast_conditions"} == (
            condition
        )
    assert set(streams) == expected
