"""The forced IFS-FESOM stream sets load and describe IFS_ATMO as a forcing."""

from pathlib import Path

import numpy as np

from weathergen.common.config import load_streams, parse_timedelta
from weathergen.utils.utils import is_stream_fe_only

STREAMS = Path(__file__).resolve().parents[1] / "config" / "streams"
IFS_CHANNELS = ["10u", "10v", "2d", "2t", "cp", "lsp", "msl", "tsr", "tsrc"]


def _check_forced_set(name, mode):
    streams = load_streams(STREAMS / name)
    ifs = streams["IFS_ATMO"]
    assert ifs["type"] == "forcing" and ifs["reader"] == "fesom"
    assert ifs["injection"]["mode"] == mode
    assert ifs["injection"]["time_reduce"] == "mean"
    assert list(ifs["source"]) == IFS_CHANNELS
    assert parse_timedelta(ifs["frequency"]) == np.timedelta64(6, "h")
    assert is_stream_fe_only(ifs)
    for ocean in ("FESOM_NODES", "FESOM_ELEMS"):
        assert streams[ocean]["type"] == "fesom"
        assert not is_stream_fe_only(streams[ocean])
    return streams


def test_fesom_forced_stream_set():
    streams = _check_forced_set("fesom_forced", "additive")
    assert set(streams.keys()) == {"FESOM_NODES", "FESOM_ELEMS", "IFS_ATMO"}


def test_fesom_forced_adaln_stream_set():
    streams = _check_forced_set("fesom_forced_adaln", "adaln_local")
    assert set(streams.keys()) == {"FESOM_NODES", "FESOM_ELEMS", "IFS_ATMO", "Forecast_conditions"}
    assert streams["Forecast_conditions"]["type"] == "condition"
    assert is_stream_fe_only(streams["Forecast_conditions"])


def test_ocean_streams_are_shared_with_the_unforced_set():
    for name in ("fesom_forced", "fesom_forced_adaln"):
        for fname in ("fesom.yml", "fesom_elem.yml"):
            link = STREAMS / name / fname
            assert link.is_symlink()
            assert link.resolve() == (STREAMS / "fesom" / fname).resolve()
