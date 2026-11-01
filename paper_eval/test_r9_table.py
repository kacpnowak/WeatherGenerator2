"""The comparison table of the 9-year rollouts, on synthetic score files with known values."""

import numpy as np
import xarray as xr

from paper_eval.r9_table import STREAMS, build_table

CHANNELS = {
    "TEMP": ["temp_107", "temp_450", "temp_975"],
    "SALT": ["salt_107", "salt_450", "salt_975"],
    "SURFACE": ["ssh"],
    "U": ["u_2"],
    "V": ["v_2"],
}


def _write_scores(root, run, ratio, bias):
    """Climatology RMSE 2 everywhere, emulator RMSE 2 * ratio, emulator bias `bias`."""
    (root / run).mkdir()
    for stream in sum(STREAMS.values(), ()):
        channels = CHANNELS[stream]
        score = np.zeros((3, 1, 3, 1, 2, len(channels)))
        score[0, 0, 0] = 2 * ratio
        score[2, 0, 0] = 2.0
        score[0, 0, 1] = bias
        ds = xr.Dataset(
            {
                "score": (
                    ("model", "region", "metric", "sample", "lead", "channel"),
                    score,
                )
            },
            coords={
                "model": ["emulator", "persistence", "climatology"],
                "region": ["global"],
                "metric": ["rmse", "bias", "acc"],
                "sample": [0],
                "lead": [365, 3285],
                "channel": channels,
            },
        )
        ds.to_netcdf(root / run / f"scores_{stream}.nc")


def test_table_holds_the_ratios_and_biases_of_each_run(tmp_path):
    _write_scores(tmp_path, "r9a", 0.5, 0.25)
    _write_scores(tmp_path, "r9b", 1.25, -0.125)
    table = build_table(tmp_path, ["r9a", "r9b"], [365, 3285])
    rows = {line.split("|")[1].strip(): line for line in table.splitlines() if "r9" in line}
    assert set(rows) == {"r9a", "r9b"}
    assert rows["r9a"].count("0.500") == 2 * len(STREAMS)
    assert rows["r9b"].count("1.250") == 2 * len(STREAMS)
    assert rows["r9a"].count("0.250") == 2 * 6
    assert rows["r9b"].count("-0.125") == 2 * 6


def test_a_run_without_scores_is_marked(tmp_path):
    _write_scores(tmp_path, "r9a", 0.5, 0.25)
    table = build_table(tmp_path, ["r9a", "r9b"], [365])
    assert "| r9b | not scored |" in table
