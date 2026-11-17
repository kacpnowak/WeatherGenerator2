"""Compare the 9-year rollouts of long-rollout round 1 from their stage-1 score caches.

    python -m paper_eval.r9_table --cache-root <output.cache_dir> --runs r9nn003 r9ctrl r9drop

Prints a Markdown table with one row per run. Per year-end lead: the mean over channels of
RMSE_emulator / RMSE_climatology for the node and the elem streams (below 1 beats climatology),
and the emulator bias of temperature and salinity at 107, 450 and 975 m. Global region, mean
over samples.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

STREAMS = {"node": ("TEMP", "SALT", "SURFACE"), "elem": ("U", "V")}
BIAS_STREAMS = {"TEMP": "temp", "SALT": "salt"}
BIAS_DEPTHS = (107, 450, 975)
YEAR_END_LEADS = [365, 730, 1095, 1461, 1826, 2191, 2556, 2922, 3285]


def _read_scores(cache_root: Path, run: str) -> dict[str, xr.DataArray]:
    """Every stream's global scores; each file is read and closed before the next, opening
    many lazily segfaults netCDF4."""
    scores = {}
    for stream in sum(STREAMS.values(), ()):
        with xr.open_dataset(cache_root / run / f"scores_{stream}.nc") as ds:
            scores[stream] = ds.score.sel(region="global").load()
    return scores


def _clim_ratio(scores: dict[str, xr.DataArray], streams: tuple, lead: int) -> float:
    ratios = []
    for stream in streams:
        rmse = scores[stream].sel(metric="rmse", lead=lead).mean("sample")
        ratios.append((rmse.sel(model="emulator") / rmse.sel(model="climatology")).values)
    return float(np.mean(np.concatenate(ratios)))


def _bias(scores: dict[str, xr.DataArray], stream: str, channel: str, lead: int) -> float:
    selected = scores[stream].sel(model="emulator", metric="bias", lead=lead, channel=channel)
    return float(selected.mean("sample"))


def build_table(cache_root: Path, runs: list[str], leads: list[int]) -> str:
    """The Markdown table of `runs` at `leads`; a run without score files says so."""
    columns = [f"{entity}/clim d{lead}" for lead in leads for entity in STREAMS]
    channels = [
        (stream, f"{prefix}_{depth}")
        for stream, prefix in BIAS_STREAMS.items()
        for depth in BIAS_DEPTHS
    ]
    columns += [f"bias {channel} d{lead}" for lead in leads for _, channel in channels]
    lines = ["| run | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for run in runs:
        try:
            scores = _read_scores(cache_root, run)
        except FileNotFoundError:
            lines.append(f"| {run} | not scored |")
            continue
        cells = [
            f"{_clim_ratio(scores, streams, lead):.3f}"
            for lead in leads
            for streams in STREAMS.values()
        ]
        cells += [
            f"{_bias(scores, stream, channel, lead):.3f}"
            for lead in leads
            for stream, channel in channels
        ]
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--leads", type=int, nargs="+", default=YEAR_END_LEADS)
    args = parser.parse_args()
    sys.stdout.write(build_table(args.cache_root, args.runs, args.leads) + "\n")


if __name__ == "__main__":
    main()
