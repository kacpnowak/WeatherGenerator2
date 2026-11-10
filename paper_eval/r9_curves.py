"""Overlay the 9-year rollouts of the long-rollout rounds: error against lead for every run.

    python -m paper_eval.r9_curves --cache-root <output.cache_dir> --prefix r9 --out <dir>

For one start date (run ids `<prefix><name>`), two figures with a linear lead axis in years:
the 30-day running mean of the global RMSE, and its cumulative mean from day 1, each for SST,
SSS, SSH, temperature and salinity at 450 m and surface current speed. Climatology is dashed.
A run without score files is left out.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

#: run name (after the prefix), legend label, colour, line style
RUNS = [
    ("nn003", "fvnn003 (reference)", "#8a979e", "-"),
    ("ctrl", "C control", "#2b7bb9", "-"),
    ("drop", "A dropout off", "#7fc4dc", "-"),
    ("spin", "B spin-up 0-365", "#8a5a9c", "-"),
    ("bias", "D mean-error loss", "#c0513a", "-"),
    ("ctrl2", "C2 = C + cycle", "#123f6c", "-"),
    ("drop2", "A2 = A + cycle", "#2a9d8f", "-"),
    ("r2ctl", "R2 A2 + cycle", "#e09f3e", "--"),
    ("r2spn", "R2 A2 + spin-up", "#b5179e", "--"),
    ("r2mix", "R2 A2 + mixed spin-up", "#2d6a4f", "--"),
]

#: panel title, unit, (stream, channel) or the u/v pair for current speed
PANELS = [
    ("SST", "°C", ("TEMP", "temp_2")),
    ("SSS", "psu", ("SALT", "salt_2")),
    ("SSH", "m", ("SURFACE", "ssh")),
    ("Temperature 450 m", "°C", ("TEMP", "temp_450")),
    ("Salinity 450 m", "psu", ("SALT", "salt_450")),
    ("Surface current", "m/s", (("U", "u_2"), ("V", "v_2"))),
]
STREAMS = ("TEMP", "SALT", "SURFACE", "U", "V")
WINDOW = 30


def _read(cache_root: Path, run: str) -> dict[str, xr.DataArray] | None:
    """Global RMSE of every stream, (model, lead, channel); each file read and closed before the
    next, opening many lazily segfaults netCDF4."""
    out = {}
    for stream in STREAMS:
        path = cache_root / run / f"scores_{stream}.nc"
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            rmse = ds.score.sel(region="global", metric="rmse")
            out[stream] = np.sqrt((rmse**2).mean("sample")).load()
    return out


def _curve(scores: dict[str, xr.DataArray], spec, model: str) -> xr.DataArray:
    if isinstance(spec[0], tuple):
        (su, cu), (sv, cv) = spec
        u = scores[su].sel(model=model, channel=cu)
        v = scores[sv].sel(model=model, channel=cv)
        return np.sqrt(u**2 + v**2)
    stream, channel = spec
    return scores[stream].sel(model=model, channel=channel)


def _running(da: xr.DataArray) -> np.ndarray:
    return da.rolling(lead=WINDOW, center=True, min_periods=WINDOW // 2).mean().values


def _cumulative(da: xr.DataArray) -> np.ndarray:
    values = da.values
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def build(cache_root: Path, prefix: str, out: Path, start: str) -> list[Path]:
    """The two figures for the runs `<prefix><name>` that have score files."""
    present = []
    for name, label, color, style in RUNS:
        scores = _read(cache_root, prefix + name)
        if scores is not None:
            present.append((label, color, style, scores))
    if not present:
        raise SystemExit(f"no scored runs with prefix {prefix} under {cache_root}")

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, reduce, what in (
        ("running", _running, f"{WINDOW}-day running mean of the global RMSE"),
        ("cumulative", _cumulative, "cumulative mean of the global RMSE from day 1"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
        for ax, (title, unit, spec) in zip(axes.flat, PANELS, strict=True):
            for label, color, style, scores in present:
                da = _curve(scores, spec, "emulator")
                years = da.lead.values / 365.25
                ax.plot(years, reduce(da), color=color, linestyle=style, linewidth=1.4, label=label)
            clim = _curve(present[0][3], spec, "climatology")
            ax.plot(
                clim.lead.values / 365.25,
                reduce(clim),
                color="black",
                linestyle=":",
                linewidth=1.2,
                label="climatology",
            )
            ax.set_title(title)
            ax.set_ylabel(f"RMSE [{unit}]")
            ax.set_xlim(0, 9)
            ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("lead [years]")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 6))
        fig.suptitle(f"9-year rollouts from {start}: {what}")
        path = out / f"{prefix}_{kind}_rmse.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="r9 (from 2201-01-01) or r0 (2001-01-01)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = {"r9": "2201-01-01", "r0": "2001-01-01"}.get(args.prefix, args.prefix)
    for path in build(args.cache_root, args.prefix, args.out, start):
        print(path)  # noqa: T201


if __name__ == "__main__":
    main()
