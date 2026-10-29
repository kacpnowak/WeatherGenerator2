"""Per-channel global RMSE of one-year rollouts from the paper_eval score caches, in the
layout of paper_eval/figures/f363_compare (12 channel panels + channel-mean RMSE ratio).

Compares fvars003 with training-like latent noise at inference (nztl09/nztl50/nztl00),
fvars003 deterministic (yr1fv003) and the old lineage (yr1win03). Old-lineage channels
are looked up by name, sst/sss standing in for temp_2/salt_2 where missing. Curves that
blow up are clipped to the axes of the bounded runs.

    python scripts/instab/plot_year_compare.py [--out paper_eval/figures/fnoise_compare]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paper_eval.plotting import style

CACHE = Path("paper_eval/cache_oe3i7whz")
ALIAS = {"temp_2": "sst", "salt_2": "sss"}
PANELS = [
    ("temp_2", "SST"), ("salt_2", "SSS"), ("ssh", "SSH"),
    ("u_2", "u, 2 m"), ("v_2", "v, 2 m"), ("temp_107", "Temperature, 107 m"),
    ("salt_107", "Salinity, 107 m"), ("u_107", "u, 107 m"), ("temp_450", "Temperature, 450 m"),
    ("salt_450", "Salinity, 450 m"), ("u_450", "u, 450 m"), ("v_450", "v, 450 m"),
]
C = style._CATEGORICAL
# run id -> (label, colour, start year)
YEAR_RUNS = {
    "nztl09": ("fvars003 + training-like latent noise (nztl09)", C[0], 2209),
    "yr1fv003": ("fvars003, deterministic (yr1fv003)", C[1], 2209),
    "yr1win03": ("old lineage, frcwin03 (yr1win03)", C[3], 2209),
}
START_RUNS = {
    "nztl09": ("start 2209-01-01 (nztl09)", C[0], 2209),
    "nztl50": ("start 2150-01-01 (nztl50)", C[2], 2150),
    "nztl00": ("start 2200-01-01 (nztl00)", C[6], 2200),
}
BLOWUP = {"yr1fv003"}


def load(run: str) -> dict[str, xr.DataArray]:
    """channel -> score(model, lead), global RMSE of the single sample, plus its entity."""
    out = {}
    for f in sorted((CACHE / run).glob("scores_*.nc")):
        ds = xr.open_dataset(f)
        entity = "elem" if f.stem.split("_", 1)[1] in ("U", "V", "FESOM_ELEM") else "node"
        da = ds.score.sel(region="global", metric="rmse").isel(sample=0).load()
        for ch in da.channel.values:
            out[str(ch)] = (da.sel(channel=ch), entity)
    return out


def get(scores, ch):
    if ch in scores:
        return scores[ch][0]
    return scores[ALIAS[ch]][0] if ALIAS.get(ch) in scores else None


def channel_figure(runs: dict, scores: dict, title: str, per_run_clim: bool, out: Path):
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    for ax, (ch, name) in zip(axes.flat, PANELS):
        ymax = 0.0
        for run, (label, colour, _) in runs.items():
            da = get(scores[run], ch)
            if da is None:
                continue
            emu = da.sel(model="emulator")
            ax.plot(emu.lead, emu, color=colour, label=label)
            if run not in BLOWUP:
                ymax = max(ymax, float(np.nanmax(emu)))
            if per_run_clim:
                clim = da.sel(model="climatology")
                ax.plot(clim.lead, clim, color=colour, ls="--", lw=1.0, alpha=0.7)
                ymax = max(ymax, float(np.nanmax(clim)))
        if not per_run_clim:
            clim = get(scores[next(iter(runs))], ch).sel(model="climatology")
            ax.plot(clim.lead, clim, color="#3d3c38", ls="--", label="climatology 2180–2209")
            ymax = max(ymax, float(np.nanmax(clim)))
        ax.set_ylim(0, 1.15 * ymax)
        ax.set_xlim(0, 365)
        ax.set_xticks(range(0, 361, 60))
        ax.set_title(f"{name}  [{style.channel_unit(ch)}]", loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE")
    for ax in axes[-1]:
        ax.set_xlabel("lead time (days)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if per_run_clim:
        handles.append(plt.Line2D([], [], color="#3d3c38", ls="--", lw=1.0))
        labels.append("climatology 2180–2209 (dashed, run colour)")
    labels = [l + " — off scale after blow-up" if "deterministic" in l else l for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)


def ratio_figure(runs: dict, scores: dict, out: Path):
    """Mean over channels of RMSE / climatology RMSE, on the channels all runs share."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, entity, desc in zip(axes, ("node", "elem"), ("T, S, SSH", "u, v")):
        common = set.intersection(*[
            {c for c, (_, e) in scores[r].items() if e == entity and c not in ALIAS.values()}
            for r in runs
        ])
        ymax = 1.0
        for run, (label, colour, _) in runs.items():
            ratio = xr.concat(
                [scores[run][c][0].sel(model="emulator") / scores[run][c][0].sel(model="climatology")
                 for c in sorted(common)], "channel").mean("channel")
            ax.plot(ratio.lead, ratio, color=colour, label=label)
            if run not in BLOWUP:
                ymax = max(ymax, float(np.nanmax(ratio)))
        ax.axhline(1.0, color="#3d3c38", ls="--", label="climatology (= 1)")
        ax.set_ylim(0, 1.15 * ymax)
        ax.set_xlim(0, 365)
        ax.set_xticks(range(0, 361, 60))
        ax.set_xlabel("lead time (days)")
        ax.set_title(f"{desc}, levels to 450 m ({len(common)} channels)", loc="left")
    axes[0].set_ylabel("RMSE / climatology RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [l + " — off scale after blow-up" if "deterministic" in l else l for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("RMSE relative to climatology, channel mean (below 1 = better than climatology)",
                 x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_eval/figures/fnoise_compare")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    scores = {r: load(r) for r in {**YEAR_RUNS, **START_RUNS}}
    channel_figure(YEAR_RUNS, scores, "Global RMSE, one-year forced forecast from 2209-01-01",
                   False, out / "rmse_channels")
    channel_figure(START_RUNS, scores,
                   "Global RMSE, one-year forecasts of fvars003 + training-like latent noise, three starts",
                   True, out / "rmse_channels_starts")
    ratio_figure({**YEAR_RUNS, **{k: v for k, v in START_RUNS.items() if k != "nztl09"}}, scores,
                 out / "rmse_vs_clim_mean")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
