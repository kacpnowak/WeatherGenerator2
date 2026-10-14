"""
Acceptance check of the long-horizon stability runs against the spec's success criteria
(docs/superpowers/specs/2026-10-08-long-horizon-stability-design.md).

    .venv/bin/python compare_year.py

Reads the paper_eval scores of the one-year runs (paper_eval/cache_oe3i7whz/<run>/), the drift
diagnostics (paper_eval/figures/year_diag/diag_<run>.nc) and the fine-tunes' training metrics;
prints one table per criterion and writes RMSE figures to paper_eval/figures/year_compare/.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent
CACHE = REPO / "paper_eval/cache_oe3i7whz"
DIAG = REPO / "paper_eval/figures/year_diag"
OUT = REPO / "paper_eval/figures/year_compare"
YEAR = {"window": "yr1win01", "control": "yr1ctl01", "frcadd04": "yr1add04",
        "ykwx7bl2": "yr1ykwx7"}
TRAINED = {"window": "frcwin01", "control": "frcctl01", "frcadd04": "frcadd04"}
REF = "frcadd04"
DEEP = ("temp_450", "salt_450", "salt_107")
COLOURS = dict(zip(YEAR, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True))
CLIM_COLOUR, INK, INK2, GRID, SURFACE = "#52514e", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def scores(run, stream):
    path = CACHE / run / f"scores_FESOM_{stream}.nc"
    return xr.open_dataset(path).score.sel(region="global").isel(sample=0)


def clim_ratio(s):
    """Channel-mean RMSE / climatology RMSE, per lead."""
    rmse = s.sel(metric="rmse")
    return (rmse.sel(model="emulator") / rmse.sel(model="climatology")).mean("channel")


def last_validation(run):
    path = REPO / "results" / run / f"{run}_train_metrics.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    val = [r for r in rows if r.get("stage") == "val"][-1]
    return val["LossPhysical.FESOM_NODE.mse.avg"], val["LossPhysical.FESOM_ELEM.mse.avg"]


def main():
    S = {(k, st): scores(r, st) for k, r in YEAR.items() for st in ("NODE", "ELEM")}
    last = int(S[(REF, "NODE")].lead.max())
    ratio = {key: clim_ratio(s) for key, s in S.items()}

    print(f"1. Drift: channel-mean RMSE / climatology RMSE (global), day 180 and day {last}")
    for st in ("NODE", "ELEM"):
        for k in YEAR:
            r = ratio[(k, st)]
            print(f"   {st:4} {k:9} d180 {float(r.sel(lead=180)):5.2f}   "
                  f"d{last} {float(r.sel(lead=last)):5.2f}")

    print("\n2. Deep fields: RMSE, persistence, share of frcadd04's gap to persistence closed")
    for field in DEEP:
        for lead in (180, last):
            ref = float(S[(REF, "NODE")].sel(model="emulator", metric="rmse",
                                             channel=field, lead=lead))
            pers = float(S[(REF, "NODE")].sel(model="persistence", metric="rmse",
                                              channel=field, lead=lead))
            row = f"   {field:9} d{lead:<4} pers {pers:.4f} |"
            for k in YEAR:
                v = float(S[(k, "NODE")].sel(model="emulator", metric="rmse",
                                             channel=field, lead=lead))
                closed = (ref - v) / (ref - pers) if ref != pers else np.nan
                row += f" {k} {v:.4f} ({100 * closed:+.0f}%)"
            print(row)

    print("\n3. Short range (limit: at most ~5 % worse than frcadd04)")
    ref_node, ref_elem = last_validation(TRAINED[REF])
    for k, run in TRAINED.items():
        node, elem = last_validation(run)
        first30 = {st: float(ratio[(k, st)].sel(lead=slice(1, 30)).mean()
                             / ratio[(REF, st)].sel(lead=slice(1, 30)).mean())
                   for st in ("NODE", "ELEM")}
        print(f"   {k:9} 15-step val MSE / frcadd04: NODE {node / ref_node:5.3f} "
              f"ELEM {elem / ref_elem:5.3f} | days 1-30 RMSE/clim vs frcadd04: "
              f"NODE {first30['NODE']:5.3f} ELEM {first30['ELEM']:5.3f}")

    print("\n4. Variability ratio, leads 30 to the end (target about 0.7-1.3)")
    for k, run in YEAR.items():
        d = xr.open_dataset(DIAG / f"diag_{run}.nc")["variability_ratio"].sel(lead=slice(30, None))
        lows = d.min("lead")
        highs = d.max("lead")
        cells = " ".join(f"{f} {float(lows.sel(field=f)):.2f}-{float(highs.sel(field=f)):.2f}"
                         for f in d.field.values)
        print(f"   {k:9} {cells}")

    plot(S, ratio)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(bottom=0)


def plot(S, ratio):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, st in zip(axes, ("NODE", "ELEM"), strict=True):
        for k in YEAR:
            r = ratio[(k, st)]
            ax.plot(r.lead, r, color=COLOURS[k], lw=2, label=k)
            ax.annotate(k, (float(r.lead[-1]), float(r[-1])), xytext=(4, 0),
                        textcoords="offset points", color=INK2, fontsize=8, va="center")
        ax.axhline(1, color=CLIM_COLOUR, lw=1.5, ls=(0, (4, 3)), label="climatology (= 1)")
        ax.set_title(f"FESOM_{st}: channel mean", loc="left")
        ax.set_xlabel("lead time (days)")
        _style(ax)
    axes[0].set_ylabel("RMSE / climatology RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("One-year forecast from 2209-01-01: RMSE relative to climatology",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(OUT / "rmse_vs_clim_mean.png", dpi=160)
    plt.close(fig)

    panels = [("NODE", "sst"), ("NODE", "ssh"), ("ELEM", "u_2"), ("NODE", "temp_107"),
              ("NODE", "salt_107"), ("NODE", "temp_450"), ("NODE", "salt_450"),
              ("ELEM", "u_107"), ("NODE", "sss")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    for ax, (st, ch) in zip(axes.flat, panels, strict=True):
        for k in YEAR:
            s = S[(k, st)].sel(model="emulator", metric="rmse", channel=ch)
            ax.plot(s.lead, s, color=COLOURS[k], lw=1.8, label=k)
        clim = S[(REF, st)].sel(model="climatology", metric="rmse", channel=ch)
        ax.plot(clim.lead, clim, color=CLIM_COLOUR, lw=1.4, ls=(0, (4, 3)), label="climatology")
        pers = S[(REF, st)].sel(model="persistence", metric="rmse", channel=ch)
        ax.plot(pers.lead, pers, color=CLIM_COLOUR, lw=1.0, ls=(0, (1, 2)), label="persistence")
        ax.set_title(ch, loc="left")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("One-year forecast from 2209-01-01: global RMSE", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.savefig(OUT / "rmse_channels.png", dpi=160)
    plt.close(fig)
    print(f"\nfigures: {OUT}")


if __name__ == "__main__":
    main()
