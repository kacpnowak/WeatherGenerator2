"""Ten-year free-running forecast (fvnn10y, from 2200-01-01): per-channel global RMSE against
lead and the drift of global means, from the fast scorer's daily series
(~/claude_api_tools/fable_diag/data/lead3650/part1_daily_RUN.nc; node channels only, u/v come
with paper_eval).

    python scripts/instab/plot_10y.py [--run fvnn10y] [--ref fvnnf00]
        [--out paper_eval/figures/fvnn10y]

RMSE: daily values thin, 30-day running mean bold; the dashed line is the 2180-2209
day-of-year climatology against the same truth (every 5 days, three channels only). The ref
run (the one-year run from the same start and checkpoint) is drawn over its year to show the
two agree.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paper_eval.plotting import style

DATA = Path.home() / "claude_api_tools/fable_diag/data"
C = style._CATEGORICAL
PANELS = [
    ("temp_2", "SST", "°C"), ("salt_2", "SSS", "psu"), ("ssh", "SSH", "m"),
    ("temp_107", "Temperature, 107 m", "°C"), ("salt_107", "Salinity, 107 m", "psu"),
    ("m_ice", "Sea-ice thickness", "m"),
    ("temp_450", "Temperature, 450 m", "°C"), ("salt_450", "Salinity, 450 m", "psu"),
    ("a_ice", "Sea-ice concentration", "1"),
    ("temp_975", "Temperature, 975 m", "°C"), ("salt_975", "Salinity, 975 m", "psu"),
    ("salt_2035", "Salinity, 2035 m", "psu"),
]
# global-mean quantity -> (title, unit, scale)
DRIFT = [
    ("vmT_0_200", "Volume-mean temperature, 0–200 m", "°C", 1.0),
    ("vmT_0_2150", "Volume-mean temperature, 0–2150 m", "°C", 1.0),
    ("vmS_0_200", "Volume-mean salinity, 0–200 m", "psu", 1.0),
    ("vmS_0_2150", "Volume-mean salinity, 0–2150 m", "psu", 1.0),
    ("ssh_mean", "Mean sea surface height", "m", 1.0),
    ("hc_0_2150", "Heat content, 0–2150 m", "J", 1.0),
    ("ice_area_nh", "Sea-ice area, NH", "10⁶ km²", 1e-12),
    ("ice_area_sh", "Sea-ice area, SH", "10⁶ km²", 1e-12),
]


def years(ds):
    return ds.lead.values / 365.25


def rmse_figure(ds, ref, out):
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    for ax, (ch, name, unit) in zip(axes.flat, PANELS):
        e = ds.rmse.sel(channel=ch)
        ax.plot(years(ds), e, color=C[0], lw=0.4, alpha=0.45)
        ax.plot(years(ds), e.rolling(lead=30, center=True).mean(), color=C[0], lw=1.6,
                label="emulator (fvnn10y), 30-day mean")
        if ref is not None:
            ax.plot(years(ref), ref.rmse.sel(channel=ch), color=C[1], lw=0.8,
                    label="one-year run from the same start (fvnnf00)")
        if f"clim_rmse_{ch}" in ds:
            cl = ds[f"clim_rmse_{ch}"].dropna("lead")
            ax.plot(cl.lead / 365.25, cl, color="#3d3c38", ls="--", lw=0.9,
                    label="climatology 2180–2209")
        ax.set_ylim(0, 1.1 * float(e.max()))
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_title(f"{name}  [{unit}]", loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE")
    for ax in axes[-1]:
        ax.set_xlabel("lead time (years)")
    h, labels = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                h.append(hh)
                labels.append(ll)
    fig.legend(h, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Global RMSE, ten-year free-running forced forecast from 2200-01-01 "
                 "(fvnn003, deterministic)", x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def drift_figure(ds, out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    for ax, (q, name, unit, sc) in zip(axes.flat, DRIFT):
        for var, colour, label in (("truth", "#3d3c38", "AWI-CM3 (truth)"),
                                   ("pred", C[0], "emulator (fvnn10y)")):
            ax.plot(years(ds), ds[var].sel(quantity=q) * sc, color=colour, lw=0.9, label=label)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_title(f"{name}  [{unit}]", loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("lead time (years)")
    h, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Global means, ten-year free-running forced forecast from 2200-01-01",
                 x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fvnn10y")
    ap.add_argument("--ref", default="fvnnf00")
    ap.add_argument("--out", default="paper_eval/figures/fvnn10y")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(DATA / "lead3650" / f"part1_daily_{a.run}.nc").load()
    ref_path = DATA / "lead363" / f"part1_daily_{a.ref}.nc"
    ref = xr.open_dataset(ref_path).load() if a.ref and ref_path.exists() else None
    rmse_figure(ds, ref, out / "rmse_channels_10y")
    drift_figure(ds, out / "global_means_10y")
    print("wrote", sorted(p.name for p in out.glob("*_10y.*")))


if __name__ == "__main__":
    main()
