"""
Drift diagnostics of one-year forced runs: regional means against the truth and the
variability ratio, per lead, for selected FESOM_NODE fields.

    .venv/bin/python diag_forced_year.py RUN [RUN ...]

For each run, reads results/<run>/validation_chkpt00000_rank*.zip (one forecast, written by
sbatch_fesom_forced_rollout.sh) and the paper_eval day-of-year climatology, and writes
paper_eval/figures/year_diag/diag_<run>.nc; then draws regional_means.png and
variability_ratio.png there for all runs given.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))  # paper_eval is a plain folder at the repo root

from paper_eval.climatology import ClimLookup  # noqa: E402
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402
from paper_eval.meshtools import load_mesh, masks_and_weights  # noqa: E402

warnings.filterwarnings("ignore")
STREAM = "FESOM_NODE"
FIELDS = ("sst", "ssh", "temp_107", "salt_107", "temp_450", "salt_450")
BANDS = {
    "global": None,
    "90S-30S": (-90.0, -30.0),
    "30S-30N": (-30.0, 30.0),
    "30N-90N": (30.0, 90.0),
}
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
CLIM = REPO / "paper_eval/cache_oe3i7whz/climatology/clim_FESOM_NODE_2180-2209.zarr"
OUT = REPO / "paper_eval/figures/year_diag"
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # validated categorical slots 1-4
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def band_mask(lat, bounds):
    """Points with south <= lat < north (all points for None), paper_eval's region rule."""
    lat = np.asarray(lat)
    if bounds is None:
        return np.ones(lat.shape, dtype=bool)
    south, north = bounds
    return (lat >= south) & (lat < north)


def weighted_mean(x, w):
    """Area-weighted mean of x over the points with weight > 0 (values elsewhere ignored)."""
    keep = w > 0
    return float(np.sum(w[keep] * x[keep]) / np.sum(w[keep]))


def anomaly_std(x, clim, w):
    """Area-weighted spatial standard deviation of the anomaly x - clim."""
    a = x - clim
    m = weighted_mean(a, w)
    return float(np.sqrt(weighted_mean((a - m) ** 2, w)))


def variability_ratio(pred, truth, clim, w):
    """Spread of the forecast anomaly over that of the true anomaly (1 = realistic)."""
    return anomaly_std(pred, clim, w) / anomaly_std(truth, clim, w)


def diagnose(run):
    root = open_merged_ranks(REPO / "results" / run, 0)
    s = root.store.samples[0]
    init = root.store.init_times[s]
    target0 = root[f"{s}/{STREAM}/1/target"]
    channels = list(target0.attrs["channels"])
    cols = [channels.index(f) for f in FIELDS]
    mask, w = masks_and_weights(load_mesh(MESH), STREAM, channels)
    lat = np.asarray(target0["coords"])[:, 0]
    weights = {
        band: (w[cols] * mask[cols] * band_mask(lat, bounds)[None, :]).astype(np.float64)
        for band, bounds in BANDS.items()
    }
    clim = ClimLookup(CLIM, channels)
    leads = sorted(int(k) for k in root[f"{s}/{STREAM}"].group_keys() if int(k) > 0)

    pred_mean = np.full((len(leads), len(BANDS), len(FIELDS)), np.nan)
    true_mean = np.full_like(pred_mean, np.nan)
    ratio = np.full((len(leads), len(FIELDS)), np.nan)
    for i, lead in enumerate(leads):
        truth = np.asarray(root[f"{s}/{STREAM}/{lead}/target"]["data"])[:, cols].T
        pred = np.asarray(root[f"{s}/{STREAM}/{lead}/prediction"]["data"])[:, cols, 0].T
        c = clim.get(init + np.timedelta64(lead, "D"))[:, cols].T
        for j, wb in enumerate(weights.values()):
            for k in range(len(FIELDS)):
                pred_mean[i, j, k] = weighted_mean(pred[k], wb[k])
                true_mean[i, j, k] = weighted_mean(truth[k], wb[k])
        for k in range(len(FIELDS)):
            ratio[i, k] = variability_ratio(pred[k], truth[k], c[k], weights["global"][k])

    dims = ("lead", "band", "field")
    return xr.Dataset(
        {
            "pred_mean": (dims, pred_mean),
            "true_mean": (dims, true_mean),
            "variability_ratio": (("lead", "field"), ratio),
        },
        coords={"lead": leads, "band": list(BANDS), "field": list(FIELDS)},
        attrs={"run": run, "init_time": str(init)},
    )


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(diags):
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    first = diags[0]
    fig, axes = plt.subplots(len(FIELDS), len(BANDS), figsize=(13, 15), constrained_layout=True)
    for r, field in enumerate(FIELDS):
        for c, band in enumerate(BANDS):
            ax = axes[r, c]
            truth = first["true_mean"].sel(field=field, band=band)
            ax.plot(first.lead, truth, color=INK, lw=1.2, label="truth")
            for ds, colour in zip(diags, COLOURS, strict=False):
                ax.plot(ds.lead, ds["pred_mean"].sel(field=field, band=band), color=colour,
                        lw=1.6, label=ds.attrs["run"])
            ax.set_title(f"{field}, {band}", loc="left")
            _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Area-weighted means against the truth, one-year forecast from 2209-01-01",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "regional_means.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, field in zip(axes.flat, FIELDS, strict=True):
        ax.axhspan(0.7, 1.3, color=GRID, alpha=0.6, lw=0)
        ax.axhline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)))
        for ds, colour in zip(diags, COLOURS, strict=False):
            ax.plot(ds.lead, ds["variability_ratio"].sel(field=field), color=colour, lw=1.6,
                    label=ds.attrs["run"])
        ax.set_title(field, loc="left")
        ax.set_xlabel("lead time (days)")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Variability ratio: forecast anomaly spread / true anomaly spread "
                 "(shaded 0.7-1.3)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "variability_ratio.png", dpi=150)
    plt.close(fig)


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    diags = []
    for run in runs:
        ds = diagnose(run)
        ds.to_netcdf(OUT / f"diag_{run}.nc")
        diags.append(ds)
        print(f"{run}: {len(ds.lead)} leads -> {OUT / f'diag_{run}.nc'}")
    plot(diags)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
