"""
TEST 3 of the instab tests: 150-day rollouts with the inference latent filter vs the
unfiltered one-year rollouts yr1fv002 / yr1fv003.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_filter.py \
        compute RUN [RUN ...] [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_filter.py report

compute  daily global means / RMSE of a filtered run with diag_fvars_year's per-lead code
         (same definitions: wet-area weights, levels, heat content) -> DATA/part1_daily_RUN.nc
report   tables (RMSE, mean error, lag-1 autocorrelation of the daily increments of the
         global-mean errors, blow-up days) -> OUT/test3_filter.txt, figures -> FIG/
The unfiltered reference series are /home/a/a270225/claude_api_tools/diag_yr1/part1_daily_yr1fv00X.nc.
"""

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import diag_fvars_year as dfy  # noqa: E402
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402

MAIN = Path("/work/ab0995/a270225/WeatherGenerator2")
OUT = Path("/home/a/a270225/claude_api_tools/instab_tests")
DATA = OUT / "data"
FIG = MAIN / "paper_eval/figures/instab_tests"
REF = Path("/home/a/a270225/claude_api_tools/diag_yr1")

FILTERS = {"rx10": "relax nu=0.1", "rx25": "relax nu=0.25",
           "sd10": "second_difference nu=0.1", "sd25": "second_difference nu=0.25"}
RMSE_CH = ("salt_75", "salt_125", "temp_2", "temp_450", "ssh")
MEAN_Q = ("salt_75", "salt_125", "temp_2", "temp_450", "ssh_mean", "hc_0_2150", "vmS_0_200")
AC_Q = ("salt_75", "salt_125", "temp_450", "vmS_0_200", "hc_0_2150", "temp_2", "ssh_mean")
DAYS = (1, 2, 3, 5, 10, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150)
WINDOWS = ((2, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 105), (106, 120),
           (121, 150))
THRESH = (("salt_75", 0.5), ("salt_75", 1.0), ("temp_2", 2.0), ("temp_450", 1.0))


def runs_of(m):
    return [f"yr1fv00{m}"] + [f"ins{m}{k}" for k in FILTERS]


def label(run):
    return "no filter" if run.startswith("yr1") else FILTERS[run[4:]]


# ── compute ──────────────────────────────────────────────────────────────────────────
def compute(run, workers, tmp):
    m = run[3]
    cfg = yaml.safe_load((WT / f"paper_eval/configs/yr1fv00{m}.yaml").read_text())
    cfg["run_id"], cfg["results_dir"] = run, str(MAIN / "results" / run)
    cfg_path = Path(tmp) / f"{run}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    s = root.store.samples[0]
    init = np.datetime64(root[f"{s}/TEMP/1/target/times"][0]) - np.timedelta64(1, "D")
    leads = sorted(int(k) for k in root[f"{s}/TEMP"].group_keys() if int(k) > 0)
    pred = truth = rmse = None
    nonfinite = np.zeros(len(leads), np.int64)
    with ProcessPoolExecutor(workers, initializer=dfy._init_worker,
                             initargs=(str(cfg_path),)) as ex:
        for i, (lead, p, t, r, nf, _err, _aice, level_ch) in enumerate(ex.map(dfy._lead, leads)):
            assert lead == leads[i]
            if pred is None:
                qnames = dfy.quantity_names(level_ch)
                rnames = level_ch + ["ssh", "a_ice", "m_ice"]
                pred = np.full((len(leads), len(qnames)), np.nan)
                truth, rmse = pred.copy(), np.full((len(leads), len(rnames)), np.nan)
            pred[i], truth[i], rmse[i], nonfinite[i] = p, t, r, nf
    valid = np.array([init + np.timedelta64(L, "D") for L in leads], "datetime64[ns]")
    ds = xr.Dataset(
        {"pred": (("lead", "quantity"), pred), "truth": (("lead", "quantity"), truth),
         "rmse": (("lead", "channel"), rmse), "n_nonfinite_pred_wet": (("lead",), nonfinite)},
        coords={"lead": leads, "quantity": qnames, "channel": rnames,
                "valid_time": ("lead", valid)},
        attrs={"run": run, "init_time": str(init), "definitions": dfy.DEFINITIONS},
    )
    DATA.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(DATA / f"part1_daily_{run}.nc")
    print(f"{run}: {len(leads)} leads -> {DATA / f'part1_daily_{run}.nc'}", flush=True)


# ── report ───────────────────────────────────────────────────────────────────────────
def load(run, max_lead=150):
    path = REF / f"part1_daily_{run}.nc" if run.startswith("yr1") else DATA / f"part1_daily_{run}.nc"
    ds = xr.open_dataset(path)
    return ds.sel(lead=ds.lead[ds.lead <= max_lead])


def extension(lines):
    """Runs continued past 150 days (same rollout, resumed): full-length comparison."""
    ext_days = (150, 180, 210, 240, 270, 300, 330, 363)
    for m in (3, 2):
        runs = [r for r in runs_of(m)[1:] if (DATA / f"part1_daily_{r}.nc").exists()]
        runs = [r for r in runs if int(load(r, 999).lead.max()) > 150]
        if not runs:
            continue
        ref = load(f"yr1fv00{m}", 999)
        lines += [f"===== fvars00{m}: runs continued past day 150 =====", ""]
        for r in runs:
            ds = load(r, 999)
            lead = ds.lead.values
            lines.append(f"{r} ({label(r)}), leads 1-{int(lead.max())}")
            row = "  blow-up days:"
            for c, x in THRESH:
                d = first_exceed(ds.rmse.sel(channel=c).values, lead, x)
                dref = first_exceed(ref.rmse.sel(channel=c).values, ref.lead.values, x)
                row += f"  {c}>{x:g}: {d} (no filter {dref})"
            lines.append(row)
            for c in RMSE_CH:
                sc = 1e3 if c == "ssh" else 1.0
                lines.append(f"  RMSE {c:<9}" + "".join(
                    f"{'d' + str(d)}={float(ds.rmse.sel(channel=c, lead=d)) * sc:.3g}"
                    f"({float(ref.rmse.sel(channel=c, lead=d)) * sc:.3g})  "
                    for d in ext_days if d in lead))
            for q in ("salt_75", "hc_0_2150", "vmS_0_200", "ssh_mean"):
                sc, unit = scale_unit(q)
                d, dr = err(ds, q) * sc, err(ref, q) * sc
                lines.append(f"  mean err {q} ({unit}) " + "".join(
                    f"d{x}={d[lead == x][0]:.3g}({dr[ref.lead.values == x][0]:.3g})  "
                    for x in ext_days if x in lead))
            lines.append("  ac1 salt_75 mean error, windows 151-180 .. 331-363: " + ", ".join(
                f"{ac1(err(ds, 'salt_75'), lead, a, b):.2f}"
                for a, b in ((151, 180), (181, 210), (211, 240), (241, 270), (271, 300),
                             (301, 330), (331, 363)) if lead.max() >= b))
            lines.append("  (in brackets: the unfiltered yr1fv00X at the same lead)")
            lines.append("")


def err(ds, q):
    return (ds.pred.sel(quantity=q) - ds.truth.sel(quantity=q)).values


def scale_unit(q):
    if q.startswith("hc_"):
        return 1e-21, "ZJ"
    if q.startswith("ssh"):
        return 1e3, "mm"
    if q.startswith(("temp", "vmT")):
        return 1.0, "K"
    return 1.0, "psu"


def ac1(d, lead, a, b):
    sel = (lead >= a) & (lead <= b)
    inc = np.diff(d[sel])
    return np.corrcoef(inc[:-1], inc[1:])[0, 1] if inc.size > 3 else np.nan


def hp_rms(d, lead, a, b):
    hp = np.full_like(d, np.nan)
    hp[1:-1] = d[1:-1] - 0.5 * (d[:-2] + d[2:])
    sel = (lead >= a) & (lead <= b)
    return np.sqrt(np.nanmean(hp[sel] ** 2))


def first_exceed(v, lead, x):
    idx = np.nonzero(v > x)[0]
    return int(lead[idx[0]]) if idx.size else None


def efold(v, lead, a, b):
    sel = (lead >= a) & (lead <= b)
    slope = np.polyfit(lead[sel], np.log(v[sel]), 1)[0]
    return 1.0 / slope if slope > 0 else np.inf


def report():
    lines = ["TEST 3: inference latent filter, 150-day rollouts from 2209-01-01",
             "Definitions: RMSE = wet-area-weighted global RMSE of prediction - truth; mean error =",
             "  global wet-area mean (or integral) of prediction minus that of the truth, as in",
             "  diag_fvars_year.py (hc_0_2150 in ZJ, ssh_mean in mm); ac1 = lag-1 autocorrelation of",
             "  the daily increments d(L) - d(L-1) of the global-mean error d over the lead window",
             "  (-0.5 for white noise, -1 for a pure period-2 alternation, > 0 for a smooth drift);",
             "  hp = RMS over the window of d(L) - (d(L-1) + d(L+1))/2. Blow-up days: first lead at",
             "  which the channel's RMSE exceeds the threshold. e-fold: 1/slope of log RMSE(salt_75)",
             "  fitted over leads 30-80 (days). Reference: yr1fv00X (unfiltered, same initial date).",
             ""]
    FIG.mkdir(parents=True, exist_ok=True)
    for m in (3, 2):
        runs = [r for r in runs_of(m) if r.startswith("yr1") or (DATA / f"part1_daily_{r}.nc").exists()]
        if len(runs) == 1:
            continue
        dss = {r: load(r) for r in runs}
        lines += [f"===== fvars00{m} =====", ""]
        lines.append("Blow-up days and pre-onset growth")
        head = f"  {'run':<10}{'filter':<26}" + "".join(
            f"{f'{c}>{x:g}':>15}" for c, x in THRESH) + f"{'efold30-80':>12}{'maxlead':>9}"
        lines += [head]
        for r in runs:
            ds = dss[r]
            lead = ds.lead.values
            row = f"  {r:<10}{label(r):<26}"
            for c, x in THRESH:
                d = first_exceed(ds.rmse.sel(channel=c).values, lead, x)
                row += f"{('none' if d is None else d):>15}"
            row += f"{efold(ds.rmse.sel(channel='salt_75').values, lead, 30, 80):>12.1f}"
            row += f"{int(lead.max()):>9}"
            lines.append(row)
        lines.append("")
        for c in RMSE_CH:
            sc, unit = (1e3, "mm") if c == "ssh" else (1.0, "K" if c.startswith("temp") else "psu")
            lines.append(f"RMSE {c} ({unit})")
            lines.append(f"  {'run':<10}" + "".join(f"{'d' + str(d):>8}" for d in DAYS))
            for r in runs:
                ds = dss[r]
                v = ds.rmse.sel(channel=c)
                lines.append(f"  {r:<10}" + "".join(
                    f"{float(v.sel(lead=d)) * sc:>8.3g}" if d in ds.lead.values else f"{'-':>8}"
                    for d in DAYS))
            lines.append("")
        for q in MEAN_Q:
            sc, unit = scale_unit(q)
            lines.append(f"Mean error {q} ({unit})")
            lines.append(f"  {'run':<10}" + "".join(f"{'d' + str(d):>8}" for d in DAYS))
            for r in runs:
                ds = dss[r]
                d = err(ds, q) * sc
                lead = ds.lead.values
                lines.append(f"  {r:<10}" + "".join(
                    f"{d[lead == x][0]:>8.3g}" if x in lead else f"{'-':>8}" for x in DAYS))
            lines.append("")
        lines.append("ac1 of the daily increments of the global-mean errors")
        lines.append(f"  {'quantity':<12}{'run':<10}" + "".join(f"{f'{a}-{b}':>9}" for a, b in WINDOWS))
        for q in AC_Q:
            for r in runs:
                ds = dss[r]
                d, lead = err(ds, q), ds.lead.values
                lines.append(f"  {q:<12}{r:<10}" + "".join(f"{ac1(d, lead, a, b):>9.2f}"
                                                         for a, b in WINDOWS))
            lines.append("")
        lines.append("hp RMS of the global-mean errors (2-day high-pass)")
        lines.append(f"  {'quantity':<16}{'run':<10}" + "".join(f"{f'{a}-{b}':>9}" for a, b in WINDOWS))
        for q in ("hc_0_2150", "salt_75", "vmS_0_200"):
            sc, unit = scale_unit(q)
            for r in runs:
                ds = dss[r]
                d, lead = err(ds, q) * sc, ds.lead.values
                lines.append(f"  {q + ' (' + unit + ')':<16}{r:<10}" + "".join(
                    f"{hp_rms(d, lead, a, b):>9.3g}" for a, b in WINDOWS))
            t = dss[runs[0]].truth.sel(quantity=q).values * sc
            lines.append(f"  {q + ' truth':<16}{'':<10}" + "".join(
                f"{hp_rms(t, dss[runs[0]].lead.values, a, b):>9.3g}" for a, b in WINDOWS))
            lines.append("")
        figure(m, runs, dss)
    extension(lines)
    txt = "\n".join(lines) + "\n"
    (OUT / "test3_filter.txt").write_text(txt)
    print(txt)


def figure(m, runs, dss):
    colors = {"yr1": "k", "rx10": "tab:blue", "rx25": "tab:cyan", "sd10": "tab:red",
              "sd25": "tab:orange"}
    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, c in zip(axs.flat[:5], RMSE_CH):
        for r in runs:
            ds = dss[r]
            sc = 1e3 if c == "ssh" else 1.0
            ax.semilogy(ds.lead, ds.rmse.sel(channel=c) * sc, color=colors[r[:3] if r.startswith("yr1") else r[4:]],
                        label=label(r), lw=1.2)
        ax.set_title(f"RMSE {c}" + (" (mm)" if c == "ssh" else ""))
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=8)
    for ax, q in zip(axs.flat[5:7], ("salt_75", "hc_0_2150")):
        sc, unit = scale_unit(q)
        for r in runs:
            ds = dss[r]
            ax.plot(ds.lead, err(ds, q) * sc, color=colors[r[:3] if r.startswith("yr1") else r[4:]], lw=1.0)
        ax.set_title(f"global-mean error {q} ({unit})")
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    ax = axs.flat[7]
    mids = [0.5 * (a + b) for a, b in WINDOWS]
    for r in runs:
        ds = dss[r]
        d, lead = err(ds, "salt_75"), ds.lead.values
        ax.plot(mids, [ac1(d, lead, a, b) for a, b in WINDOWS], "o-",
                color=colors[r[:3] if r.startswith("yr1") else r[4:]])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("ac1 of daily increments, salt_75 mean error")
    ax.set_xlabel("lead window centre (days)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    fig.suptitle(f"fvars00{m}: latent filter vs unfiltered rollout from 2209-01-01")
    fig.savefig(FIG / f"test3_filter_fvars00{m}.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("runs", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "compute":
        DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=DATA) as tmp:
            for r in a.runs:
                compute(r, a.workers, tmp)
    else:
        report()


if __name__ == "__main__":
    main()
