"""Round-3 measurements for the 2209 review, from the regional series (regional_diag.py).

    python scripts/instab/y2209_round3.py taugroup    # effective year by channel group, monthly
    python scripts/instab/y2209_round3.py seasonage   # deep-bias seasonal cycle per model year
    python scripts/instab/y2209_round3.py trigger     # request-9 experiments vs fvnn10y

taugroup: tau (y2209_requests.tau_series, 2150-2209 truth fit, significant-trend channels only)
of prediction and truth per channel group and month, lag = tau(truth) - tau(pred), and per
model year the Apr->Jul, Jul->Sep, Sep->Jan(+1), Jan->Apr lag increments.
seasonage: per run, model year, region and level (975, 2035 m) the monthly mean bias
(pred - truth) minus its annual mean: half range, months of max and min, and the half range in
years of the 2150-2199 truth trend.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import y2209_requests as yr  # noqa: E402

DATA = Path.home() / "claude_api_tools/y2209/data"
OUT = Path.home() / "claude_api_tools/y2209/in3"
RUNS = ("fvnn10y", "fvnnf09", "fvnnf00", "fvnn07y", "fvnn08y", "fvnn01y1", "fvnn02y1", "fvnn04y1",
        "fvfs02y", "fvfs09y")
ATL = ("n_atlantic", "trop_atlantic", "s_atlantic")
PACIND = ("n_pacific", "s_pacific", "indian")
GROUPS = {
    "all_ge450": lambda st, r, d: True,
    "Med_S": lambda st, r, d: st == "SALT" and r == "mediterranean",
    "Atlantic_T": lambda st, r, d: st == "TEMP" and r in ATL,
    "Atlantic_S": lambda st, r, d: st == "SALT" and r in ATL,
    "PacInd_T": lambda st, r, d: st == "TEMP" and r in PACIND,
    "PacInd_S": lambda st, r, d: st == "SALT" and r in PACIND,
    "Arctic_T": lambda st, r, d: st == "TEMP" and r == "arctic",
    "Southern_T": lambda st, r, d: st == "TEMP" and r == "southern",
    "band_450_735": lambda st, r, d: 440 <= d <= 740 and r != "global",
    "band_850_1255": lambda st, r, d: 840 <= d <= 1260 and r != "global",
    "band_1415_2035": lambda st, r, d: 1400 <= d and r != "global",
}


def monthly(run):
    ds = xr.open_dataset(DATA / f"rollout_regional_{run}.nc").load()
    t = pd.DatetimeIndex(ds.valid_time.values)
    grp = t.to_period("M").astype(str)
    pred, truth = {}, {}
    for st in ("TEMP", "SALT"):
        for reg in ds.region.values:
            for d in ds.depth.values:
                k = (st, str(reg), float(d))
                pred[k] = pd.Series(ds[f"{st}_pred"].sel(region=reg, depth=d).values,
                                    index=grp).groupby(level=0).mean()
                truth[k] = pd.Series(ds[f"{st}_truth"].sel(region=reg, depth=d).values,
                                     index=grp).groupby(level=0).mean()
    return pred, truth


def taugroup(a):
    fit = yr.truth_fit(DATA)
    rows = []
    for run in RUNS:
        if not (DATA / f"rollout_regional_{run}.nc").exists():
            print("missing", run)
            continue
        pred, truth = monthly(run)
        months = next(iter(pred.values())).index
        for g, sel in GROUPS.items():
            keys = [k for k in pred if sel(*k)]
            if not keys:
                continue
            sp, used, _ = yr.tau_series({k: pred[k].values for k in keys}, fit)
            stt, _, _ = yr.tau_series({k: truth[k].values for k in keys}, fit)
            if not used:
                continue
            for i, m in enumerate(months):
                rows.append({"run": run, "group": g, "month": m, "n_channels": len(used),
                             "pred_tau": sp["tau_median"][i], "truth_tau": stt["tau_median"][i],
                             "lag": stt["tau_median"][i] - sp["tau_median"][i]})
    df = pd.DataFrame(rows)
    inc = []
    for (run, g), d in df.groupby(["run", "group"]):
        lag = d.set_index("month")["lag"]
        for y in sorted({int(m[:4]) for m in lag.index}):
            def L(mm, yy=y):
                return lag.get(f"{yy}-{mm:02d}", np.nan)
            inc.append({"run": run, "group": g, "year": y, "lag_Apr": L(4), "lag_Jul": L(7),
                        "lag_Sep": L(9), "Apr_to_Jul": L(7) - L(4), "Jul_to_Sep": L(9) - L(7),
                        "Sep_to_Jan": L(1, y + 1) - L(9), "Jan_to_Apr": L(4) - L(1)})
    with open(OUT / "tau_by_group_monthly.csv", "w") as fh:
        fh.write("# Effective year tau by channel group, monthly means (same method as "
                 "effective_year_series.csv, restricted to the group's significant-trend channels "
                 "at >= 450 m). lag = truth tau - pred tau (years). Groups: Med_S, Atlantic_T/S "
                 "(n/trop/s_atlantic), PacInd_T/S (n/s_pacific, indian), Arctic_T, Southern_T, and "
                 "level bands over all regions except global.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    with open(OUT / "tau_by_group_seasonal_increments.csv", "w") as fh:
        fh.write("# Per run, group and model year: lag (years) in Apr, Jul, Sep and its changes "
                 "Apr->Jul, Jul->Sep, Sep->Jan(next year), Jan->Apr (positive = the prediction falls "
                 "further behind the truth along the trend).\n")
        pd.DataFrame(inc).to_csv(fh, index=False, float_format="%.3g")
    print(pd.DataFrame(inc).query("run == 'fvnn10y' and group in ['Med_S', 'Atlantic_T', "
                                  "'PacInd_S', 'Arctic_T']").round(1).to_string())


def seasonage(a):
    fit = yr.truth_fit(DATA)
    rows = []
    for run in RUNS:
        p = DATA / f"rollout_regional_{run}.nc"
        if not p.exists():
            print("missing", run)
            continue
        ds = xr.open_dataset(p).load()
        t = pd.DatetimeIndex(ds.valid_time.values)
        dep = ds.depth.values
        for st in ("TEMP", "SALT"):
            b = ds[f"{st}_pred"] - ds[f"{st}_truth"]
            for reg in ds.region.values:
                for d0 in (975.0, 2035.0):
                    d = float(dep[np.argmin(np.abs(dep - d0))])
                    s = pd.Series(b.sel(region=reg, depth=d).values, index=t)
                    if not np.isfinite(s.values).any():
                        continue
                    f = fit.get((st, str(reg), d))
                    for y, sy in s.groupby(s.index.year):
                        mon = sy.groupby(sy.index.month).mean()
                        if len(mon) < 12:
                            continue
                        mon = mon - mon.mean()
                        amp = (mon.max() - mon.min()) / 2
                        rows.append({"run": run, "year": y, "var": st, "region": reg, "depth": d,
                                     "amp": amp, "month_max": int(mon.idxmax()),
                                     "month_min": int(mon.idxmin()),
                                     "amp_years_of_trend": amp / abs(f["b"]) if f else np.nan})
    df = pd.DataFrame(rows)
    with open(OUT / "seasonal_leakage_by_year.csv", "w") as fh:
        fh.write("# Per run and model year (complete years only): monthly mean regional bias "
                 "(pred - truth) minus its annual mean at 975 and 2035 m; amp = half range, months "
                 "of max/min, and amp in years of the 2150-2209 truth trend (amp / |trend per yr|).\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.query("region == 'global'").pivot_table(index=["run", "year"], columns=["var", "depth"],
                                                     values="amp_years_of_trend").round(1))


TRIGGER = ("fvtr_fp32", "fvtr_fs08", "fvtr_fs12", "fvtr_nz3s0", "fvtr_nz3s1", "fvtr_nz2s0",
           "fvtr_anom0", "fvtr_f2208", "fvtr_anom0a", "fvtr_f2208a", "fvtr_anom08a",
           "fvtr_sa12", "fvtr_sc12")


def trigger(a):
    """Request-9 experiments branched from fvnn10y's latent at step 3450 (2209-06-13) or 3390
    (2209-04-14): regional bias over the step window, half-change date, and the lag (tau) by
    group, each against fvnn10y over the same dates."""
    fit = yr.truth_fit(DATA)
    ref = xr.open_dataset(DATA / "rollout_regional_fvnn10y.nc").load()
    dep = ref.depth.values
    lev = lambda d: float(dep[np.argmin(np.abs(dep - d))])  # noqa: E731
    rows, series = [], []
    for run in ("fvnn10y",) + TRIGGER:
        p = DATA / f"rollout_regional_{run}.nc"
        if not p.exists():
            print("missing", run)
            continue
        ds = xr.open_dataset(p).load()
        t = pd.DatetimeIndex(ds.valid_time.values)
        start = "2209-04-14" if run.endswith("a") else "2209-06-13"
        if run == "fvnn10y":
            start = "2209-06-13"
        r = {"run": run, "branch_date": start if run != "fvnn10y" else "-"}
        sel = (t > "2209-03-01") & (t <= "2209-11-10")
        for st, reg, d in (("TEMP", "global", 975.0), ("TEMP", "global", 2035.0),
                           ("TEMP", "n_atlantic", 975.0), ("TEMP", "arctic", 2035.0),
                           ("SALT", "mediterranean", 975.0), ("SALT", "global", 975.0),
                           ("TEMP", "global", 2.0), ("TEMP", "global", 107.0)):
            b = (ds[f"{st}_pred"] - ds[f"{st}_truth"]).sel(region=reg, depth=lev(d)).to_series()
            b.index = t
            rb = (ref[f"{st}_pred"] - ref[f"{st}_truth"]).sel(region=reg, depth=lev(d)).to_series()
            rb.index = pd.DatetimeIndex(ref.valid_time.values)
            b0 = rb.loc["2209-06-10":"2209-06-16"].mean()  # same pre-step value for all
            b1 = b.loc["2209-10-01":"2209-11-10"].mean()
            key = f"{st[0]}{int(d)}_{reg}"
            r[f"bias_Jun13_{key}"] = b0
            r[f"bias_Oct1-Nov10_{key}"] = b1
            r[f"change_{key}"] = b1 - b0
            if key == "T975_global":
                sm = b.rolling(9, center=True).mean()
                half = b0 + 0.5 * (rb.loc["2209-10-01":"2209-11-10"].mean() - b0)
                cross = sm.loc["2209-06-13":][(sm.loc["2209-06-13":] - half) <= 0]
                r["half_ref_change_date_T975_global"] = str(cross.index[0].date()) if len(cross) else "none"
            if key in ("T975_global", "S975_mediterranean"):
                for dt, v in b[sel].iloc[::(1 if key == "S975_mediterranean" else 5)].items():
                    series.append({"run": run, "date": str(dt.date()), "series": key, "bias": v})
        # surface / upper-ocean RMSE (degradation check), mean over Oct 1 - Nov 10 and over
        # the first 30 days after the branch
        for d in (2.0, 107.0):
            e = ds["TEMP_rmse"].sel(region="global", depth=lev(d)).to_series()
            e.index = t
            r[f"rmse_T{int(d)}_global_Oct1-Nov10"] = e.loc["2209-10-01":"2209-11-10"].mean()
            r[f"rmse_T{int(d)}_global_first30d"] = e.loc[start:].iloc[1:31].mean()
        # lag (all channels >= 450 m) monthly Jun-Nov 2209
        grp = t.to_period("M").astype(str)
        pred = {}
        truth = {}
        for st in ("TEMP", "SALT"):
            for reg in ds.region.values:
                for d in ds.depth.values:
                    k = (st, str(reg), float(d))
                    pred[k] = pd.Series(ds[f"{st}_pred"].sel(region=reg, depth=d).values, index=grp).groupby(level=0).mean()
                    truth[k] = pd.Series(ds[f"{st}_truth"].sel(region=reg, depth=d).values, index=grp).groupby(level=0).mean()
        months = next(iter(pred.values())).index
        sp, _, _ = yr.tau_series({k: v.values for k, v in pred.items()}, fit)
        stt, _, _ = yr.tau_series({k: v.values for k, v in truth.items()}, fit)
        for m in ("2209-04", "2209-05", "2209-06", "2209-07", "2209-08", "2209-09", "2209-10", "2209-11"):
            if m in months:
                i = list(months).index(m)
                r[f"lag_{m}"] = stt["tau_median"][i] - sp["tau_median"][i]
        rows.append(r)
    df = pd.DataFrame(rows)
    with open(OUT / "trigger_experiments.csv", "w") as fh:
        fh.write("# Request-9 experiments: rollouts branched from fvnn10y's exact latent (fvnnre states, "
                 "bit-identical to fvnn10y) at step 3450 (2209-06-13) or 3390 (2209-04-14, runs *a), "
                 "one change each, to 2209-11-10. bias = regional pred - truth; bias_Jun13 = fvnn10y "
                 "mean 2209-06-10..16 (common pre-step value); change = mean(Oct 1 - Nov 10) - "
                 "bias_Jun13. half_ref_change_date = first date on/after Jun 13 at which the 9-day "
                 "smoothed global T975 bias reaches half of fvnn10y's change (none = never). lag_<month> "
                 "= tau(truth) - tau(pred), all significant channels >= 450 m, monthly (months partly "
                 "covered by the run use the days present). fp32 = with_mixed_precision False and "
                 "attention_dtype fp32; fs08/fs12 = forcing_scale; nz3s0/nz3s1/nz2s0 = fe_noise token_rms "
                 "1e-3 seeds 0/1 and 1e-2 seed 0; anom0 = forcing_split alpha_anom 0 with C = 2209 time "
                 "mean of the injection (anom08a: C = 2208 mean); f2208 = forcing read from the same calendar day of 2208. rmse_T2 / rmse_T107: global RMSE (degradation check).\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    with open(OUT / "trigger_series.csv", "w") as fh:
        fh.write("# Request-9 experiments: global T975 bias every 5th day and Mediterranean S975 bias daily, "
                 "2209-03..11, per run (fvnn10y = reference).\n")
        pd.DataFrame(series).to_csv(fh, index=False, float_format="%.4g")
    cols = ["run", "change_T975_global", "change_S975_mediterranean", "change_T2035_global",
            "half_ref_change_date_T975_global"] + [c for c in df.columns if c.startswith("lag_")]
    print(df[[c for c in cols if c in df.columns]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["taugroup", "seasonage", "trigger"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.what](a)


if __name__ == "__main__":
    main()
