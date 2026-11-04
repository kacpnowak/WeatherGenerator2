"""Measurements requested by the 2209 review (round 1), from stored outputs and archives.

    python scripts/instab/y2209_requests.py lambda      # effective-year series per run
    python scripts/instab/y2209_requests.py stepreg     # step change vs truth trend, per level
    python scripts/instab/y2209_requests.py seasonal    # seasonal cycle of the deep bias
    python scripts/instab/y2209_requests.py pattern     # node-level pattern test (archive)
    python scripts/instab/y2209_requests.py bench       # persistence / climatology benchmarks
    python scripts/instab/y2209_requests.py integrity   # archive calendar, columns, NaNs
        [--workers 24] [--data ~/claude_api_tools/y2209/data] [--out ~/claude_api_tools/y2209/in2]

Effective year: for each region x {T, S} x level >= 450 m with a trend-dominated truth (|trend
per decade| > 3 x detrended interannual std over 2150-2199), invert the truth's linear 2150-2209
fit at the predicted value, tau_c = 2180 + (pred - a_c) / b_c. The joint tau is the median over
channels, weighted tau the mean with weights (b_c / std_c)^2; IQR is reported.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr
from zarr.storage import ZipStore

MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
ARCHIVE = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
START = np.datetime64("2200-01-01")


def _channels(stream):
    return yaml.safe_load(open(f"config/streams/fesom_vars/{stream.lower()}.yml"))[stream]["target"]


def truth_fit(data):
    """Per (var, region, depth): linear fit of annual truth means 2150-2209 and the trend ratio."""
    tr = xr.open_dataset(data / "truth_regional_2010_2209.nc").load()
    yrs = pd.DatetimeIndex(tr.time.values).year
    out = {}
    for st in ("TEMP", "SALT"):
        ann = tr[st].assign_coords(year=("time", yrs)).groupby("year").mean()
        for reg in tr.region.values:
            for d in tr.depth.values:
                s = ann.sel(region=reg, depth=d).to_series()
                y = s.loc[2150:2209]
                if y.isna().any():
                    continue
                b, a = np.polyfit(y.index.values - 2180.0, y.values, 1)
                fit50 = s.loc[2150:2199]
                bb, aa = np.polyfit(fit50.index.values - 2175.0, fit50.values, 1)
                resid = fit50.values - (aa + bb * (fit50.index.values - 2175.0))
                out[(st, str(reg), float(d))] = dict(a=a, b=b, std=np.std(resid, ddof=1),
                                                     trend_dec=bb * 10)
    return out


def tau_series(pred, fit, min_depth=440.0):
    """pred: dict (st, reg, depth) -> array over time. Returns per-time tau stats."""
    keys = [k for k in pred if k in fit and k[2] >= min_depth
            and abs(fit[k]["trend_dec"]) > 3 * fit[k]["std"]]
    taus = np.stack([2180.0 + (pred[k] - fit[k]["a"]) / fit[k]["b"] for k in keys])
    w = np.array([(fit[k]["b"] / fit[k]["std"]) ** 2 for k in keys])[:, None]
    ok = np.isfinite(taus)
    wmean = np.nansum(taus * w, 0) / np.sum(w * ok, 0)
    return dict(tau_median=np.nanmedian(taus, 0), tau_wmean=wmean,
                tau_q25=np.nanpercentile(taus, 25, 0), tau_q75=np.nanpercentile(taus, 75, 0),
                n_channels=ok.sum(0)), keys, taus


def lambda_cmd(a):
    fit = truth_fit(a.data)
    rows = []
    for run, freq in (("fvnn10y", "M"), ("fvnnf09", "30D"), ("fvnnf00", "30D"),
                      ("fvnn07y", "M"), ("fvnn08y", "M")):
        p = a.data / f"rollout_regional_{run}.nc"
        if not p.exists():
            print("missing", p)
            continue
        ds = xr.open_dataset(p).load()
        t = pd.DatetimeIndex(ds.valid_time.values)
        grp = t.to_period("M").astype(str) if freq == "M" else (np.arange(len(t)) // 30)
        for st in ("TEMP", "SALT"):
            ds[f"{st}_pred"].attrs = {}
        pred, truth = {}, {}
        for st in ("TEMP", "SALT"):
            for reg in ds.region.values:
                for d in ds.depth.values:
                    pred[(st, str(reg), float(d))] = pd.Series(
                        ds[f"{st}_pred"].sel(region=reg, depth=d).values, index=grp
                    ).groupby(level=0).mean().values
                    truth[(st, str(reg), float(d))] = pd.Series(
                        ds[f"{st}_truth"].sel(region=reg, depth=d).values, index=grp
                    ).groupby(level=0).mean().values
        dates = pd.Series(t, index=grp).groupby(level=0).first().values
        sp, _, _ = tau_series(pred, fit)
        stt, _, _ = tau_series(truth, fit)
        for i, d in enumerate(dates):
            yr = pd.Timestamp(d).year + (pd.Timestamp(d).dayofyear - 0.5) / 365.25
            rows.append({"run": run, "date": str(pd.Timestamp(d).date()), "calendar_year": yr,
                         **{f"pred_{k}": v[i] for k, v in sp.items()},
                         **{f"truth_{k}": v[i] for k, v in stt.items() if k != "n_channels"}})
    df = pd.DataFrame(rows)
    df["lag_years_pred_vs_truth_median"] = df.truth_tau_median - df.pred_tau_median
    with open(a.out / "effective_year_series.csv", "w") as fh:
        fh.write("# Effective year tau (see y2209_requests.py docstring): per run and period, tau "
                 "of the prediction and of the truth (same channels, same method), median, "
                 "weighted mean, IQR over channels, number of channels. lag = truth tau - pred "
                 "tau (median).\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    print(df.groupby("run").agg({"lag_years_pred_vs_truth_median": ["first", "last", "max"]}))


def stepreg_cmd(a):
    fit = truth_fit(a.data)
    ds = xr.open_dataset(a.data / "rollout_regional_fvnn10y.nc").load()
    t = pd.DatetimeIndex(ds.valid_time.values)
    pre = (t >= "2209-05-01") & (t < "2209-07-01")
    post = t >= "2209-10-01"
    rows = []
    for d in ds.depth.values:
        x, y, lab = [], [], []
        for st in ("TEMP", "SALT"):
            bias = ds[f"{st}_pred"] - ds[f"{st}_truth"]
            for reg in ds.region.values:
                k = (st, str(reg), float(d))
                if k not in fit:
                    continue
                b = bias.sel(region=reg, depth=d).values
                x.append(fit[k]["trend_dec"])
                y.append(np.nanmean(b[post]) - np.nanmean(b[pre]))
                lab.append(f"{st[0]}:{reg}")
        x, y = np.array(x), np.array(y)
        slope = np.sum(x * y) / np.sum(x * x)
        ss_res, ss_tot = np.sum((y - slope * x) ** 2), np.sum(y ** 2)
        rows.append({"depth": d, "n": len(x), "slope_years_through_origin": slope * 10,
                     "r2_uncentred": 1 - ss_res / ss_tot,
                     "corr": np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan,
                     "max_abs_step": np.max(np.abs(y)), "max_abs_trend_dec": np.max(np.abs(x))})
    df = pd.DataFrame(rows)
    with open(a.out / "step_vs_trend_by_level.csv", "w") as fh:
        fh.write("# fvnn10y: per level, across 11 regions x {T, S}: step = bias(Oct-Dec 2209) - "
                 "bias(May-Jun 2209), regressed through the origin on the 2150-2199 truth trend "
                 "per decade. slope in years (negative = retreat along the trend), uncentred R2, "
                 "Pearson corr.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.to_string())


def seasonal_cmd(a):
    ds = xr.open_dataset(a.data / "rollout_regional_fvnn10y.nc").load()
    t = pd.DatetimeIndex(ds.valid_time.values)
    sel = t < "2209-01-01"
    rows = []
    for st in ("TEMP", "SALT"):
        for reg in ds.region.values:
            for d in (450.0, 975.0, 2035.0):
                dd = float(ds.depth.values[np.argmin(np.abs(ds.depth.values - d))])
                b = (ds[f"{st}_pred"] - ds[f"{st}_truth"]).sel(region=reg, depth=dd).values
                tru = ds[f"{st}_truth"].sel(region=reg, depth=dd).values
                if not np.isfinite(b).any():
                    continue
                mb = pd.Series(b[sel], index=t[sel].month).groupby(level=0).mean()
                # truth seasonal cycle after removing a linear trend
                tt = np.arange(sel.sum())
                tr_d = tru[sel] - np.polyval(np.polyfit(tt, tru[sel], 1), tt)
                mt = pd.Series(tr_d, index=t[sel].month).groupby(level=0).mean()
                rows.append({"var": st, "region": reg, "depth": dd,
                             "bias_seasonal_amp": (mb.max() - mb.min()) / 2,
                             "bias_month_max": int(mb.idxmax()), "bias_month_min": int(mb.idxmin()),
                             "bias_mean_2200_2208": mb.mean(),
                             "truth_seasonal_amp_detrended": (mt.max() - mt.min()) / 2,
                             "truth_month_max": int(mt.idxmax())})
    df = pd.DataFrame(rows)
    with open(a.out / "deep_bias_seasonal_cycle.csv", "w") as fh:
        fh.write("# fvnn10y 2200-2208: monthly climatology of the regional bias (pred - truth) "
                 "and of the detrended truth: half range (amp) and months of max/min.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.sort_values("bias_seasonal_amp", ascending=False).head(15).to_string())


# ----- archive-reading parts --------------------------------------------------------------

_G = {}


def _init():
    m = xr.open_dataset(MESH)
    _G["lon"], _G["lat"] = np.asarray(m.lon), np.asarray(m.lat)
    nz = len(_channels("TEMP"))
    _G["wet"] = np.arange(nz)[:, None] < (np.asarray(m.nlevels_nod2D) - 1)[None, :]
    _G["area"] = np.asarray(m.nod_area)[:nz]
    _G["n"] = len(m.lon)
    _G["z"] = zarr.open_group(store=ZipStore(
        "results/fvnn10y/validation_chkpt00000_rank0000.zip", mode="r", read_only=True), mode="r")


def _arch_day(year, month, day, chans):
    """Truth from the yearly archive on (year, month, day); Feb 29 -> Feb 28 in non-leap years."""
    z = zarr.open(str(ARCHIVE / "ocean_node" / f"ocean_node_{year}.zarr"), mode="r")
    try:
        d = np.datetime64(f"{year}-{month:02d}-{day:02d}")
    except ValueError:
        d = np.datetime64(f"{year}-{month:02d}-28")
    i = int((d - np.datetime64(f"{year}-01-01")).astype(int))
    n = _G["n"]
    names = list(z["data"].attrs["colnames"])
    blk = np.asarray(z["data"][i * n:(i + 1) * n], dtype=np.float64)
    dt = str(np.asarray(z["dates"][i * n, 0]))[:10]
    assert dt == str(d), f"archive date {dt} != {d}"
    return blk[:, [names.index(c) for c in chans]]


def _zip(stream, lead, kind):
    x = np.asarray(_G["z"][f"0/{stream}/{lead}/{kind}/data"][...], dtype=np.float64)
    return x[..., 0] if x.ndim == 3 else x


def _date(lead):
    return pd.Timestamp(START + np.timedelta64(lead, "D"))


def pattern_cmd(a):
    _init()
    ct, cs = _channels("TEMP"), _channels("SALT")
    rows = []
    l_a, l_b = 3450, 3600
    l_c0, l_c1 = 134, 3420
    for st, chans in (("TEMP", ct), ("SALT", cs)):
        pa, pb = _zip(st, l_a, "prediction"), _zip(st, l_b, "prediction")
        tc0, tc1 = _zip(st, l_c0, "target"), _zip(st, l_c1, "target")
        pc0, pc1 = _zip(st, l_c0, "prediction"), _zip(st, l_c1, "prediction")
        trend = 0
        for lead in (l_a, l_b):
            d = _date(lead)
            trend = trend + (_arch_day(2199, d.month, d.day, chans)
                             - _arch_day(2179, d.month, d.day, chans)) / 2
        for depth in (450, 975, 2035):
            li = chans.index(f"{st.lower()}_{depth}")
            wet = _G["wet"][li]
            w = _G["area"][li][wet]
            x = trend[wet, li]
            for name, y in (("step: pred(2209-11-10) - pred(2209-06-13)", (pb - pa)[wet, li]),
                            ("drift: [pred - truth](2209-05-15) - [pred - truth](2200-05-15)",
                             ((pc1 - tc1) - (pc0 - tc0))[wet, li])):
                for reg, msk in (("global", np.ones_like(x, bool)),
                                 ("global excl. Mediterranean",
                                  ~(((_G["lon"][wet] > -5.5) & (_G["lon"][wet] < 36.5)
                                     & (_G["lat"][wet] > 30) & (_G["lat"][wet] < 46))))):
                    ww, xx, yy = w[msk], x[msk], y[msk]
                    xm, ym = np.average(xx, weights=ww), np.average(yy, weights=ww)
                    cov = np.average((xx - xm) * (yy - ym), weights=ww)
                    r = cov / np.sqrt(np.average((xx - xm) ** 2, weights=ww)
                                      * np.average((yy - ym) ** 2, weights=ww))
                    rows.append({"var": st, "depth": depth, "quantity": name, "mask": reg,
                                 "pattern_corr_area_weighted": r,
                                 "slope_vs_20yr_trend": cov / np.average((xx - xm) ** 2, weights=ww),
                                 "rms_quantity": np.sqrt(np.average(yy ** 2, weights=ww)),
                                 "rms_20yr_trend_map": np.sqrt(np.average(xx ** 2, weights=ww))})
    df = pd.DataFrame(rows)
    with open(a.out / "node_pattern_test.csv", "w") as fh:
        fh.write("# Node level, area weighted over wet nodes. x = 20-year truth change map, "
                 "truth(2199) - truth(2179) on the calendar days 06-13 and 11-10 (mean of the two). "
                 "slope = regression of the quantity on x (centred, area weighted); a slope of -1 "
                 "means the quantity equals minus the 20-year change.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.to_string())


def _wcorr(xx, yy, ww):
    xm, ym = np.average(xx, weights=ww), np.average(yy, weights=ww)
    cov = np.average((xx - xm) * (yy - ym), weights=ww)
    vx = np.average((xx - xm) ** 2, weights=ww)
    return cov / np.sqrt(vx * np.average((yy - ym) ** 2, weights=ww)), cov / vx


def pattern2_cmd(a):
    """Noise-reduced node pattern test: the 20-year truth change from 5-year means of monthly
    snapshots (2195-2199 minus 2175-2179), the rollout quantities from 21-day means (every 2nd
    day), each correlated at node level and after averaging into 5x5 degree boxes."""
    _init()
    ct, cs = _channels("TEMP"), _channels("SALT")

    def win(st, centre, kind):
        return np.mean([_zip(st, l, kind) for l in range(centre - 10, centre + 11, 2)], 0)

    lon, lat = _G["lon"], _G["lat"]
    box = (np.floor((lat + 90) / 5) * 72 + np.floor(((lon + 180) % 360) / 5)).astype(int)
    med = (lon > -5.5) & (lon < 36.5) & (lat > 30) & (lat < 46)
    rows = []
    for st, chans in (("TEMP", ct), ("SALT", cs)):
        late = np.mean([_arch_day(y, m, 15, chans) for y in range(2195, 2200) for m in range(1, 13)], 0)
        early = np.mean([_arch_day(y, m, 15, chans) for y in range(2175, 2180) for m in range(1, 13)], 0)
        trend = late - early
        pa, pb = win(st, 3450, "prediction"), win(st, 3600, "prediction")
        ta, tb = win(st, 3450, "target"), win(st, 3600, "target")
        p0, p1 = win(st, 134, "prediction"), win(st, 3420, "prediction")
        t0, t1 = win(st, 134, "target"), win(st, 3420, "target")
        qs = {"step_pred: pred(2209-11-10) - pred(2209-06-13)": pb - pa,
              "step_bias: bias(2209-11-10) - bias(2209-06-13)": (pb - tb) - (pa - ta),
              "drift_bias: bias(2209-05-15) - bias(2200-05-15)": (p1 - t1) - (p0 - t0),
              "bias(2209-11-10)": pb - tb}
        for depth in (450, 975, 2035):
            li = chans.index(f"{st.lower()}_{depth}")
            wet = _G["wet"][li] & np.isfinite(trend[:, li])
            for qn, q in qs.items():
                for mname, m in (("global", wet), ("global excl. Mediterranean", wet & ~med)):
                    w = _G["area"][li][m]
                    x, y = trend[m, li], q[m, li]
                    r, sl = _wcorr(x, y, w)
                    # 5x5 degree box means (area weighted)
                    b = box[m]
                    sw = np.bincount(b, w, 72 * 36)
                    ok = sw > 0
                    bx = np.bincount(b, w * x, 72 * 36)[ok] / sw[ok]
                    by = np.bincount(b, w * y, 72 * 36)[ok] / sw[ok]
                    rb, slb = _wcorr(bx, by, sw[ok])
                    rows.append({"var": st, "depth": depth, "quantity": qn, "mask": mname,
                                 "corr_node": r, "slope_node": sl, "corr_5deg_box": rb,
                                 "slope_5deg_box": slb, "rms_quantity": np.sqrt(np.average(y ** 2, weights=w)),
                                 "rms_trend_map": np.sqrt(np.average(x ** 2, weights=w))})
    df = pd.DataFrame(rows)
    with open(a.out / "node_pattern_test_v2.csv", "w") as fh:
        fh.write("# Noise-reduced version of node_pattern_test.csv. Trend map x = 20-year truth change "
                 "from 5-year means of monthly (15th) snapshots, mean(2195-2199) - mean(2175-2179). "
                 "Rollout quantities (fvnn10y) are 21-day means centred on the named date (every 2nd "
                 "day, leads c-10..c+10). bias = pred - truth. Area-weighted correlation and regression "
                 "slope of the quantity on x, at node level and after 5x5 degree box averaging. "
                 "Slope -1 = minus the 20-year change.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.to_string())


def _bench_lead(lead):
    ct, cs = _channels("TEMP"), _channels("SALT")
    d, d1 = _date(lead), _date(1)
    out = {}
    for st, chans in (("TEMP", ct), ("SALT", cs)):
        truth = _zip(st, lead, "target")
        ic = _zip(st, 1, "target")
        pred = _zip(st, lead, "prediction")
        clim_l = np.mean([_arch_day(y, d.month, d.day, chans) for y in range(2190, 2200)], 0)
        key = ("clim1", st)
        if key not in _G:  # the lead-1 climatology is the same for every lead
            _G[key] = np.mean([_arch_day(y, d1.month, d1.day, chans) for y in range(2190, 2200)], 0)
        clim_1 = _G[key]
        for name, f in (("emulator", pred), ("persistence_day1", ic),
                        ("persistence_plus_seasonal_2190_2199", ic + clim_l - clim_1),
                        ("climatology_2190_2199", clim_l)):
            e2 = (np.nan_to_num(f) - np.nan_to_num(truth)) ** 2
            w = _G["area"] * _G["wet"]
            out[(st, name)] = np.sqrt((e2.T * w).sum(1) / w.sum(1))
    return lead, out


def bench_cmd(a):
    leads = [1, 15, 30] + list(range(60, 3651, 60)) + [3650]
    with Pool(a.workers, initializer=_init) as pool:
        res = pool.map(_bench_lead, leads, chunksize=1)
    rows = []
    ct, cs = _channels("TEMP"), _channels("SALT")
    for lead, out in sorted(res):
        for (st, name), v in out.items():
            chans = ct if st == "TEMP" else cs
            r = {"lead": lead, "date": str(_date(lead).date()), "var": st, "model": name}
            for li, c in enumerate(chans):
                if c.split("_")[1] in ("2", "107", "210", "450", "735", "975", "1415", "2035"):
                    r[c] = v[li]
            rows.append(r)
    df = pd.DataFrame(rows)
    with open(a.out / "benchmarks_fvnn10y.csv", "w") as fh:
        fh.write("# fvnn10y vs benchmarks, global area-weighted RMSE over wet nodes at selected "
                 "levels. persistence_day1: truth of lead 1 (2200-01-02) held fixed. "
                 "persistence_plus_seasonal: that plus the 2190-2199 day-of-year climatology "
                 "change between the two calendar days. climatology_2190_2199: 10-year "
                 "day-of-year mean of 2190-2199 (out of sample for 2200-2209; 10 years only).\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    print(df[df.lead.isin([30, 360, 1800, 3420, 3650])].to_string())


def _integrity_year(args):
    arch, year = args
    if arch == "atmos_all":
        z = zarr.open(str(ARCHIVE / arch / f"{year}.zarr"), mode="r")
        step = np.timedelta64(6, "h")
    else:
        z = zarr.open(str(ARCHIVE / arch / f"{arch}_{year}.zarr"), mode="r")
        step = np.timedelta64(1, "D")
    n = z["data"].chunks[0]
    rows_total = z["dates"].shape[0]
    first = np.asarray(z["dates"][::n, 0])
    diffs = np.diff(first)
    leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    feb29 = np.datetime64(f"{year}-02-29") if leap else None
    days = first.astype("datetime64[D]")
    r = {"archive": arch, "year": year, "rows": rows_total, "rows_per_chunk": n,
         "n_chunks": rows_total / n, "first": str(first[0]), "last": str(first[-1]),
         "all_steps_equal": bool((diffs == step).all()), "max_gap_h": float(diffs.max() / np.timedelta64(1, "h")),
         "leap_year": leap, "feb29_present": bool((days == np.datetime64(f"{year}-02-29")).any()) if leap else
         bool(False), "n_colnames": len(z["data"].attrs["colnames"])}
    # within-chunk constancy of dates for a few chunks
    for k in (0, len(first) // 2, len(first) - 1):
        blk = np.asarray(z["dates"][k * n:(k + 1) * n, 0])
        r[f"chunk{k}_dates_constant"] = bool((blk == blk[0]).all())
    if year == 2209:
        names = list(z["data"].attrs["colnames"])
        cols = [names.index(c) for c in (["10u", "10v", "2t", "2d", "msl", "cp", "lsp", "tcc", "tsr"]
                                          if arch == "atmos_all" else names[2:])]
        idx = np.where((first >= np.datetime64("2209-07-01")) & (first < np.datetime64("2209-11-01")))[0]
        nans = [int(np.isnan(np.asarray(z["data"][i * n:(i + 1) * n])[:, cols]).sum()) for i in idx]
        r["nan_count_jul_oct_2209_total"] = int(np.sum(nans))
        r["nan_count_jul_oct_2209_max_per_snapshot"] = int(np.max(nans))
        r["snapshots_checked_jul_oct_2209"] = len(idx)
    return r


def integrity_cmd(a):
    tasks = [(arch, y) for arch in ("ocean_node", "ocean_elem", "atmos_all")
             for y in [2000, 2010, 2100, 2150] + list(range(2190, 2210))]
    with Pool(a.workers) as pool:
        rows = pool.map(_integrity_year, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    with open(a.out / "archive_integrity.csv", "w") as fh:
        fh.write("# Yearly archives: row counts, dates per chunk (first row of each chunk), step "
                 "regularity, Feb 29, number of columns, and for 2209 the NaN count of the used "
                 "columns per snapshot Jul-Oct.\n")
        df.to_csv(fh, index=False)
    print(df.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lambda", "stepreg", "seasonal", "pattern", "pattern2", "bench",
                                     "integrity"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--data", default=str(Path.home() / "claude_api_tools/y2209/data"))
    ap.add_argument("--out", default=str(Path.home() / "claude_api_tools/y2209/in2"))
    a = ap.parse_args()
    a.data, a.out = Path(a.data), Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    globals()[f"{a.what}_cmd"](a)


if __name__ == "__main__":
    main()
