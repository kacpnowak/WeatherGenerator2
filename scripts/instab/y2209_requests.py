"""Measurements requested by the 2209 review (round 1), from stored outputs and archives.

    python scripts/instab/y2209_requests.py lambda      # effective-year series per run
    python scripts/instab/y2209_requests.py stepreg     # step change vs truth trend, per level
    python scripts/instab/y2209_requests.py seasonal    # seasonal cycle of the deep bias
    python scripts/instab/y2209_requests.py pattern     # node-level pattern test (archive)
    python scripts/instab/y2209_requests.py bench       # persistence / climatology benchmarks
    python scripts/instab/y2209_requests.py integrity   # archive calendar, columns, NaNs
    python scripts/instab/y2209_requests.py anchored    # IC-anchored re-scoring (lead-1 bias removed)
    python scripts/instab/y2209_requests.py scaledec    # basin / 10-degree box / node decomposition
    python scripts/instab/y2209_requests.py pattern_f09 # pattern2 applied to fvnnf09
    python scripts/instab/y2209_requests.py poststep    # fvnn10y post-step state vs fresh encodings
    python scripts/instab/y2209_requests.py anchored2   # anchored re-scoring, anchors at leads 30/90
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
                      ("fvnn07y", "M"), ("fvnn08y", "M"), ("fvnn01y1", "M"), ("fvnn02y1", "M"),
                      ("fvnn04y1", "M"), ("fvfs02y", "M"), ("fvfs09y", "M")):
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


def _init(open_zip=True):
    m = xr.open_dataset(MESH)
    _G["lon"], _G["lat"] = np.asarray(m.lon), np.asarray(m.lat)
    nz = len(_channels("TEMP"))
    _G["wet"] = np.arange(nz)[:, None] < (np.asarray(m.nlevels_nod2D) - 1)[None, :]
    _G["area"] = np.asarray(m.nod_area)[:nz]
    _G["n"] = len(m.lon)
    _G["zs"] = {}  # fresh zip handles per process (handles inherited over fork are not safe)
    _G.pop("z", None)
    if open_zip:
        _G["z"] = _zg("fvnn10y")


def _zg(run):
    """Output zip of a rollout run, opened once per process (fvnn10y is also _G["z"])."""
    zs = _G.setdefault("zs", {})
    if run not in zs:
        if run == "fvnn10y" and "z" in _G:
            zs[run] = _G["z"]
        else:
            zs[run] = zarr.open_group(store=ZipStore(
                f"results/{run}/validation_chkpt00000_rank0000.zip", mode="r", read_only=True),
                mode="r")
    return zs[run]


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


def _zip(stream, lead, kind, run="fvnn10y"):
    x = np.asarray(_zg(run)[f"0/{stream}/{lead}/{kind}/data"][...], dtype=np.float64)
    return x[..., 0] if x.ndim == 3 else x


def _zdate(stream, lead, run):
    """Valid date of a forecast step, read from the zip's target times (not assumed)."""
    t = np.asarray(_zg(run)[f"0/{stream}/{lead}/target/times"][:1])
    return str(t[0])[:10]


def _nleads(run, stream="TEMP"):
    return max(int(k) for k in _zg(run)[f"0/{stream}"].keys())


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


# ----- review round 2: IC-anchored re-scoring, scale decomposition, fvnnf09 pattern test ------

ANCH_LEVELS = (2, 107, 450, 975, 2035)
DECOMP_REGIONS = ("mediterranean", "red_sea", "arctic", "n_atlantic", "trop_atlantic",
                  "s_atlantic", "indian", "n_pacific", "s_pacific", "southern")


def _layer_dz(chans):
    """Layer thicknesses (m) from the level depths in the channel names: interfaces at the
    midpoints between levels, 0 m at the top and 2150 m at the bottom."""
    z = np.array([float(c.split("_")[1]) for c in chans])
    edges = np.concatenate([[0.0], (z[1:] + z[:-1]) / 2, [2150.0]])
    return np.diff(edges)


def _anch_task(args):
    run, lead = args
    out = {}
    for st in ("TEMP", "SALT"):
        chans = _channels(st)
        if ("b1", run, st) not in _G:  # lead-1 bias and truth, once per process and run
            t1 = np.nan_to_num(_zip(st, 1, "target", run))
            _G[("t1", run, st)] = t1
            _G[("b1", run, st)] = np.nan_to_num(_zip(st, 1, "prediction", run)) - t1
        t1, b1 = _G[("t1", run, st)], _G[("b1", run, st)]
        truth = np.nan_to_num(_zip(st, lead, "target", run))
        pred = np.nan_to_num(_zip(st, lead, "prediction", run))
        w = (_G["area"] * _G["wet"]).T  # (nod, lev)
        wv = w * _layer_dz(chans)[None, :]
        for name, e in (("raw", pred - truth), ("anchored", pred - b1 - truth),
                        ("persistence_lead1", t1 - truth)):
            out[(st, f"rmse_{name}")] = (np.sqrt((e ** 2 * w).sum(0) / w.sum(0)),
                                         np.sqrt((e ** 2 * wv).sum() / wv.sum()))
            if name != "persistence_lead1":
                out[(st, f"mean_err_{name}")] = ((e * w).sum(0) / w.sum(0),
                                                 (e * wv).sum() / wv.sum())
    return run, lead, _zdate("TEMP", lead, run), out


def anchored_cmd(a):
    _init(open_zip=False)
    tasks = []
    for run in ("fvnnf09", "fvnnf00", "fvnn10y"):
        n = _nleads(run)
        leads = sorted(set([1, 15, 30] + list(range(30, n + 1, 30)) + [n]))
        tasks += [(run, l) for l in leads]
    for z in _G["zs"].values():  # do not hand open zip handles to the forked workers
        z.store.close()
    _G["zs"] = {}
    with Pool(a.workers, initializer=_init, initargs=(False,)) as pool:
        res = pool.map(_anch_task, tasks, chunksize=1)
    rows = []
    for run, lead, date, out in res:
        for (st, q), (lev, vol) in out.items():
            chans = _channels(st)
            r = {"run": run, "lead": lead, "date": date, "var": st, "quantity": q}
            for d in ANCH_LEVELS:
                r[f"z{d}"] = lev[chans.index(f"{st.lower()}_{d}")]
            r["vol_0_2150"] = vol
            rows.append(r)
    df = pd.DataFrame(rows)
    with open(a.out / "anchored_rescoring.csv", "w") as fh:
        fh.write("# IC-anchored re-scoring (fvnnf09 start 2209-01-01, fvnnf00 and fvnn10y start "
                 "2200-01-01; date = valid date read from the zip's target times). b1 = pred(lead 1) "
                 "- truth(lead 1) per node and level. Global area-weighted over wet nodes at the levels "
                 "z<depth> (m). rmse_raw: RMSE of pred - truth. rmse_anchored: RMSE of (pred - b1) - "
                 "truth (0 at lead 1 by construction). rmse_persistence_lead1: RMSE of truth(lead 1) - "
                 "truth. mean_err_raw / mean_err_anchored: area-weighted mean of pred - truth / "
                 "(pred - b1) - truth. vol_0_2150: the same over the 31 levels 0-2150 m, area x layer "
                 "thickness weighted (thickness from midpoints between level depths, 0 and 2150 m at "
                 "the ends); on rmse_* rows the volume RMSE, on mean_err_* rows the volume-mean error.\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    sel = df[(df.lead.isin([1, 30, 90, 180, 360, 363, 1800, 3420, 3600, 3650]))
             & df.quantity.str.startswith("rmse")]
    print(sel.to_string())


def _trend_map(data):
    """20-year truth change, mean(2195-2199) - mean(2175-2179) of monthly 15th snapshots, TEMP
    then SALT channels (nod, 62). Cached in the data directory."""
    p = data / "trend_map_5ymean_2195_2199_minus_2175_2179.npy"
    if p.exists():
        return np.load(p)
    chans = _channels("TEMP") + _channels("SALT")
    late = np.mean([_arch_day(y, m, 15, chans) for y in range(2195, 2200) for m in range(1, 13)], 0)
    early = np.mean([_arch_day(y, m, 15, chans) for y in range(2175, 2180) for m in range(1, 13)], 0)
    trend = late - early
    tmp = p.with_suffix(f".tmp{np.random.randint(1 << 30)}.npy")
    np.save(tmp, trend)
    tmp.rename(p)
    return trend


def _win(st, centre, kind, run="fvnn10y", last=None):
    """21-day mean centred on `centre` (every 2nd day, leads c-10..c+10, clipped to 1..last)."""
    leads = [l for l in range(centre - 10, centre + 11, 2) if l >= 1 and (last is None or l <= last)]
    return np.mean([_zip(st, l, kind, run) for l in leads], 0), leads


def _decomp(f, w, reg, box):
    """Nested decomposition f - mean = basin + box10 + node (orthogonal under the area weights:
    basin = basin mean - global mean, box10 = mean of (f - basin mean) over the 10x10 degree box
    within the basin, node = remainder)."""
    sw = w.sum()
    m = np.sum(w * f) / sw
    bm = np.bincount(reg, w * f, reg.max() + 1) / np.maximum(np.bincount(reg, w, reg.max() + 1), 1e-300)
    r1 = f - bm[reg]
    cell = reg * 648 + box
    cw = np.bincount(cell, w, cell.max() + 1)
    cm = np.bincount(cell, w * r1, cell.max() + 1) / np.maximum(cw, 1e-300)
    parts = {"basin": bm[reg] - m, "box10": cm[cell], "node": r1 - cm[cell]}
    vt = np.sum(w * (f - m) ** 2) / sw
    return m, vt, parts, {k: np.sum(w * v ** 2) / sw / vt for k, v in parts.items()}


def scaledec_cmd(a):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regional_diag import regions

    _init(open_zip=False)
    run = "fvnn10y"
    lon, lat = _G["lon"], _G["lat"]
    rg = regions(lon, lat)
    reg = np.full(lon.shape, len(DECOMP_REGIONS), int)  # "other"
    for i, name in reversed(list(enumerate(DECOMP_REGIONS))):  # first match wins
        reg[rg[name]] = i
    box10 = (np.floor((lat + 90) / 10).clip(0, 17) * 36
             + np.floor(((lon + 180) % 360) / 10).clip(0, 35)).astype(int)
    trend_all = _trend_map(a.data)

    def lead_of(date):
        lead = int((np.datetime64(date) - START).astype(int))
        assert _zdate("TEMP", lead, run) == date, (date, _zdate("TEMP", lead, run))
        return lead

    rows = []
    for st in ("TEMP", "SALT"):
        chans = _channels(st)
        trend = trend_all[:, :31] if st == "TEMP" else trend_all[:, 31:]
        cache = {}

        def w21(date, kind):
            if (date, kind) not in cache:
                cache[(date, kind)] = _win(st, lead_of(date), kind, run)[0]
            return cache[(date, kind)]

        def bias(date):
            return w21(date, "prediction") - w21(date, "target")

        qs = {}
        for y in (2209, 2206, 2204):
            a_, b_ = f"{y}-06-13", f"{y}-11-10"
            qs[(y, f"step_pred: pred({b_}) - pred({a_})")] = w21(b_, "prediction") - w21(a_, "prediction")
            qs[(y, f"step_bias: bias({b_}) - bias({a_})")] = bias(b_) - bias(a_)
        qs[(2209, "drift_bias: bias(2209-05-15) - bias(2200-05-15)")] = bias("2209-05-15") - bias("2200-05-15")
        for depth in (450, 975, 2035):
            li = chans.index(f"{st.lower()}_{depth}")
            m = _G["wet"][li] & np.isfinite(trend[:, li])
            w = _G["area"][li][m]
            xm, xvt, xp, xf = _decomp(trend[m, li], w, reg[m], box10[m])
            rows.append({"var": st, "depth": depth, "year": "", "quantity": "trend_map_20yr",
                         "mean": xm, "rms_centred": np.sqrt(xvt),
                         **{f"varfrac_{k}": v for k, v in xf.items()}})
            for (y, qn), q in qs.items():
                ym, yvt, yp, yf = _decomp(q[m, li], w, reg[m], box10[m])
                r = {"var": st, "depth": depth, "year": y, "quantity": qn, "mean": ym,
                     "rms_centred": np.sqrt(yvt), **{f"varfrac_{k}": v for k, v in yf.items()}}
                r["corr_total"], r["slope_total"] = _wcorr(trend[m, li], q[m, li], w)
                for k in ("basin", "box10", "node"):
                    r[f"corr_{k}"], r[f"slope_{k}"] = _wcorr(xp[k], yp[k], w)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["varfrac_sum"] = df.varfrac_basin + df.varfrac_box10 + df.varfrac_node
    with open(a.out / "scale_decomposition.csv", "w") as fh:
        fh.write("# fvnn10y scale decomposition, global wet nodes (and finite trend), area weighted. "
                 "Quantities are 21-day means centred on the named dates (every 2nd day, leads c-10..c+10; "
                 "bias = pred - truth); step_* for the same calendar windows in model years 2209, 2206, "
                 "2204; trend_map_20yr = mean(2195-2199) - mean(2175-2179) of monthly 15th truth snapshots. "
                 "Each field f is split as f - mean = basin + box10 + node: basin = mean over its basin "
                 "minus the global mean (non-overlapping basins: each node to the first matching region "
                 "of regional_diag.regions in the order mediterranean, red_sea, arctic, n_atlantic, "
                 "trop_atlantic, s_atlantic, indian, n_pacific, s_pacific, southern, else other); box10 = "
                 "mean of (f - basin mean) over the 10x10 degree box intersected with the basin; node = "
                 "remainder. The parts are orthogonal, varfrac_* = part variance / centred total "
                 "variance (sum to 1). mean = global area-weighted mean (not in the variance). corr_*/"
                 "slope_* = area-weighted node-level correlation / regression slope of the part of the "
                 "quantity on the same part of the trend map; corr_total/slope_total for the full fields.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.drop(columns=[c for c in df.columns if c.startswith("slope_")]).to_string())


def pattern_f09_cmd(a):
    """pattern2 applied to fvnnf09 (start 2209-01-01): bias = pred - truth at leads 1 (one day),
    90, 180, 360 (21-day means, clipped to the run), vs the 20-year truth trend map."""
    _init(open_zip=False)
    run = "fvnnf09"
    last = _nleads(run)
    trend_all = _trend_map(a.data)
    lon, lat = _G["lon"], _G["lat"]
    box = (np.floor((lat + 90) / 5) * 72 + np.floor(((lon + 180) % 360) / 5)).astype(int)
    med = (lon > -5.5) & (lon < 36.5) & (lat > 30) & (lat < 46)
    rows = []
    for st in ("TEMP", "SALT"):
        chans = _channels(st)
        trend = trend_all[:, :31] if st == "TEMP" else trend_all[:, 31:]
        for lead in (1, 90, 180, 360):
            lead = min(lead, last)
            if lead == 1:
                q, leads = _zip(st, 1, "prediction", run) - _zip(st, 1, "target", run), [1]
            else:
                p, leads = _win(st, lead, "prediction", run, last)
                q = p - _win(st, lead, "target", run, last)[0]
            info = {"run": run, "lead": lead, "date": _zdate(st, lead, run),
                    "window_first": _zdate(st, leads[0], run), "window_last": _zdate(st, leads[-1], run),
                    "n_days": len(leads), "var": st}
            for depth in (450, 975, 2035):
                li = chans.index(f"{st.lower()}_{depth}")
                wet = _G["wet"][li] & np.isfinite(trend[:, li]) & np.isfinite(q[:, li])
                for mname, m in (("global", wet), ("global excl. Mediterranean", wet & ~med)):
                    w = _G["area"][li][m]
                    x, y = trend[m, li], q[m, li]
                    r, sl = _wcorr(x, y, w)
                    b = box[m]
                    sw = np.bincount(b, w, 72 * 36)
                    ok = sw > 0
                    bx = np.bincount(b, w * x, 72 * 36)[ok] / sw[ok]
                    by = np.bincount(b, w * y, 72 * 36)[ok] / sw[ok]
                    rb, slb = _wcorr(bx, by, sw[ok])
                    rows.append({**info, "depth": depth, "mask": mname, "corr_node": r, "slope_node": sl,
                                 "corr_5deg_box": rb, "slope_5deg_box": slb,
                                 "mean_bias": np.average(y, weights=w),
                                 "rms_quantity": np.sqrt(np.average(y ** 2, weights=w)),
                                 "rms_trend_map": np.sqrt(np.average(x ** 2, weights=w))})
    df = pd.DataFrame(rows)
    with open(a.out / "pattern_fvnnf09.csv", "w") as fh:
        fh.write("# pattern2 method (node_pattern_test_v2.csv) applied to fvnnf09 (start 2209-01-01). "
                 "Quantity = bias = pred - truth at the named lead: lead 1 a single day, other leads "
                 "21-day means (every 2nd day, leads c-10..c+10, clipped to the run's last lead; "
                 "window_first/last/n_days give the days used; dates read from the zip). Trend map x = "
                 "mean(2195-2199) - mean(2175-2179) of monthly 15th truth snapshots. Area-weighted "
                 "correlation and regression slope of the bias on x at node level and after 5x5 degree "
                 "box averaging; slope -1 = minus the 20-year change. mean_bias = area-weighted mean.\n")
        df.to_csv(fh, index=False, float_format="%.4g")
    print(df.to_string())


# ----- review round 4: fvnn10y post-step state vs fresh encodings, later anchors ---------------

POST_DATE = "2209-11-10"
POST_RUNS = ("fvnn10y", "fvnnf09", "fvnn07y", "fvnn08y")
POST_PAIRS = (("fvnn10y", "fvnnf09"), ("fvnn10y", "fvnn07y"), ("fvnn10y", "fvnn08y"),
              ("fvnnf09", "fvnn07y"))


def _lead_of_date(run, date):
    """Lead whose valid date (read from the zip) is `date`; first guess from the lead-1 date."""
    d1 = np.datetime64(_zdate("TEMP", 1, run))
    lead = 1 + int((np.datetime64(date) - d1).astype(int))
    for cand in [lead] + [lead + k for k in (-1, 1, -2, 2)]:
        if cand >= 1 and _zdate("TEMP", cand, run) == date:
            return cand
    raise ValueError(f"{run}: no lead with valid date {date} near {lead}")


def _basin_index(lon, lat):
    """Non-overlapping basins as in scaledec: first matching region of DECOMP_REGIONS, else other."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regional_diag import regions

    rg = regions(lon, lat)
    reg = np.full(lon.shape, len(DECOMP_REGIONS), int)
    for i, name in reversed(list(enumerate(DECOMP_REGIONS))):
        reg[rg[name]] = i
    return reg, list(DECOMP_REGIONS) + ["other"]


def _post_task(args):
    run, st = args
    centre = _lead_of_date(run, POST_DATE)
    p, leads = _win(st, centre, "prediction", run)
    t = _win(st, centre, "target", run)[0]
    return run, st, centre, _zdate(st, leads[0], run), _zdate(st, leads[-1], run), len(leads), p, p - t


def poststep_cmd(a):
    _init(open_zip=False)
    lon, lat = _G["lon"], _G["lat"]
    reg, rnames = _basin_index(lon, lat)
    box = (np.floor((lat + 90) / 5).clip(0, 35) * 72 + np.floor(((lon + 180) % 360) / 5).clip(0, 71)).astype(int)
    tasks = [(run, st) for run in POST_RUNS for st in ("TEMP", "SALT")]
    with Pool(min(a.workers, len(tasks)), initializer=_init, initargs=(False,)) as pool:
        res = pool.map(_post_task, tasks, chunksize=1)
    F, info = {}, {}
    for run, st, centre, w0, w1, nd, p, b in res:
        F[(run, st, "pred")], F[(run, st, "bias")] = p, b
        info[(run, st)] = dict(centre=centre, w0=w0, w1=w1, nd=nd)
        print(run, st, "centre lead", centre, w0, w1, nd)
    rows = []
    for st in ("TEMP", "SALT"):
        chans = _channels(st)
        for depth in (975, 2035):
            li = chans.index(f"{st.lower()}_{depth}")
            for q in ("bias", "pred"):
                m = _G["wet"][li].copy()
                for run in POST_RUNS:  # common mask over all four runs
                    m &= np.isfinite(F[(run, st, q)][:, li])
                w = _G["area"][li][m]
                rr, bb = reg[m], box[m]
                for ra, rb in POST_PAIRS:
                    xa, xb = F[(ra, st, q)][m, li], F[(rb, st, q)][m, li]
                    ga, gb = np.average(xa, weights=w), np.average(xb, weights=w)
                    ya, yb = xa - ga, xb - gb
                    r_node = _wcorr(ya, yb, w)[0]
                    sw = np.bincount(bb, w, 72 * 36)
                    ok = sw > 0
                    r_box = _wcorr(np.bincount(bb, w * ya, 72 * 36)[ok] / sw[ok],
                                   np.bincount(bb, w * yb, 72 * 36)[ok] / sw[ok], sw[ok])[0]
                    ia, ib = info[(ra, st)], info[(rb, st)]
                    base = {"run_a": ra, "run_b": rb, "var": st, "depth": depth, "quantity": q,
                            "lead_a": ia["centre"], "window_a": f"{ia['w0']}..{ia['w1']}",
                            "lead_b": ib["centre"], "window_b": f"{ib['w0']}..{ib['w1']}",
                            "n_days": ia["nd"]}
                    rows.append({**base, "region": "global", "corr_node": r_node, "corr_5deg_box": r_box,
                                 "mean_a": ga, "mean_b": gb, "mean_diff": ga - gb,
                                 "mean_diff_centred": 0.0, "rms_centred_a": np.sqrt(np.average(ya ** 2, weights=w)),
                                 "rms_centred_b": np.sqrt(np.average(yb ** 2, weights=w))})
                    rw = np.bincount(rr, w, len(rnames))
                    ma = np.bincount(rr, w * xa, len(rnames)) / np.maximum(rw, 1e-300)
                    mb = np.bincount(rr, w * xb, len(rnames)) / np.maximum(rw, 1e-300)
                    for k, name in enumerate(rnames):
                        if rw[k] == 0:
                            continue
                        rows.append({**base, "region": name, "mean_a": ma[k], "mean_b": mb[k],
                                     "mean_diff": ma[k] - mb[k],
                                     "mean_diff_centred": (ma[k] - ga) - (mb[k] - gb),
                                     "area_frac": rw[k] / w.sum()})
    df = pd.DataFrame(rows)
    with open(a.out / "poststep_vs_fresh.csv", "w") as fh:
        fh.write(f"# fvnn10y post-step state vs fresh out-of-range encodings (fvnnf09 start 2209-01-01, "
                 f"fvnn07y 2207-01-01, fvnn08y 2208-01-01; fvnn10y 2200-01-01). Fields: 21-day means "
                 f"centred on {POST_DATE} (every 2nd day, leads c-10..c+10, c = lead whose zip valid date is "
                 f"{POST_DATE}; lead_*/window_* give each run's centre lead and first..last valid dates) of "
                 "pred (quantity=pred) and of bias = pred - truth (quantity=bias), TEMP/SALT at 975 and "
                 "2035 m. Nodes: wet at the level and finite in all four runs; area weighted. Pairs "
                 "a vs b: fvnn10y vs each fresh run, and fvnnf09 vs fvnn07y (reference pair of two fresh "
                 "encodings). region=global rows: corr_node = area-weighted node-level Pearson correlation "
                 "of the two fields after removing each field's global area-weighted mean; corr_5deg_box "
                 "= the same after area-weighted averaging into 5x5 degree boxes (box weight = summed "
                 "area); mean_a/mean_b = global area-weighted means, mean_diff = mean_a - mean_b; "
                 "rms_centred_* = area-weighted RMS after removing the global mean. Basin rows: "
                 "non-overlapping basins as in scale_decomposition.csv (each node to the first matching "
                 "region of regional_diag.regions in the order mediterranean, red_sea, arctic, n_atlantic, "
                 "trop_atlantic, s_atlantic, indian, n_pacific, s_pacific, southern, else other); "
                 "mean_a/mean_b = basin area-weighted means, mean_diff = mean_a - mean_b (raw), "
                 "mean_diff_centred = the basin mean difference after removing each field's global mean "
                 "(= mean_diff - global mean_diff); area_frac = basin share of the masked area.\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    g = df[df.region.isin(["global", "mediterranean"])]
    print(g[["run_a", "run_b", "var", "depth", "quantity", "region", "corr_node", "corr_5deg_box",
             "mean_diff", "mean_diff_centred"]].to_string())


A2_RUNS = ("fvnnf09", "fvnn07y", "fvnn08y")
A2_ANCHORS = (30, 90)
A2_LEVELS = (450, 975, 2035)


def _a2_leads(run, last):
    """Scored leads per anchor (clipped to the run) and the set of all leads to read."""
    sel = {}
    for an in A2_ANCHORS:
        ls = [an + k for k in (0, 15, 30, 60, 90, 180, 270, 360)] + [last]
        if run == "fvnn07y":
            ls += list(range(an, last + 1, 90))
        sel[an] = sorted(set(l for l in ls if l <= last))
    return sel


def _a2_anchor_task(args):
    run, an = args
    out = {}
    for st in ("TEMP", "SALT"):
        bs = [np.nan_to_num(_zip(st, l, "prediction", run)) - np.nan_to_num(_zip(st, l, "target", run))
              for l in range(an - 2, an + 3)]
        out[st] = np.mean(bs, 0)
    return run, an, [_zdate("TEMP", l, run) for l in (an - 2, an + 2)], out


def _a2_task(args):
    """Per (run, lead): volume-mean raw error (every lead); for scored leads also level RMSE /
    mean error and the anchored versions for each anchor whose scored set contains the lead."""
    run, lead, anchors, mapdir = args
    res = {"vol_mean_raw": {}, "scored": {}}
    for st in ("TEMP", "SALT"):
        chans = _channels(st)
        e = np.nan_to_num(_zip(st, lead, "prediction", run)) - np.nan_to_num(_zip(st, lead, "target", run))
        w = (_G["area"] * _G["wet"]).T
        wv = w * _layer_dz(chans)[None, :]
        res["vol_mean_raw"][st] = (e * wv).sum() / wv.sum()
        if not anchors:
            continue

        def stats(x):
            return {"rmse": (np.sqrt((x ** 2 * w).sum(0) / w.sum(0)), np.sqrt((x ** 2 * wv).sum() / wv.sum())),
                    "mean_err": ((x * w).sum(0) / w.sum(0), (x * wv).sum() / wv.sum())}

        res["scored"][(st, None)] = stats(e)
        for an in anchors:
            key = ("amap", run, an, st)
            if key not in _G:
                _G[key] = np.load(Path(mapdir) / f"{run}_{an}_{st}.npy")
            res["scored"][(st, an)] = stats(e - _G[key])
    return run, lead, _zdate("TEMP", lead, run), res


def anchored2_cmd(a):
    import shutil
    import tempfile

    _init(open_zip=False)
    lasts = {run: _nleads(run) for run in A2_RUNS}
    sels = {run: _a2_leads(run, lasts[run]) for run in A2_RUNS}
    for z in _G["zs"].values():  # do not hand open zip handles to the forked workers
        z.store.close()
    _G["zs"] = {}
    mapdir = Path(tempfile.mkdtemp(prefix="anchor_maps_", dir=a.data))
    try:
        with Pool(a.workers, initializer=_init, initargs=(False,)) as pool:
            amap_res = pool.map(_a2_anchor_task, [(r, an) for r in A2_RUNS for an in A2_ANCHORS], chunksize=1)
            win = {}
            for run, an, wd, out in amap_res:
                win[(run, an)] = wd
                for st, m in out.items():
                    np.save(mapdir / f"{run}_{an}_{st}.npy", m)
            tasks = []
            for run in A2_RUNS:
                first = min(A2_ANCHORS) + 30
                for lead in range(1, lasts[run] + 1):
                    ans = [an for an in A2_ANCHORS if lead in sels[run][an]]
                    if ans or lead >= first:
                        tasks.append((run, lead, ans, str(mapdir)))
            res = pool.map(_a2_task, tasks, chunksize=4)
    finally:
        shutil.rmtree(mapdir, ignore_errors=True)
    rows, drift = [], []
    series = {}
    for run, lead, date, r in res:
        series.setdefault(run, []).append((lead, date, r["vol_mean_raw"]))
        for (st, an), sts in r["scored"].items():
            chans = _channels(st)
            for an2 in ([an] if an is not None else [x for x in A2_ANCHORS if lead in sels[run][x]]):
                kind = "raw" if an is None else "anchored"
                for qn, (lev, vol) in sts.items():
                    row = {"run": run, "anchor_lead": an2, "anchor_window": "{}..{}".format(*win[(run, an2)]),
                           "lead": lead, "date": date, "var": st, "quantity": f"{qn}_{kind}"}
                    for d in A2_LEVELS:
                        row[f"z{d}"] = lev[chans.index(f"{st.lower()}_{d}")]
                    row["vol_0_2150"] = vol
                    rows.append(row)
    for run in A2_RUNS:
        s = sorted(series[run])
        leads = np.array([x[0] for x in s])
        dates = np.array([np.datetime64(x[1]) for x in s])
        for an in A2_ANCHORS:
            amask = np.array([x[0] >= an + 30 for x in s])
            yrs = (dates[amask] - dates[amask][0]).astype(int) / 365.25
            for st in ("TEMP", "SALT"):
                y =np.array([x[2][st] for x in s])[amask]
                slope, icpt = np.polyfit(yrs, y, 1)
                drift.append({"run": run, "anchor_lead": an, "anchor_window": "{}..{}".format(*win[(run, an)]),
                              "lead": "drift", "date": f"{dates[amask][0]}..{dates[amask][-1]}", "var": st,
                              "quantity": f"drift_vol_mean_err_anchored_per_yr ({'K' if st == 'TEMP' else 'psu'}/yr)",
                              "vol_0_2150": slope, "n_leads_fit": int(amask.sum()),
                              "first_lead_fit": int(leads[amask][0]), "last_lead_fit": int(leads[amask][-1])})
    df = pd.DataFrame(rows).sort_values(["run", "anchor_lead", "var", "quantity", "lead"])
    df = pd.concat([df, pd.DataFrame(drift)], ignore_index=True)
    with open(a.out / "anchored_late_anchor.csv", "w") as fh:
        fh.write("# Anchored re-scoring with later anchors (fvnnf09 start 2209-01-01, fvnn07y 2207-01-01, "
                 "fvnn08y 2208-01-01; date = valid date read from the zip's target times). Anchor map for "
                 "anchor lead a: A_a = mean of pred - truth over leads a-2..a+2 (5 days, anchor_window = "
                 "their valid dates), per node and level. Scored leads: a, a+15, a+30, a+60, a+90, a+180, "
                 "a+270, a+360 and the run's last lead (fvnn07y also every 90 days from a to the end), "
                 "only those <= last lead. rmse_raw / mean_err_raw: RMSE / area-weighted mean of pred - "
                 "truth; rmse_anchored / mean_err_anchored: the same for (pred - A_a) - truth (near 0 at "
                 "lead a, not exactly 0 since A_a is a 5-day mean); raw rows are repeated for each anchor. "
                 "Global area-weighted over wet nodes (NaN set to 0, as in anchored_rescoring.csv) at the "
                 "levels z<depth> (m). vol_0_2150: the same over the 31 levels 0-2150 m, area x layer "
                 "thickness weighted (thickness from midpoints between level depths, 0 and 2150 m at the "
                 "ends); on rmse_* rows the volume RMSE, on mean_err_* rows the volume-mean error. "
                 "lead=drift rows: least-squares slope (per 365.25 days of valid date) of the daily "
                 "anchored volume-mean error over all leads >= a+30 to the last lead (K/yr for TEMP, "
                 "psu/yr for SALT), in vol_0_2150; it equals the slope of the raw volume-mean error over "
                 "the same leads, since A_a is constant in time.\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    sel = df[df.quantity.isin(["rmse_raw", "rmse_anchored"]) | (df.lead == "drift")]
    print(sel.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lambda", "stepreg", "seasonal", "pattern", "pattern2", "bench",
                                     "integrity", "anchored", "scaledec", "pattern_f09", "poststep",
                                     "anchored2"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--data", default=str(Path.home() / "claude_api_tools/y2209/data"))
    ap.add_argument("--out", default=str(Path.home() / "claude_api_tools/y2209/in2"))
    a = ap.parse_args()
    a.data, a.out = Path(a.data), Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    globals()[f"{a.what}_cmd"](a)


if __name__ == "__main__":
    main()
