"""Analyse the day-1 start-year scan (sbatch_day1_scan.sh, run ids PREFIX<YYYYMMDD>).

Per start: regional area-weighted means of prediction and truth and the RMSE for TEMP/SALT at
forecast steps 1 and 2 (regional_diag.py), the effective year tau of prediction and truth
(y2209_requests.py, same channels and method as effective_year_series.csv), and statistics of
the carried latent after step 2 (results/RUN/rollout_state.pt): per-token RMS, the cell-mean
vector, its PCA over all starts, and its projection on the "time direction" (mean latent of the
2170-2190 starts minus that of the 2010-2030 starts).

    python scripts/instab/day1_scan_analyse.py fvd1 [--out ~/claude_api_tools/y2209/in3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import regional_diag as rd  # noqa: E402
import y2209_requests as yr  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"


def regional(run):
    rd._init_rollout(run)
    out = {}
    for step in (1, 2):
        _, date, o = rd._rollout_step(step)
        out[step] = (date, o)
    return out


def latent_stats(run):
    s = torch.load(RESULTS / run / "rollout_state.pt", map_location="cpu", weights_only=False)
    # the carried state is z_pre_norm (patch_tokens is its layer-normed copy, token RMS 1)
    z = s["latent_state"].z_pre_norm.float()  # (batch, cells, dim)
    z = z.reshape(-1, z.shape[-1])
    rms = z.pow(2).mean(1).sqrt()
    return int(s["done_steps"]), z.mean(0).numpy(), z.half().flatten(), {
        "token_rms_mean": float(rms.mean()), "token_rms_p05": float(rms.quantile(0.05)),
        "token_rms_p95": float(rms.quantile(0.95)), "latent_frob": float(z.norm())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--data", default=str(Path.home() / "claude_api_tools/y2209/data"))
    ap.add_argument("--out", default=str(Path.home() / "claude_api_tools/y2209/in3"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in RESULTS.glob(f"{a.prefix}[0-9]*")
                  if (p / "validation_chkpt00000_rank0000.zip").exists())
    fit = yr.truth_fit(Path(a.data))
    depth = [float(c.split("_")[1]) for c in rd._channels("TEMP")]
    rows, tau_rows, lat_rows, means, fulls = [], [], [], [], []
    pred = {1: {}, 2: {}}
    truth = {1: {}, 2: {}}
    starts = []
    for run in runs:
        start = pd.Timestamp(run[len(a.prefix):])
        starts.append(start)
        reg = regional(run)
        for step, (date, o) in reg.items():
            for st in ("TEMP", "SALT"):
                for r in rd._G["reg_n"]:
                    p, t, e = o[(st, "pred", r)], o[(st, "truth", r)], o[(st, "rmse", r)]
                    for li, d in enumerate(depth):
                        pred[step].setdefault((st, r, d), []).append(p[li])
                        truth[step].setdefault((st, r, d), []).append(t[li])
                        if d in (2.0, 107.0, 450.0, 975.0, 2035.0):
                            rows.append({"start": start.date(), "step": step, "valid": date,
                                         "var": st, "region": r, "depth": d, "pred": p[li],
                                         "truth": t[li], "bias": p[li] - t[li], "rmse": e[li]})
        done, mean, full, stats = latent_stats(run)
        means.append(mean)
        fulls.append(full)
        lat_rows.append({"start": start.date(), "done_steps": done, **stats})
        print(run, "done", flush=True)
    for step in (1, 2):
        pk = {k: np.array(v) for k, v in pred[step].items()}
        tk = {k: np.array(v) for k, v in truth[step].items()}
        sp, _, _ = yr.tau_series(pk, fit)
        stt, _, _ = yr.tau_series(tk, fit)
        for i, s in enumerate(starts):
            tau_rows.append({"start": s.date(), "step": step,
                             **{f"pred_{k}": v[i] for k, v in sp.items()},
                             **{f"truth_{k}": v[i] for k, v in stt.items() if k != "n_channels"},
                             "lag_years_median": stt["tau_median"][i] - sp["tau_median"][i]})
    # latent PCA over starts and projection on the in-training time direction
    M = np.stack(means)
    yrs = np.array([s.year + (s.dayofyear - 1) / 365.25 for s in starts])
    Mc = M - M.mean(0)
    u, sv, vt = np.linalg.svd(Mc, full_matrices=False)
    early, late = (yrs >= 2010) & (yrs < 2031), (yrs >= 2170) & (yrs < 2191)
    tdir = M[late].mean(0) - M[early].mean(0)
    tnorm = np.linalg.norm(tdir)
    # time coordinate: projection on tdir, mapped so early-mean -> 2020, late-mean -> 2180
    proj = (M - M[early].mean(0)) @ tdir / tnorm ** 2
    jan = np.array([s.month == 1 for s in starts])
    for i, r in enumerate(lat_rows):
        r.update({"pc1": u[i, 0] * sv[0], "pc2": u[i, 1] * sv[1], "pc3": u[i, 2] * sv[2],
                  "time_dir_coord_years": 2020 + 160 * proj[i],
                  "resid_norm_from_time_dir": float(np.linalg.norm(
                      (M[i] - M[early].mean(0)) - proj[i] * tdir))})
    # lag from the local truth trend around each start year (the 2150-2209 fit used by tau is
    # not valid for early starts): per channel >= 440 m, linear fit of annual truth means over
    # [Y-20, Y+20] within 2010-2209; lag = -(pred - truth) / slope; median over channels whose
    # local trend exceeds 3 sigma of the fit residual per decade
    tr = xr.open_dataset(Path(a.data) / "truth_regional_2010_2209.nc").load()
    ty = pd.DatetimeIndex(tr.time.values).year
    ann = {st: tr[st].assign_coords(year=("time", ty)).groupby("year").mean() for st in ("TEMP", "SALT")}
    pk1 = {k: np.array(v) for k, v in pred[1].items()}
    tk1 = {k: np.array(v) for k, v in truth[1].items()}
    for i, s0 in enumerate(starts):
        y0 = s0.year
        lags = []
        for (st, r, d), pv in pk1.items():
            if d < 440 or r == "global":
                continue
            ser = ann[st].sel(region=r).sel(depth=d, method="nearest").to_series().loc[
                max(2010, y0 - 20):min(2209, y0 + 20)]
            if ser.isna().any() or len(ser) < 10:
                continue
            b, c = np.polyfit(ser.index.values - y0, ser.values, 1)
            res = ser.values - (c + b * (ser.index.values - y0))
            if abs(b * 10) > 3 * np.std(res, ddof=1):
                lags.append(-(pv[i] - tk1[(st, r, d)][i]) / b)
        tau_rows[i]["lag_local_trend_median"] = float(np.median(lags)) if lags else np.nan
        tau_rows[i]["lag_local_trend_n"] = len(lags)
    # linear probes latent -> start year, trained on starts < 2200, tested on 2200-2209
    train = yrs < 2200
    def ridge_probe(X):
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-12
        Z = (X - mu) / sd
        Zt, yt = Z[train], yrs[train]
        best = None
        # explicit leave-one-out refits (the hat-matrix shortcut is unstable when
        # features >> samples and lambda is small)
        for lam in 10.0 ** np.arange(-2, 7):
            loo = []
            for i in range(len(yt)):
                m = np.arange(len(yt)) != i
                Ki = Zt[m] @ Zt[m].T
                al = np.linalg.solve(Ki + lam * np.eye(m.sum()), yt[m] - yt[m].mean())
                loo.append(yt[i] - (yt[m].mean() + Zt[i] @ Zt[m].T @ al))
            sc = float(np.sqrt(np.mean(np.square(loo))))
            if best is None or sc < best[0]:
                best = (sc, lam)
        sc, lam = best
        K = Zt @ Zt.T
        alpha = np.linalg.solve(K + lam * np.eye(len(yt)), yt - yt.mean())
        yhat = yt.mean() + Z @ Zt.T @ alpha
        te = ~train
        slope = np.polyfit(yrs[te], yhat[te], 1)[0] if te.sum() > 2 else np.nan
        return yhat, sc, lam, slope
    # kernel PCA (linear kernel) of the full latent, centred on the training starts
    n = len(fulls)
    G = np.zeros((n, n))
    for i in range(n):
        fi = fulls[i].float()
        for j in range(i, n):
            G[i, j] = G[j, i] = float(fi @ fulls[j].float())
    T = np.where(train)[0]
    C = G - G[:, T].mean(1, keepdims=True) - G[T][:, :].mean(0, keepdims=True) + G[np.ix_(T, T)].mean()
    lam_, A = np.linalg.eigh(C[np.ix_(T, T)])
    order = np.argsort(lam_)[::-1]
    lam_, A = lam_[order], A[:, order]
    k = int(min(50, (lam_ > 1e-9 * lam_[0]).sum()))
    P = C[:, T] @ A[:, :k] / np.sqrt(lam_[:k])
    probes = {}
    for name, X in (("cellmean", M), (f"fullpc{k}", P)):
        yhat, sc, lam, slope = ridge_probe(X)
        probes[name] = (sc, lam, slope)
        for i, r in enumerate(lat_rows):
            r[f"probe_year_{name}"] = yhat[i]
    for i, r in enumerate(lat_rows):
        r["fullpc1"], r["fullpc2"], r["fullpc3"] = P[i, 0], P[i, 1], P[i, 2]
    probe_txt = "; ".join(f"{nm}: LOO residual std {v[0]:.2f} yr (ridge lambda {v[1]:g}), "
                          f"slope of probe year vs start year over 2200-2209 starts {v[2]:.2f}"
                          for nm, v in probes.items())
    expl = sv ** 2 / (sv ** 2).sum()
    hdr = {"regional": "# Day-1 scan: regional area-weighted means of prediction and truth, bias "
                       "and RMSE at steps 1 and 2 for every start date (selected levels).\n",
           "tau": "# Day-1 scan: effective year tau of prediction and truth per start (same "
                  "channels and method as effective_year_series.csv); lag = truth tau - pred tau. "
                  "lag_local_trend_median (step-1 rows only): -(pred - truth) / local truth trend, "
                  "trend fitted to annual regional means over start year +-20 yr (within "
                  "2010-2209), median over channels >= 440 m with a significant local trend.\n",
           "latent": f"# Day-1 scan: carried latent (z_pre_norm) after step 2 per start. token_rms = RMS over "
                     f"features per cell token. pc1-3 = PCA of the cell-mean latent vector over "
                     f"all starts (explained variance {expl[0]:.3f}, {expl[1]:.3f}, {expl[2]:.3f}). "
                     f"time_dir_coord_years = projection on the direction mean(2170-2190 starts) - "
                     f"mean(2010-2030 starts), scaled so these means map to 2020 and 2180 "
                     f"(Jan and Jul starts pooled; Jan-only starts: {int(jan.sum())}). "
                     f"probe_year_* = ridge regression latent -> start year trained on starts before "
                     f"2200 (leave-one-out choice of lambda), applied to all starts; features: the "
                     f"cell-mean vector, or the leading kernel-PCA coordinates of the full latent "
                     f"(fullpc1-3 given) centred on the training starts. {probe_txt}.\n"}
    for name, df in (("regional", pd.DataFrame(rows)), ("tau", pd.DataFrame(tau_rows)),
                     ("latent", pd.DataFrame(lat_rows))):
        with open(out / f"day1_scan_{name}.csv", "w") as fh:
            fh.write(hdr[name])
            df.to_csv(fh, index=False, float_format="%.5g")
    print(pd.DataFrame(tau_rows).query("step == 1")[["start", "pred_tau_median",
                                                      "truth_tau_median", "lag_years_median"]])
    print(pd.DataFrame(lat_rows)[["start", "pc1", "pc2", "time_dir_coord_years",
                                  "token_rms_mean"]])


if __name__ == "__main__":
    main()
