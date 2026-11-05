"""Projection of the daily forcing injection on the latent's year axes (2209 review, r4 request 1).

forcing_decomp.py stores, per start year, the cell-mean gated injection f_k = gate * emb(forcing_k)
for 363 days (f_cellmean). The year axes come from the day-1 scan (fvd1* states, cell-mean
z_pre_norm): pc2 of their PCA, and the time direction mean(2170-2190 starts) - mean(2010-2030
starts) scaled so that one unit = one year (160 yr between the two means). Per year: the daily
projections, their Apr-Sep sum and annual mean, the cosine of f_k with the time direction, and
the projection of the year's deviation from the all-years daily mean.

    python scripts/instab/forcing_projection.py [--out ~/claude_api_tools/y2209/in5]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESULTS = Path(__file__).resolve().parents[2] / "results"
D1 = Path.home() / "claude_api_tools/fable_diag/D1/data"


def axes():
    starts, means = [], []
    for p in sorted(RESULTS.glob("fvd1[0-9]*")):
        if not (p / "rollout_state.pt").exists():
            continue
        s = torch.load(p / "rollout_state.pt", map_location="cpu", weights_only=False)
        z = s["latent_state"].z_pre_norm.float()
        means.append(z.reshape(-1, z.shape[-1]).mean(0).numpy())
        starts.append(pd.Timestamp(p.name[4:]))
    M = np.stack(means).astype(np.float64)
    yrs = np.array([s.year for s in starts])
    _, _, vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    tdir = M[(yrs >= 2170) & (yrs < 2191)].mean(0) - M[(yrs >= 2010) & (yrs < 2031)].mean(0)
    # orient pc2 so that it increases with year (as in the scan)
    pc2 = vt[1] * np.sign(np.corrcoef((M - M.mean(0)) @ vt[1], yrs)[0, 1])
    return pc2, tdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "claude_api_tools/y2209/in5"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pc2, tdir = axes()
    tyr = tdir * 160.0 / (tdir @ tdir)  # projection f . tyr = years along the time direction
    F = {}
    for p in sorted(D1.glob("forcing_decomp_fvnn003_*-01-01.npz")):
        y = int(p.stem.split("_")[-1][:4])
        F[y] = np.load(p)["f_cellmean"].astype(np.float64)
    n = min(len(v) for v in F.values())
    clim = np.mean([v[:n] for v in F.values()], 0)
    rows, daily = [], []
    t0 = {y: pd.Timestamp(f"{y}-01-01") for y in F}
    for y, f in sorted(F.items()):
        f = f[:n]
        dates = [t0[y] + pd.Timedelta(days=k + 1) for k in range(n)]  # forcing index k = day k+1
        ppc2, pt = f @ pc2, f @ tyr
        dpc2, dpt = (f - clim) @ pc2, (f - clim) @ tyr
        cos = (f @ tdir) / (np.linalg.norm(f, axis=1) * np.linalg.norm(tdir))
        aps = np.array([(d.month >= 4) & (d.month <= 9) for d in dates])
        jja = np.array([(d.month >= 6) & (d.month <= 9) for d in dates])
        rows.append({"year": y, "n_days": n,
                     "pc2_mean": ppc2.mean(), "pc2_AprSep_sum": ppc2[aps].sum(),
                     "timedir_years_mean": pt.mean(), "timedir_years_AprSep_sum": pt[aps].sum(),
                     "timedir_years_JunSep_mean": pt[jja].mean(),
                     "anom_timedir_years_AprSep_sum": dpt[aps].sum(),
                     "anom_timedir_years_JunSep_mean": dpt[jja].mean(),
                     "anom_pc2_AprSep_sum": dpc2[aps].sum(),
                     "cos_timedir_mean": cos.mean(), "norm_f_mean": np.linalg.norm(f, axis=1).mean()})
        for k in range(0, n, 5):
            daily.append({"year": y, "date": str(dates[k].date()), "pc2": ppc2[k],
                          "timedir_years": pt[k], "anom_timedir_years": dpt[k],
                          "cos_timedir": cos[k]})
    df = pd.DataFrame(rows)
    with open(out / "forcing_projection_by_year.csv", "w") as fh:
        fh.write("# Cell-mean gated forcing injection f_k (fvnn003, forcing_decomp.py, 363 days from Jan 1 "
                 "of each year) projected on the day-1 scan's latent year axes: pc2 (unit vector, "
                 "oriented to increase with start year) and the time direction scaled to years "
                 "(timedir_years = f . tdir * 160 / |tdir|^2, i.e. the shift along the 2020->2180 axis "
                 "that adding f once would produce). anom_* = projection of f minus the all-years daily "
                 f"mean ({min(F)}-{max(F)}). Scaling the forcing by s scales every projection by s. "
                 "cos_timedir = cosine of f_k with the time direction.\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    with open(out / "forcing_projection_daily.csv", "w") as fh:
        fh.write("# Every 5th day of the projections in forcing_projection_by_year.csv.\n")
        pd.DataFrame(daily).to_csv(fh, index=False, float_format="%.5g")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
