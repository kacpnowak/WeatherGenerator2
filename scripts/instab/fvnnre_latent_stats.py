"""Latent statistics along the deterministic ten-year rerun fvnnre (= fvnn10y, bit-identical).

For every saved state (results/fvnnre/states, every 30 steps; carried latent z_pre_norm after
done_steps forecast steps, valid date 2200-01-01 + done_steps days):
- projections of the cell-mean latent on pc1 (season axis) and pc2 (year axis) of the day-1
  scan (fvd1* states, PCA of the cell-mean latents of all 44 starts, as day1_scan_analyse.py)
  and the time-direction coordinate (mean 2170-2190 starts minus mean 2010-2030 starts, scaled
  to 2020 / 2180);
- per-token RMS mean and std;
- relative L2 distance |z - z_enc| / |z_enc| and cosine similarity of the full latent to the
  fvd1 latent of the nearest Jan/Jul start of the same year (2200-2209), when the state's date
  is within 20 days of that fvd1 latent's date (start + 2 days).
Reference scale: the same distance between fvd1 latents of the same month in adjacent years.

    python scripts/instab/fvnnre_latent_stats.py [--out ~/claude_api_tools/y2209/in4]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESULTS = Path(__file__).resolve().parents[2] / "results"
T0 = pd.Timestamp("2200-01-01")


def load(p):
    s = torch.load(p, map_location="cpu", weights_only=False)
    z = s["latent_state"].z_pre_norm.float().reshape(-1, s["latent_state"].z_pre_norm.shape[-1])
    return int(s.get("done_steps", -1)), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "claude_api_tools/y2209/in4"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # day-1 scan basis
    scan = sorted(p for p in RESULTS.glob("fvd1[0-9]*") if (p / "rollout_state.pt").exists())
    starts, means, enc = [], [], {}
    for p in scan:
        st = pd.Timestamp(p.name[4:])
        _, z = load(p / "rollout_state.pt")
        starts.append(st)
        means.append(z.mean(0).numpy())
        if st.year >= 2199:
            enc[st] = z.half()
    M = np.stack(means)
    yrs = np.array([s.year + (s.dayofyear - 1) / 365.25 for s in starts])
    mu = M.mean(0)
    u, sv, vt = np.linalg.svd(M - mu, full_matrices=False)
    early, late = (yrs >= 2010) & (yrs < 2031), (yrs >= 2170) & (yrs < 2191)
    e0 = M[early].mean(0)
    tdir = M[late].mean(0) - e0
    def coords(m):
        return {"pc1": float((m - mu) @ vt[0]), "pc2": float((m - mu) @ vt[1]),
                "pc3": float((m - mu) @ vt[2]),
                "time_dir_coord_years": float(2020 + 160 * (m - e0) @ tdir / (tdir @ tdir))}
    # reference: same-month adjacent-year distances among fvd1 latents
    ref = []
    for st, z in enc.items():
        nxt = pd.Timestamp(year=st.year + 1, month=st.month, day=1)
        if nxt in enc:
            d = (z.float() - enc[nxt].float()).norm() / enc[nxt].float().norm()
            ref.append({"pair": f"{st.date()} vs {nxt.date()}", "rel_l2": float(d)})
    rows = []
    for p in sorted((RESULTS / "fvnnre" / "states").glob("state_*.pt")):
        n, z = load(p)
        date = T0 + pd.Timedelta(days=n)
        rms = z.pow(2).mean(1).sqrt()
        r = {"done_steps": n, "date": str(date.date()), "token_rms_mean": float(rms.mean()),
             "token_rms_std": float(rms.std()), **coords(z.mean(0).numpy())}
        # nearest fvd1 latent of the same year (dates start + 2 days)
        best = None
        for st, ze in enc.items():
            dd = abs((date - (st + pd.Timedelta(days=2))).days)
            if dd <= 20 and (best is None or dd < best[0]):
                best = (dd, st, ze)
        if best:
            ze = best[2].float()
            r["fvd1_start"] = str(best[1].date())
            r["days_apart"] = best[0]
            r["rel_l2_to_fvd1"] = float((z - ze).norm() / ze.norm())
            r["cos_to_fvd1"] = float((z.flatten() @ ze.flatten()) / (z.norm() * ze.norm()))
        rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates("done_steps").sort_values("done_steps")
    scan_c = pd.DataFrame([{"start": str(s.date()), **coords(m)} for s, m in zip(starts, M)])
    with open(out / "fvnnre_latent_stats.csv", "w") as fh:
        fh.write("# Latent (z_pre_norm) along fvnnre (= fvnn10y, bit-identical) every 30 steps. pc1 / "
                 "pc2 / pc3: projection of the cell-mean latent on the principal axes of the day-1 "
                 "scan's 44 cell-mean latents (pc1 = season axis: Jan vs Jul starts; pc2 = year axis); "
                 "time_dir_coord_years: projection on mean(2170-2190 starts) - mean(2010-2030 starts) "
                 "scaled to 2020/2180. rel_l2_to_fvd1 / cos_to_fvd1: full-latent distance to the fvd1 "
                 "latent (fresh encoding + 2 steps) of the nearest Jan/Jul start of 2199-2209 within 20 "
                 f"days. Reference: same-month adjacent-year fvd1 distances rel_l2 "
                 f"{np.min([x['rel_l2'] for x in ref]):.3f}-{np.max([x['rel_l2'] for x in ref]):.3f} "
                 f"(median {np.median([x['rel_l2'] for x in ref]):.3f}).\n")
        df.to_csv(fh, index=False, float_format="%.5g")
    with open(out / "fvd1_scan_coords.csv", "w") as fh:
        fh.write("# Day-1 scan starts in the same coordinates as fvnnre_latent_stats.csv.\n")
        scan_c.to_csv(fh, index=False, float_format="%.5g")
    pd.set_option("display.width", 200)
    print(df[["date", "pc1", "pc2", "time_dir_coord_years", "rel_l2_to_fvd1", "token_rms_mean"]]
          .iloc[::4].round(3).to_string())
    print(df[df.date >= "2209-01-01"][["date", "pc1", "pc2", "time_dir_coord_years",
                                         "rel_l2_to_fvd1"]].round(3).to_string())


if __name__ == "__main__":
    main()
