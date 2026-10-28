"""
Old-lineage counterpart of the fvars trajectory statistics (Fable diagnostic D8): frcwin03
(rollout config of yr1win03) vs fvars003, free rollouts from 2209-01-01, from the traj probe
(instab_probe.py --mode traj, with the cumulative-forcing statistics).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/d8_traj_table.py \
        --data DIR --out DIR [--fig DIR]

Reads DIR/traj_frcwin03_d360.json and DIR/traj_fvars003_d150.json (this probe run),
the fvars003 default / freeze trajectories of the landcells study and the encoded-truth
|F_n| / |z_true_n| of latent_error.py (latenterr) for cross-checks.
Per day n: forcing injection |f_(n+1)| / |z_n| (step n -> n+1), |F_n| / |z_n| with
F_n = sum_(k<=n) f_k, coherence |F_n| / sum_k |f_k|, land / ocean token RMS, land share of
the increment energy |dz_n|^2, cos of consecutive increments (land / ocean cells),
|dz_n| / |z_n|. Fits |F_n| = a n^beta (log-log least squares) over several windows.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HOME = Path("/home/a/a270225/claude_api_tools")
REF = {"fvars003 freeze (landcells)": HOME / "landcells/data/traj_fvars003_freeze_d150.json",
       "fvars003 default (landcells)": HOME / "landcells/data/traj_fvars003_none_d150.json",
       "frcwin03 (oldlineage, earlier probe)": HOME / "oldlineage/data/traj_frcwin03_d360.json"}
LATENTERR = HOME / "latenterr/data/latenterr_fvars003_2209-01-01.json"
COLS = [("forcing_rel", "inj |f|/|z|", "{:.3f}"), ("F_cum_rel", "|F|/|z|", "{:.2f}"),
        ("F_coherence", "|F|/sum|f|", "{:.3f}"), ("cell_rms_land", "land RMS", "{:.3f}"),
        ("cell_rms_ocean", "ocean RMS", "{:.3f}"),
        ("d1_land_energy_share", "land share |dz|^2", "{:.3f}"),
        ("cos_inc_land", "cos_inc_land", "{:+.2f}"), ("cos_inc_ocean", "cos_inc_ocean", "{:+.2f}"),
        ("d1_rel", "|dz|/|z|", "{:.3f}")]
#: further quantities, in the window-mean table only
WIN_ONLY = [("z_norm", "|z|", "{:.0f}"), ("f_norm", "|f|", "{:.0f}"),
            ("cos_f_Fprev", "cos(f_n, F_(n-1))", "{:+.3f}"),
            ("forcing_cell_rms_land", "forcing cell RMS land", "{:.3f}"),
            ("forcing_cell_rms_ocean", "forcing cell RMS ocean", "{:.3f}"),
            ("forcing_land_energy_share", "land share |f|^2", "{:.3f}"),
            ("F_cum_land_energy_share", "land share |F|^2", "{:.3f}")]
DAYS = list(range(1, 31)) + list(range(35, 151, 5))
EXTRA_DAYS = [180, 210, 240, 270, 300, 330, 360]
WINDOWS = [(1, 150), (1, 30), (10, 150), (30, 150), (60, 150), (150, 360)]


def traj(path):
    return {k: np.asarray(v, float) for k, v in json.loads(Path(path).read_text())
            ["trajectory"].items()}


def val(T, key, n):
    i = np.flatnonzero(T["n"] == n)
    return T[key][i[0]] if len(i) and key in T else np.nan


def fit(T, a, b):
    n, F = T["n"], T.get("F_cum_norm")
    if F is None:
        return np.nan, np.nan
    sel = (n >= a) & (n <= b) & np.isfinite(F) & (F > 0)
    if sel.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(n[sel]), np.log(F[sel])
    c = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c[0], r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fig", default=None)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M = {"frcwin03": traj(data / "traj_frcwin03_d360.json"),
         "fvars003": traj(data / "traj_fvars003_d150.json")}
    R = {k: traj(v) for k, v in REF.items() if v.exists()}
    L = []
    L += ["# D8: old-lineage (frcwin03) vs fvars003 latent trajectory statistics, 2209-01-01",
          "", "Per day n of the free rollout (bf16, as inference). inj = |f_(n+1)|/|z_n| "
          "(forcing injection of the step n -> n+1, fp32); |F|/|z| = |sum_(k<=n) f_k| / |z_n|; "
          "|F|/sum|f| = coherence (1: one fixed direction; ~n^-1/2: independent steps); land / "
          "ocean RMS = sqrt(mean over cells and channels of z^2) on land (2863 cells without a "
          "mesh node) / ocean cells; land share |dz|^2 = land fraction of |z_n - z_(n-1)|^2; "
          "cos_inc = cos(dz_n, dz_(n-1)) on land / ocean cells; |dz|/|z| = |z_n - z_(n-1)|/|z_n|.",
          ""]
    hdr = " | ".join(f"{m} {c[1]}" for c in COLS for m in M)
    L += ["## Side by side (columns alternate frcwin03 / fvars003)", "",
          f"| day | {hdr} |", "|---|" + "---|" * (len(COLS) * len(M))]
    for n in DAYS + EXTRA_DAYS:
        cells = []
        for key, _, f in COLS:
            for m in M:
                v = val(M[m], key, n)
                cells.append("-" if not np.isfinite(v) else f.format(v))
        if n > 150 and all(c == "-" for c in cells[1::2]) and all(c == "-" for c in cells[::2]):
            continue
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")
    # growth of |F_n|
    L += ["## Growth of |F_n| = |sum f_k|: power-law exponent beta (|F_n| ~ n^beta; 1 = "
          "constant direction, 0.5 = independent daily injections)", "",
          "| model | " + " | ".join(f"beta {w[0]}-{w[1]} (R2)" for w in WINDOWS)
          + " | |F_150|/|z_150| | |F_150|/(150 mean|f|) | mean |f|/|z| d1-150 "
          "| |F| per day / mean|z| (d10-150 slope) |",
          "|---|" + "---|" * (len(WINDOWS) + 4)]
    for m, T in M.items():
        bs = []
        for w in WINDOWS:
            b, r2 = fit(T, *w)
            bs.append("-" if not np.isfinite(b) else f"{b:.3f} ({r2:.3f})")
        n = T["n"]
        sel = (n >= 1) & (n <= 150)
        zmean = np.nanmean(T["z_norm"][sel])
        s10 = (n >= 10) & (n <= 150)
        slope = np.polyfit(n[s10], T["F_cum_norm"][s10], 1)[0] / zmean
        mf = np.nanmean(T["f_norm"][sel])
        L.append(f"| {m} | " + " | ".join(bs)
                 + f" | {val(T, 'F_cum_rel', 150):.2f} | {val(T, 'F_cum_norm', 150) / (150 * mf):.3f}"
                 f" | {mf / zmean:.3f} | {slope:.3f} |")
    L.append("")
    # summary per window
    L += ["## Window means (days)", "",
          "| model | quantity | 1-10 | 11-30 | 31-60 | 61-90 | 91-150 | 151-360 |",
          "|---|---|---|---|---|---|---|---|"]
    wins = [(1, 10), (11, 30), (31, 60), (61, 90), (91, 150), (151, 360)]
    for m, T in M.items():
        for key, name, f in COLS + WIN_ONLY:
            if key not in T:
                continue
            cells = []
            for lo, hi in wins:
                s = (T["n"] >= lo) & (T["n"] <= hi)
                v = np.nanmean(T[key][s]) if s.any() and np.isfinite(T[key][s]).any() else np.nan
                cells.append("-" if not np.isfinite(v) else f.format(v))
            L.append(f"| {m} | {name} | " + " | ".join(cells) + " |")
    L.append("")
    # cross-checks
    L += ["## Cross-checks", ""]
    for name, T in R.items():
        m = "frcwin03" if "frcwin03" in name else "fvars003"
        if "freeze" in name:
            continue
        nmax = int(min(T["n"].max(), M[m]["n"].max()))
        dif = {}
        for key in ("z_norm", "forcing_rel", "cell_rms_land", "cos_inc_land", "d1_rel"):
            x = np.array([val(T, key, n) for n in range(3, nmax + 1)])
            y = np.array([val(M[m], key, n) for n in range(3, nmax + 1)])
            dif[key] = float(np.nanmax(np.abs(x - y)))
        L.append(f"- {name} vs this run ({m}, days 3-{nmax}): max |difference| "
                 + ", ".join(f"{k} {v:.3g}" for k, v in dif.items()))
    if LATENTERR.exists():
        le = json.loads(LATENTERR.read_text())["runs"]["none"]["trajectory"]
        n = np.asarray(le["n"])
        Fr = np.asarray(le["all.F_rel"], float)
        T = M["fvars003"]
        rows = []
        for d in (1, 10, 40, 80, 150):
            i = np.flatnonzero(n == d)
            if len(i):
                rows.append(f"day {d}: latenterr |F|/|z_true| {Fr[i[0]]:.2f}, "
                            f"this run |F|/|z| {val(T, 'F_cum_rel', d):.2f}")
        L.append("- fvars003 cumulative forcing vs latent_error.py (denominator |z_true_n| "
                 "there, |z_n| here): " + "; ".join(rows))
    if "fvars003 freeze (landcells)" in R:
        T = R["fvars003 freeze (landcells)"]
        L.append("- fvars003 freeze (landcells), for reference: land RMS "
                 + ", ".join(f"d{d} {val(T, 'cell_rms_land', d):.3f}" for d in (1, 40, 150))
                 + "; inj " + ", ".join(f"d{d} {val(T, 'forcing_rel', d):.3f}" for d in (1, 40, 150)))
    L.append("")
    txt = "\n".join(L) + "\n"
    (out / "tables.md").write_text(txt)
    print(txt)  # noqa: T201

    if a.fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        panels = [("forcing_rel", "injection |f|/|z|"), ("F_cum_norm", "|F_n| (abs.)"),
                  ("cell_rms_land", "land token RMS"),
                  ("d1_land_energy_share", "land share of |dz|^2"),
                  ("cos_inc_land", "cos consecutive increments (solid land, dashed ocean)"),
                  ("d1_rel", "|dz|/|z|")]
        col = {"frcwin03": "tab:blue", "fvars003": "tab:red"}
        for ax, (key, title) in zip(axs.flat, panels, strict=True):
            for m, T in M.items():
                ax.plot(T["n"], T[key], color=col[m], label=m)
                if key == "cos_inc_land":
                    ax.plot(T["n"], T["cos_inc_ocean"], color=col[m], ls="--")
            if key == "F_cum_norm":
                ax.set_xscale("log")
                ax.set_yscale("log")
                for m, T in M.items():
                    n = T["n"]
                    F1 = T["F_cum_norm"][0]
                    ax.plot(n, F1 * n, color=col[m], lw=0.5, ls=":")
                    ax.plot(n, F1 * np.sqrt(n), color=col[m], lw=0.5, ls="-.")
                title += " (dotted ~n, dash-dot ~sqrt n)"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("day")
        axs[0, 0].legend()
        fig.tight_layout()
        Path(a.fig).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(a.fig) / "d8_traj_oldlineage.png", dpi=130)


if __name__ == "__main__":
    main()
