"""
Old-lineage comparison (combined_report section 5 step 2): side-by-side table of the probe
results of frcwin03 (the yr1win03 weights) and fvars002 / fvars003, and figures.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/oldlineage_report.py

Inputs: OUT/data/gain_frcwin03_dNNN.json, OUT/data/traj_<model>_dD.json/.npz (instab_probe.py
--mode gain / traj, sbatch_instab_oldlineage.sh) and the fesom_vars gain probes of TEST 2
(instab_tests/data/gain_fvars00X_dNNN.json). -> OUT/data/probe_table.txt, FIG/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("/home/a/a270225/claude_api_tools/oldlineage")
DATA = OUT / "data"
OLD_GAIN = Path("/home/a/a270225/claude_api_tools/instab_tests/data")
FIG = Path("/work/ab0995/a270225/WeatherGenerator2/paper_eval/figures/oldlineage")
MODELS = ("frcwin03", "fvars002", "fvars003")
LABEL = {"frcwin03": "old (frcwin03 = yr1win03)", "fvars002": "fvars002", "fvars003": "fvars003"}
DAYS = (10, 40, 70)
EPS = "1e-03"


def gain(m, d):
    p = (DATA if m == "frcwin03" else OLD_GAIN) / f"gain_{m}_d{d:03d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def traj(m):
    ps = sorted(DATA.glob(f"traj_{m}_d*.json"))
    if not ps:
        return None, None
    r = json.loads(ps[-1].read_text())
    T = {k: np.asarray(v, float) for k, v in r["trajectory"].items()}
    return r, T


def at(T, key, d, half=0):
    """Value at day d (half = 0) or the mean over days d-half .. d+half."""
    n = T["n"]
    sel = (n >= d - half) & (n <= d + half)
    return float(np.nanmean(T[key][sel])) if sel.any() else np.nan


def fmt(x, f="{:.3f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def table():
    L = ["Side-by-side probe results, free rollouts from 2209-01-01 (day N = lead N)",
         "  gain: |J v| for |v| = 1, J v = (F(z_N + e v) - F(z_N - e v)) / 2e, e = 1e-3 |z_N|, fp32",
         "    (exact SDPA attention, TF32 off); F = forcing injection + forecast engine of day N+1;",
         "    v = random Gaussian (mean of 3 seeds), the increment z_(N+1) - z_N, the second",
         "    difference z_(N+1) - 2 z_N + z_(N-1); q = <v, J v>.",
         "  latent statistics from the traj probe (same rollout, bf16 as the inference):",
         "    |dz|/|z| = |z_N - z_(N-1)| / |z_N|; cos = cos(z_N - z_(N-1), z_(N-1) - z_(N-2));",
         "    5-day means over N-2..N+2 in brackets; forcing = |F_in(z_N) - z_N| / |z_N| of the",
         "    step N -> N+1 (fp32); land/ocean = RMS token value over the HEALPix level-5 cells",
         "    without / with a mesh node (2863 land, 9425 ocean cells), i.e. sqrt(mean_cells",
         "    mean_dim z^2).", ""]
    rows = []
    for d in DAYS:
        for m in MODELS:
            g = gain(m, d)
            r, T = traj(m)
            row = {"day": d, "model": m}
            if g is not None:
                f = g["fp32"]
                row["g_rand"] = np.mean([f[k][EPS]["gain"] for k in f if k.startswith("random")])
                row["g_inc"] = f["increment"][EPS]["gain"]
                row["g_d2"] = f["second_difference"][EPS]["gain"]
                row["q_inc"] = f["increment"][EPS]["q"]
                row["q_d2"] = f["second_difference"][EPS]["q"]
                row["g_d2_fp64"] = g.get("fp64", {}).get("second_difference", {}).get(
                    "1e-08", {}).get("gain")
                row["forcing_gain"] = g["forcing_injection_rel"]
                row["znorm_gain"] = g["z_norm"]
            if T is not None:
                row["d1"], row["d1m"] = at(T, "d1_rel", d), at(T, "d1_rel", d, 2)
                row["cos"], row["cosm"] = at(T, "cos_inc", d), at(T, "cos_inc", d, 2)
                row["cos_oc"], row["cos_la"] = at(T, "cos_inc_ocean", d), at(T, "cos_inc_land", d)
                row["forcing"] = at(T, "forcing_rel", d)
                row["rms_oc"], row["rms_la"] = at(T, "cell_rms_ocean", d), at(T, "cell_rms_land", d)
                row["d1_land_share"] = at(T, "d1_land_energy_share", d)
                row["forcing_land_share"] = at(T, "forcing_land_energy_share", d)
                row["znorm"] = at(T, "z_norm", d)
            rows.append(row)
    head = (f"  {'day':>3} {'model':<9} {'g_rand':>7} {'g_inc':>7} {'g_d2':>7} {'q_inc':>7} "
            f"{'q_d2':>7} {'|dz|/|z| (5d)':>15} {'cos (5d)':>16} {'cos oc/la':>13} "
            f"{'forcing':>8} {'rms ocean':>10} {'rms land':>9} {'land/ocean':>10} {'|z|':>8}")
    L += [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        L.append(f"  {r['day']:>3} {r['model']:<9} {fmt(r.get('g_rand')):>7} {fmt(r.get('g_inc')):>7} "
                 f"{fmt(r.get('g_d2')):>7} {fmt(r.get('q_inc'), '{:+.3f}'):>7} "
                 f"{fmt(r.get('q_d2'), '{:+.3f}'):>7} "
                 f"{fmt(r.get('d1')) + ' (' + fmt(r.get('d1m')) + ')':>15} "
                 f"{fmt(r.get('cos'), '{:+.3f}') + ' (' + fmt(r.get('cosm'), '{:+.3f}') + ')':>16} "
                 f"{fmt(r.get('cos_oc'), '{:+.2f}') + '/' + fmt(r.get('cos_la'), '{:+.2f}'):>13} "
                 f"{fmt(r.get('forcing')):>8} {fmt(r.get('rms_oc')):>10} {fmt(r.get('rms_la')):>9} "
                 f"{fmt(r.get('rms_la', np.nan) / r.get('rms_oc', np.nan)):>10} "
                 f"{fmt(r.get('znorm'), '{:.0f}'):>8}")
    L.append("")
    L.append("  checks: forcing share of the gain probe (fp32, same definition) and fp64 gain of the")
    L.append("  second difference at e = 1e-8 |z|; |z_N| of the gain probe vs the traj probe")
    for r in rows:
        L.append(f"  {r['day']:>3} {r['model']:<9} forcing(gain) {fmt(r.get('forcing_gain'))}  "
                 f"g_d2 fp64 {fmt(r.get('g_d2_fp64'))}  |z| gain {fmt(r.get('znorm_gain'), '{:.1f}')}"
                 f" traj {fmt(r.get('znorm'), '{:.1f}')}  land share of |dz|^2 "
                 f"{fmt(r.get('d1_land_share'))}  of forcing^2 {fmt(r.get('forcing_land_share'))}")
    L.append("")
    # along the rollouts: land / ocean token RMS
    L.append("Token RMS on ocean / land cells along the rollouts (and their ratio)")
    days = (1, 10, 20, 40, 58, 70, 78, 100, 120, 150, 200, 250, 300, 360)
    L.append(f"  {'model':<9}" + "".join(f"{d:>16}" for d in days))
    for m in MODELS:
        _, T = traj(m)
        if T is None:
            continue
        L.append(f"  {m:<9}" + "".join(
            f"{fmt(at(T, 'cell_rms_ocean', d), '{:.3f}') + '/' + fmt(at(T, 'cell_rms_land', d), '{:.3f}'):>16}"
            if d <= T['n'].max() else f"{'':>16}" for d in days))
        L.append(f"  {'  ratio':<9}" + "".join(
            f"{fmt(at(T, 'cell_rms_land', d) / at(T, 'cell_rms_ocean', d), '{:.3f}'):>16}"
            if d <= T['n'].max() else f"{'':>16}" for d in days))
    L.append("")
    L.append("Window means along the rollouts: |dz|/|z|, cos of consecutive increments (all /")
    L.append("ocean / land cells), forcing share, land share of the increment energy")
    wins = ((3, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 120), (121, 150),
            (151, 240), (241, 360))
    L.append(f"  {'model':<9}{'quantity':<22}" + "".join(f"{f'{a}-{b}':>9}" for a, b in wins))
    for m in MODELS:
        _, T = traj(m)
        if T is None:
            continue
        for k in ("d1_rel", "cos_inc", "cos_inc_ocean", "cos_inc_land", "forcing_rel",
                  "d1_land_energy_share", "d2_rel"):
            vals = []
            for a, b in wins:
                sel = (T["n"] >= a) & (T["n"] <= b)
                vals.append(f"{np.nanmean(T[k][sel]):>9.3f}" if sel.any() else f"{'':>9}")
            L.append(f"  {m:<9}{k:<22}" + "".join(vals))
    (DATA / "probe_table.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    col = {"frcwin03": "k", "fvars002": "tab:blue", "fvars003": "tab:red"}
    keys = (("z_norm", "|z|"), ("d1_rel", "|z_n - z_(n-1)| / |z_n|"),
            ("cos_inc", "cos of consecutive increments"), ("forcing_rel", "forcing injection / |z|"),
            ("cell_rms_ocean", "token RMS, ocean cells"), ("cell_rms_land", "token RMS, land cells"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (k, t) in zip(axes.ravel(), keys):
        for m in MODELS:
            _, T = traj(m)
            if T is None:
                continue
            ax.plot(T["n"], T[k], color=col[m], lw=0.8, label=LABEL[m])
        ax.set_title(t)
        ax.set_xlabel("day")
    for d, ls in ((58, ":"), (78, "--")):
        for ax in axes.ravel():
            ax.axvline(d, color="0.6", ls=ls, lw=0.6)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Latent statistics along the free rollouts from 2209-01-01 "
                 "(dotted/dashed: fvars002/fvars003 blow-up days 58/78)")
    fig.savefig(FIG / "traj_latent_stats.png", dpi=130)
    # maps of per-cell token RMS
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(15, 3.6 * len(MODELS)),
                             constrained_layout=True)
    for i, m in enumerate(MODELS):
        ps = sorted(DATA.glob(f"traj_{m}_d*.npz"))
        if not ps:
            continue
        z = np.load(ps[-1])
        lat, lon, oc = z["cell_lat"], z["cell_lon"], z["ocean_cells"]
        for j, d in enumerate((10, 40, 70)):
            ax = axes[i, j]
            v = z[f"cell_rms_d{d:03d}"]
            sc = ax.scatter(lon, lat, c=v, s=3, cmap="viridis", vmin=0, vmax=2.0)
            ax.scatter(lon[~oc], lat[~oc], s=0.3, c="w", alpha=0.3)
            ax.set_title(f"{m} day {d}: token RMS per cell (white dots: land cells)", fontsize=9)
            fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.savefig(FIG / "traj_cell_rms_maps.png", dpi=110)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    table()
    figures()
