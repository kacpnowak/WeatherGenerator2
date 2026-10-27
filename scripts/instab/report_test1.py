"""
TEST 1 of the instab tests: where the day-1 cold offset comes from.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/report_test1.py

Reads DATA/encdec_<model>_<date>.json / _maps.npz written by instab_probe.py --mode encdec
and writes OUT/test1_offset.txt and FIG/test1_*.png.
Step 0 = decode(E(x0)) without a forecast-engine step (out of distribution, see the probe);
step k >= 1 = the normal rollout's lead k.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("/home/a/a270225/claude_api_tools/instab_tests")
DATA = OUT / "data"
FIG = Path("/work/ab0995/a270225/WeatherGenerator2/paper_eval/figures/instab_tests")
MODELS = ("fvars002", "fvars003")
DATES = ("2150-01-01", "2190-07-01", "2200-01-01", "2205-01-01", "2208-01-01", "2208-07-01",
         "2209-01-01", "2209-07-01")
LEVELS = ("temp_2", "temp_25", "temp_75", "temp_125", "temp_255", "temp_450", "temp_975",
          "temp_1600", "temp_2035")
SLEVELS = ("salt_2", "salt_75", "salt_125", "salt_450", "salt_975", "salt_2035")
MAPCH = ("temp_2", "temp_75", "temp_125", "temp_450", "temp_975", "temp_2035", "salt_75", "ssh")


def load():
    res, maps = {}, {}
    for m in MODELS:
        for d in DATES:
            p = DATA / f"encdec_{m}_{d}.json"
            if p.exists():
                res[(m, d)] = json.loads(p.read_text())
                maps[(m, d)] = dict(np.load(DATA / f"encdec_{m}_{d}_maps.npz"))
    return res, maps


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    res, maps = load()
    if not res:
        sys.exit("no encdec results")
    L = ["TEST 1: origin of the day-1 cold offset (instab_probe.py --mode encdec)",
         "step 0 = decode(E(x0)) with no forecast-engine step (out of distribution: the models",
         "  were trained with forecast offset 1 only); step k = lead k of the normal rollout.",
         "Errors = prediction - truth, wet-area-weighted global means; vmT/vmS = volume means",
         "  0-2150 m (31 levels); hc = heat content 0-2150 m (ZJ); norm = divided by the stream's",
         "  normalisation std (TEMP/SALT: config/streams/fesom/wet_stats_node.json).",
         "uniform share = mean_err^2 / rmse^2 (1: a uniform offset; ~0: a zero-mean pattern).", ""]
    # column summary
    L.append("Volume-mean T error (K) / heat content error (ZJ) / volume-mean S error (psu)")
    head = f"  {'model':<9}{'init':<12}" + "".join(f"{'s' + str(s):>22}" for s in range(6))
    L.append(head)
    for (m, d), r in res.items():
        row = f"  {m:<9}{d:<12}"
        for s in range(6):
            c = r["steps"][str(s)]["column"]
            row += f"{c['vmT_err']:>+9.4f}/{c['hc_err_ZJ']:>+6.0f}/{c['vmS_err']:>+6.3f}"
        L.append(row)
    L.append("")
    for kind, chans in (("T", LEVELS), ("S", SLEVELS)):
        for key, unit in (("mean_err", "K" if kind == "T" else "psu"), ("mean_err_norm", "norm"),
                          ("rmse", "K" if kind == "T" else "psu")):
            L.append(f"Per-level {key} ({unit}), steps 0 / 1 / 5")
            L.append(f"  {'model':<9}{'init':<12}" + "".join(f"{c:>24}" for c in chans))
            for (m, d), r in res.items():
                row = f"  {m:<9}{d:<12}"
                for c in chans:
                    v = [r["steps"][str(s)]["levels"][c][key] for s in (0, 1, 5)]
                    row += f"{v[0]:>+8.3f}{v[1]:>+8.3f}{v[2]:>+8.3f}"
                L.append(row)
            L.append("")
    L.append("Uniform share mean_err^2/rmse^2 at step 1 (and step 0)")
    L.append(f"  {'model':<9}{'init':<12}" + "".join(f"{c:>16}" for c in LEVELS))
    for (m, d), r in res.items():
        row = f"  {m:<9}{d:<12}"
        for c in LEVELS:
            v1 = r["steps"]["1"]["levels"][c]
            v0 = r["steps"]["0"]["levels"][c]
            row += f"{v1['mean_err']**2 / v1['rmse']**2:>8.3f}({v0['mean_err']**2 / v0['rmse']**2:.3f})"
        L.append(row)
    L.append("")
    L.append("Normalisation: truth's wet-area mean at step 0 in normalised units, (mean - stat_mean)/stat_std,")
    L.append("  and the plain wet-node mean; and the stats used")
    m0, d0 = next(iter(res))
    r0 = res[(m0, d0)]
    L.append(f"  {'init':<12}" + "".join(f"{c:>20}" for c in LEVELS))
    for d in DATES:
        have = [m for m in MODELS if (m, d) in res]
        if not have:
            continue
        lv = res[(have[0], d)]["steps"]["0"]["levels"]
        L.append(f"  {d:<12}" + "".join(
            f"{lv[c]['truth_mean_norm']:>+10.3f}{lv[c]['truth_nodemean_norm']:>+10.3f}" for c in LEVELS))
    L.append(f"  {'stat mean':<12}" + "".join(f"{r0['steps']['0']['levels'][c]['stat_mean']:>20.3f}"
                                             for c in LEVELS))
    L.append(f"  {'stat std':<12}" + "".join(f"{r0['steps']['0']['levels'][c]['stat_std']:>20.3f}"
                                            for c in LEVELS))
    L.append("")
    L.append("Token statistics of the latent (z0 = encoder output, zk = forecast-engine output)")
    for (m, d), r in res.items():
        t = r["tokens"]
        L.append(f"  {m} {d}: rms z0 {t['0']['rms']:.3f}, z1 {t['1']['rms']:.3f}, z5 {t['5']['rms']:.3f};"
                 f" median token norm/sqrt(d) z0 {t['0']['token_norm_p50']:.3f}, z1 {t['1']['token_norm_p50']:.3f};"
                 f" |z1-z0|/|z1| {t['rel_dist_z1_z0']:.3f}, cos(z1,z0) {t['cos_z1_z0']:.3f}")
    L.append("")
    # spatial pattern correlations
    L.append("Spatial correlation of error maps (wet nodes)")
    L.append("  (a) step-1 error vs step-0 error, same model and date")
    L.append(f"  {'model':<9}{'init':<12}" + "".join(f"{c:>11}" for c in MAPCH))
    for (m, d), mp in maps.items():
        L.append(f"  {m:<9}{d:<12}" + "".join(
            f"{corr(mp[f'err_s1_{c}'], mp[f'err_s0_{c}']):>11.2f}" for c in MAPCH))
    L.append("  (b) step-1 error vs step-1 error of the other initial dates (mean over date pairs)")
    for m in MODELS:
        ds = [d for d in DATES if (m, d) in maps]
        row = f"  {m:<21}"
        for c in MAPCH:
            cs = [corr(maps[(m, a)][f"err_s1_{c}"], maps[(m, b)][f"err_s1_{c}"])
                  for i, a in enumerate(ds) for b in ds[i + 1:]]
            row += f"{np.mean(cs):>11.2f}" if cs else f"{'-':>11}"
        L.append(row)
    L.append("  (c) step-1 error fvars002 vs fvars003, same date")
    for d in DATES:
        if ("fvars002", d) in maps and ("fvars003", d) in maps:
            L.append(f"  {d:<21}" + "".join(
                f"{corr(maps[('fvars002', d)][f'err_s1_{c}'], maps[('fvars003', d)][f'err_s1_{c}']):>11.2f}"
                for c in MAPCH))
    L.append("  (d) step-5 error vs step-1 error, same model and date")
    for (m, d), mp in maps.items():
        L.append(f"  {m:<9}{d:<12}" + "".join(
            f"{corr(mp[f'err_s5_{c}'], mp[f'err_s1_{c}']):>11.2f}" for c in MAPCH))
    L.append("  (e) lead-1 error regressed on the truth in normalised units (wet nodes):")
    L.append("      err/std = a + b (truth - stat_mean)/std; b < 0 = pull toward the normalisation mean")
    L.append(f"  {'model':<9}{'init':<12}" + "".join(f"{c:>20}" for c in MAPCH))
    for (m, d), mp in maps.items():
        row = f"  {m:<9}{d:<12}"
        for c in MAPCH:
            key = f"true_s1_{c}"
            if key not in mp:
                row += f"{'-':>20}"
                continue
            lv = res[(m, d)]["steps"]["1"]["levels"][c]
            x = (mp[key] - lv["stat_mean"]) / lv["stat_std"]
            y = mp[f"err_s1_{c}"] / lv["stat_std"]
            ok = np.isfinite(x) & np.isfinite(y)
            b, a = np.polyfit(x[ok], y[ok], 1)
            row += f"  a{a:+.4f} b{b:+.4f}"
        L.append(row)
    L.append("")
    L.append("  (f) lead-1 mean error by crude basin (wet-area-weighted, K), temp_450 / temp_975")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import diag_fvars_year as dfy

    geo = dfy.Geometry("/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc")
    basin = dfy.basins(geo.lon, geo.lat)
    L.append(f"  {'model':<9}{'init':<12}" + "".join(f"{b[:12]:>18}" for b in dfy.BASINS))
    for (m, d), mp in maps.items():
        row = f"  {m:<9}{d:<12}"
        for bi in range(len(dfy.BASINS)):
            vals = []
            for c in ("temp_450", "temp_975"):
                mask, w = geo.weight(c)
                e = mp[f"err_s1_{c}"]
                sel = mask & (basin == bi) & np.isfinite(e)
                vals.append((w[sel] * e[sel]).sum() / w[sel].sum())
            row += f"{vals[0]:>+9.3f}/{vals[1]:>+7.3f} "
        L.append(row)
    L.append("")
    L.append("coordinate check (max |dlat|, |dlon| target coords vs mesh nodes): " + json.dumps(
        r0.get("coord_check_max_dlat_dlon")))
    txt = "\n".join(L) + "\n"
    (OUT / "test1_offset.txt").write_text(txt)
    print(txt)
    figures(res, maps)


def figures(res, maps):
    FIG.mkdir(parents=True, exist_ok=True)
    allt = [f"temp_{z}" for z in (2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210,
                                  255, 310, 375, 450, 535, 630, 735, 850, 975, 1110, 1255, 1415,
                                  1600, 1810, 2035)]
    depth = [int(c.split("_")[1]) for c in allt]
    fig, axs = plt.subplots(2, 4, figsize=(17, 9), sharey=True, constrained_layout=True)
    for row, m in enumerate(MODELS):
        for col, (s, key, xl) in enumerate(((0, "mean_err", "step 0 mean error (K)"),
                                            (1, "mean_err", "lead 1 mean error (K)"),
                                            (5, "mean_err", "lead 5 mean error (K)"),
                                            (1, "mean_err_norm", "lead 1 mean error (norm.)"))):
            ax = axs[row, col]
            for i, d in enumerate(DATES):
                if (m, d) in res:
                    lv = res[(m, d)]["steps"][str(s)]["levels"]
                    ax.plot([lv[c][key] for c in allt], depth, "o-", ms=3, label=d,
                            color=plt.cm.viridis(i / (len(DATES) - 1)) if d < "2209" else
                            ("tab:red" if d == "2209-01-01" else "tab:orange"))
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_xlabel(xl)
            ax.set_title(m)
            ax.grid(alpha=0.3)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_ylabel("depth (m)")
    axs[1, 0].set_ylabel("depth (m)")
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle("Global wet-area mean T error per level (prediction - truth) by initial date")
    fig.savefig(FIG / "test1_temp_offset_profiles.png", dpi=110)
    plt.close(fig)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper_eval.meshtools import load_mesh

    mesh = load_mesh("/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc")
    for m in MODELS:
        ds = [d for d in DATES if (m, d) in maps]
        if not ds:
            continue
        chans = ("temp_2", "temp_75", "temp_450", "temp_975")
        fig, axs = plt.subplots(len(chans), len(ds) + 1, figsize=(4 * (len(ds) + 1), 2.6 * len(chans)),
                                constrained_layout=True)
        for i, c in enumerate(chans):
            vals = [maps[(m, d)][f"err_s1_{c}"] for d in ds]
            lim = np.nanpercentile(np.abs(np.concatenate(vals)), 99)
            for j, d in enumerate(ds + ["step0"]):
                ax = axs[i, j]
                v = maps[(m, "2209-01-01" if d == "step0" else d)][
                    f"err_s0_{c}" if d == "step0" else f"err_s1_{c}"]
                ok = np.isfinite(v)
                sc = ax.scatter(mesh.lon[ok], mesh.lat[ok], c=v[ok], s=0.3, cmap="RdBu_r",
                                vmin=-lim, vmax=lim, rasterized=True)
                ax.set_title(f"{c} {'step 0, 2209-01-01' if d == 'step0' else 'lead 1, ' + d}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(sc, ax=axs[i, -1], shrink=0.8, label="K")
        fig.suptitle(f"{m}: prediction - truth, lead 1 for each initial date; last column step 0 "
                     "(colour range of the lead-1 maps)")
        fig.savefig(FIG / f"test1_maps_{m}.png", dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
