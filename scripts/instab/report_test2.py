"""
TEST 2 of the instab tests: leading eigenvalues/eigenvectors of the Jacobian of the trained
one-day step around states of the free rollout from 2209-01-01.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/report_test2.py

Reads DATA/linear_<model>_dNNN.json / .npz (instab_probe.py --mode linear) and writes
OUT/test2_jacobian.txt and FIG/test2_*.png.
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
GROUPS = (("0-50 m", 0, 50), ("50-150 m", 50, 150), ("150-500 m", 150, 500),
          ("500-2035 m", 500, 2100))


def depth(c):
    return int(c.split("_")[1])


def group_shares(per_ch):
    """Share of the summed squared normalised response per stream / depth group."""
    e = {c: v["rms_norm"] ** 2 for c, v in per_ch.items()}
    tot = sum(e.values())
    out = {}
    for var in ("temp", "salt"):
        for name, a, b in GROUPS:
            out[f"{var} {name}"] = sum(v for c, v in e.items()
                                       if c.startswith(var + "_") and a <= depth(c) < b) / tot
    for c in ("ssh", "a_ice", "m_ice"):
        out[c] = e[c] / tot
    return out


def ocean_cells(level=5):
    """True for the HEALPix cells (nested, the model's convention) holding at least one mesh
    node; the others are land cells without ocean data."""
    from astropy_healpix import healpy as ahp

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper_eval.meshtools import load_mesh

    mesh = load_mesh("/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc")
    theta = np.radians(90.0 - mesh.lat)
    phi = np.radians(180.0 + mesh.lon)
    pix = ahp.ang2pix(2**level, theta, phi, nest=True)
    has = np.zeros(12 * 4**level, bool)
    has[pix] = True
    return has


def load():
    res = {}
    for p in sorted(DATA.glob("linear_*_d*.json")):
        r = json.loads(p.read_text())
        res[(r["tag"], r["day"])] = (r, p.with_suffix(".npz"))
    return res


def fmt_l(re, im):
    return f"{re:+.3f}{im:+.3f}i" if abs(im) > 5e-4 else f"{re:+.3f}"


def main():
    res = load()
    if not res:
        sys.exit("no linear results")
    ocean = ocean_cells()
    L = ["TEST 2: Jacobian of the one-day step F(z) = FE(z + forcing) around free-rollout states",
         "J v by central finite differences (F(z + e v) - F(z - e v)) / 2e in fp32 (exact SDPA",
         "  attention, no TF32), e = eps_rel |z| / |v|; eigenvalues by ARPACK (scipy eigs, LM =",
         "  largest magnitude, SR = smallest real part). resid = |J v - lam v| / |v| with J at",
         "  the given eps; rayleigh = v^H J v / v^H v; bf16 = J with the rollout's bf16 autocast step.",
         "Decoded response: (D(z + a v) - D(z - a v)) / 2a of the real part of v (|v| = 1,",
         "  a = alpha |z|), in normalised units; shares = fraction of the summed squared",
         "  normalised response over all TEMP/SALT/SURFACE channels.", ""]
    L.append(f"Land cells (HEALPix level-5 cells without a mesh node): {int((~ocean).sum())} of {ocean.size} "
             f"({(~ocean).mean():.3f})")
    L.append("")
    for (tag, day), (r, npz_path) in sorted(res.items()):
        cell_e = dict(np.load(npz_path)) if npz_path.exists() else {}
        L.append(f"===== {tag}, day {day} =====")
        tj = r.get("trajectory")
        if tj:
            L.append("  rollout latent, last 5 days: |z_n - z_(n-1)|/|z_n| "
                     + ", ".join(f"{x:.3f}" for x in tj["d1_rel"][-5:])
                     + "; |second difference|/|z_n| " + ", ".join(f"{x:.3f}" for x in tj["d2_rel"][-5:])
                     + "; cos(consecutive increments) "
                     + ", ".join(f"{x:+.3f}" for x in tj["cos_consecutive_increments"][-5:]))
        L.append(f"  checks: bf16 step map vs rollout z_(N+1): rel diff {r['check_bf16_step_vs_rollout_rel']:.2e};"
                 f" fp32 vs bf16 step: {r['check_fp32_vs_bf16_step_rel']:.2e}; |F(z)-z|/|z| "
                 f"{r['step_rel_change']:.3f}; |z| {r['z_norm']:.1f}; fp32 step {r['fp32_step_seconds']:.2f} s")
        for which in ("LM", "SR"):
            e = r.get(f"eigs_{which}")
            if e is None:
                continue
            vals = [fmt_l(a, b) for a, b in zip(e["real"], e["imag"])]
            L.append(f"  {which}: converged={e['converged']}, {e['n_matvec']} matvecs, {e['seconds']:.0f} s")
            L.append("    lambda: " + ", ".join(vals))
            L.append("    |lambda|: " + ", ".join(f"{a:.3f}" for a in e["abs"]))
        for v in r.get("eigvecs", []):
            lam = fmt_l(*v["lambda"])
            res_keys = sorted(k for k in v if k.startswith("resid_eps"))
            ray_keys = sorted(k for k in v if k.startswith("rayleigh"))
            L.append(f"  eigvec {v['which']}{v['index']}  lambda {lam}")
            L.append("    resid: " + ", ".join(f"{k[6:]} {v[k]:.3g}" for k in res_keys))
            L.append("    rayleigh: " + ", ".join(f"{k[9:]} {fmt_l(*v[k])}" for k in ray_keys))
            lb = v["latent_lat_bands"]
            L.append("    latent |v|^2 by lat band: " + ", ".join(f"{k} {x:.2f}" for k, x in lb.items())
                     + f"; top 1% cells hold {v['latent_top1pct_share']:.2f}")
            ce = cell_e.get(f"cellenergy_{v['which']}{v['index']}")
            if ce is not None:
                top = np.argsort(-ce)[:5]
                L.append(f"    share of |v|^2 in land cells {ce[~ocean].sum():.3f}; top cells (lat, lon, land?): "
                         + ", ".join(f"({cell_e['cell_lat'][i]:.0f}, {cell_e['cell_lon'][i]:.0f}, "
                                     f"{'L' if not ocean[i] else 'O'})" for i in top))
            dec = v["decoded"]
            akeys = [k for k in dec if k.startswith("alpha")]
            for ak in akeys:
                sh = group_shares(dec[ak]["per_channel"])
                tot = np.sqrt(sum(x["rms_norm"] ** 2 for x in dec[ak]["per_channel"].values()))
                L.append(f"    decoded {ak} rms_tot {tot:.3g}; shares: "
                         + ", ".join(f"{k} {x:.2f}" for k, x in sh.items()))
            if "cos_with_rollout_second_difference" in v:
                L.append(f"    |cos| with the rollout's second difference at day N: "
                         f"{v['cos_with_rollout_second_difference']:.4f}")
            top = sorted(dec[akeys[-1]]["per_channel"].items(), key=lambda kv: -kv[1]["rms_norm"])[:8]
            L.append("    largest normalised responses: " + ", ".join(
                f"{c} {x['rms_norm']:.3g}" for c, x in top))
            for k in dec:
                if k.startswith("linearity"):
                    L.append(f"    {k}: " + ", ".join(
                        f"{c} corr {x['corr']:.3f} ratio {x['rms_ratio']:.3f}" for c, x in dec[k].items()))
            reg = dec.get("regions_alpha_last", {})
            for c in ("salt_75", "salt_125", "temp_75"):
                if c in reg:
                    bands = reg[c]["lat_bands"]
                    L.append(f"    {c} response energy by lat band: " + ", ".join(
                        f"{k} {x:.2f}" for k, x in bands.items()))
                    L.append(f"    {c} by basin: " + ", ".join(
                        f"{k} {x:.2f}" for k, x in reg[c]["basins"].items()))
        L.append("")
    L += gain_section() + baseline_section()
    txt = "\n".join(L) + "\n"
    (OUT / "test2_jacobian.txt").write_text(txt)
    print(txt)
    figures(res)


def gain_section():
    files = sorted(DATA.glob("gain_*_d*.json"))
    if not files:
        return []
    L = ["===== Gain of the step vs perturbation amplitude (instab_probe.py --mode gain) =====",
         "q = <v, J_eps v> and |J_eps v| for unit v, J_eps v = (F(z + e v) - F(z - e v)) / 2e,",
         "e = eps |z|; v = the rollout's own second difference z_(N+1) - 2 z_N + z_(N-1) or",
         "increment z_(N+1) - z_N at day N, or a random Gaussian direction. fp64 = forcing module and",
         "forecast engine in double precision. additivity = |J(a+b) - J(a) - J(b)| / |J(a) + J(b)|",
         "for random a, b (0 for a linear operator). forcing_injection_rel = |z + g emb(f) - z| / |z|.",
         ""]
    for f in files:
        r = json.loads(f.read_text())
        L.append(f"{r['tag']} day {r['day']}: |z| {r['z_norm']:.0f}, forcing_injection_rel "
                 f"{r['forcing_injection_rel']:.4f}, fp64 vs fp32 step "
                 f"{r.get('fp64_vs_fp32_step_rel', float('nan')):.2e}")
        for prec in ("fp32", "bf16", "fp64"):
            if prec not in r:
                continue
            for name, rows in r[prec].items():
                L.append(f"  {prec} {name:<18} q: " + "  ".join(
                    f"{e}:{x['q']:+.3f}" for e, x in rows.items()))
                L.append(f"  {prec} {name:<18} |Jv|: " + "  ".join(
                    f"{e}:{x['gain']:.3f}" for e, x in rows.items()))
        L.append("  additivity: " + ", ".join(f"{e}: {x:.3f}" for e, x in r["additivity"].items()))
        L.append("")
    return L


def baseline_section():
    files = sorted(DATA.glob("baseline_*_d*.json"))
    if not files:
        return []
    L = ["===== Decoded response of reference latent directions (instab_probe.py --mode baseline) =====",
         "Same definitions as for the eigenvectors: (D(z + a v) - D(z - a v)) / 2a, |v| = 1,",
         "a = alpha |z|; rms_tot = sqrt(sum over all TEMP/SALT/SURFACE channels of rms_norm^2).", ""]
    for f in files:
        r = json.loads(f.read_text())
        L.append(f"{r['tag']} day {r['day']}")
        for name, d in r["directions"].items():
            dec = d["decoded"]
            akeys = [k for k in dec if k.startswith("alpha")]
            tot = {k: np.sqrt(sum(x["rms_norm"] ** 2 for x in dec[k]["per_channel"].values()))
                   for k in akeys}
            sh = group_shares(dec[akeys[-1]]["per_channel"])
            L.append(f"  {name:<18} |v|/|z| {d['rel_norm']:.3f}; top1% cells {d['latent_top1pct_share']:.3f}; "
                     + ", ".join(f"rms_tot {k} {v:.3g}" for k, v in tot.items()))
            L.append("    shares: " + ", ".join(f"{k} {x:.2f}" for k, x in sh.items()))
            reg = dec.get("regions_alpha_last", {})
            if "salt_75" in reg:
                L.append("    salt_75 by lat band: " + ", ".join(
                    f"{k} {x:.2f}" for k, x in reg["salt_75"]["lat_bands"].items()))
        L.append("")
    return L


def figures(res):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 200)
    markers = {10: "o", 40: "s", 70: "^"}
    for ax, m in zip(axs, MODELS):
        ax.plot(np.cos(th), np.sin(th), color="gray", lw=0.8)
        for (tag, day), (r, _) in sorted(res.items()):
            if not tag.startswith(m):
                continue
            eps = r["eps_rel"]
            for which, col in (("LM", "tab:blue"), ("SR", "tab:red")):
                e = r.get(f"eigs_{which}")
                if e:
                    ax.scatter(e["real"], e["imag"], marker=markers.get(day, "x"),
                               facecolors=col if eps >= 1e-2 else "none", edgecolors=col,
                               label=f"day {day} {which} eps {eps:g}")
        ax.axvline(-1, color="gray", lw=0.5, ls=":")
        ax.axvline(1, color="gray", lw=0.5, ls=":")
        ax.set_aspect("equal")
        ax.set_title(f"{m}: ARPACK eigenvalues of the FD Jacobian (open: eps 1e-3, filled: 1e-2)")
        ax.set_xlabel("Re lambda")
        ax.set_ylabel("Im lambda")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(FIG / "test2_eigenvalues.png", dpi=110)
    plt.close(fig)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper_eval.meshtools import load_mesh

    mesh = load_mesh("/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc")
    for (tag, day), (r, npz_path) in sorted(res.items()):
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        vecs = r.get("eigvecs", [])
        if not vecs:
            continue
        chans = ("salt_75", "salt_125", "temp_75", "ssh")
        fig, axs = plt.subplots(len(vecs), len(chans) + 1, figsize=(4 * (len(chans) + 1), 2.4 * len(vecs)),
                                constrained_layout=True, squeeze=False)
        for i, v in enumerate(vecs):
            key = f"{v['which']}{v['index']}"
            lam = fmt_l(*v["lambda"])
            ce = z[f"cellenergy_{key}"]
            ax = axs[i, 0]
            ax.scatter(z["cell_lon"], z["cell_lat"], c=ce, s=1.5, cmap="viridis",
                       vmax=np.percentile(ce, 99.5), rasterized=True)
            ax.set_title(f"{key} lam={lam}: latent |v|^2", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for j, c in enumerate(chans):
                f = z.get(f"resp_{key}_{c}")
                ax = axs[i, j + 1]
                if f is None:
                    ax.axis("off")
                    continue
                ok = np.isfinite(f)
                lim = np.nanpercentile(np.abs(f), 99.5)
                ax.scatter(mesh.lon[ok], mesh.lat[ok], c=f[ok], s=0.3, cmap="RdBu_r", vmin=-lim,
                           vmax=lim, rasterized=True)
                ax.set_title(f"{key}: decoded {c}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{tag} day {day}: leading Jacobian eigenvectors (real part)")
        fig.savefig(FIG / f"test2_eigvecs_{tag}_d{day:03d}.png", dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    main()
