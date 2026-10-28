"""
Tables and figures of the forcing-injection decomposition (scripts/instab/forcing_decomp.py).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
        scripts/instab/forcing_decomp_report.py [TAG ...]

Reads DATA/forcing_decomp_TAG.{json,npz} (default tags fvars003_2209-01-01,
frcwin03_2209-01-01, those present) -> DATA/../decomp_tables.md and
paper_eval/figures/fable_diag/d1_decomp_*.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path("/home/a/a270225/claude_api_tools/fable_diag/D1/data")
FIG = Path("/work/ab0995/a270225/WeatherGenerator2/paper_eval/figures/fable_diag")
TAGS = ("fvars003_2209-01-01", "frcwin03_2209-01-01")
DAYS = (0, 1, 10, 30, 60, 100, 150)


def g(x, nd=3):
    return f"{x:.{nd}g}"


def main():
    tags = sys.argv[1:] or [t for t in TAGS if (DATA / f"forcing_decomp_{t}.json").exists()]
    res = {t: json.loads((DATA / f"forcing_decomp_{t}.json").read_text()) for t in tags}
    L = ["# D1 step 1: decomposition of the additive forcing injection", "",
         "f_n = gate * emb(forcing_n) per latent cell (12288 cells x 2048), forcing indices "
         "1..days as consumed by a rollout from 2209-01-01 (bf16 autocast, as in inference). "
         "C = time mean, a_n = f_n - C. Norms are Frobenius norms over the cells of a class "
         "(all / land / ocean). Latent norms |z| from a free default rollout of the same "
         "model (z_0 = encoded initial condition).", ""]
    L += ["## Summary", "",
          "| quantity | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(res[t]) for t in tags) + " |")

    row("days averaged", lambda r: str(r["days"]))
    row("gate", lambda r: f"{r['gate']:+.4f}")
    row("land cells", lambda r: str(r["num_land"]))
    row("RMS_n |f_n| (all)", lambda r: g(r["rms_f_norm"]["all"]))
    row("|C| (all)", lambda r: g(r["C_norm"]["all"]))
    row("|C| / RMS_n |f_n|: all / land / ocean",
        lambda r: " / ".join(f"{r['C_over_rms_f'][c]:.3f}" for c in ("all", "land", "ocean")))
    row("|C|^2 / mean_n |f_n|^2 (share of injected energy in C)",
        lambda r: " / ".join(f"{r['C2_share_of_mean_f2'][c]:.3f}"
                             for c in ("all", "land", "ocean")))
    row("RMS_n |a_n| / RMS_n |f_n|: all / land / ocean",
        lambda r: " / ".join(f"{r['a_over_f'][c]:.3f}" for c in ("all", "land", "ocean")))
    row("cos(C, f_n): min / mean / max",
        lambda r: " / ".join(f"{r['cos_C_f'][k]:.4f}" for k in ("min", "mean", "max")))
    row("|b2| (last-layer bias) / |W2|_F", lambda r: f"{r['b2_norm']:.3g} / {r['W2_fro']:.3g}")
    row("|gate*b2| per cell", lambda r: g(r["gate_b2_norm_per_cell"]))
    row("|gate*b2 broadcast| / |C|", lambda r: f"{r['gate_b2_broadcast_over_C']:.3f}")
    row("projection share <C, gate*b2> / |C|^2", lambda r: f"{r['gate_b2_proj_share_of_C2']:.3f}")
    row("|C - gate*b2| / |C|", lambda r: f"{r['C_minus_gate_b2_over_C']:.3f}")
    row("cell-uniform part of C: share of |C|^2", lambda r: f"{r['C_cellmean_share_of_C2']:.3f}")
    row("cos(cell mean of C, gate*b2)", lambda r: f"{r['cos_Ccellmean_gate_b2']:.3f}")
    for nm, lab in (("C_cell", "per-cell |C_i|"), ("a_cell_rms", "per-cell RMS_n |a_n,i|")):
        for c in ("land", "ocean"):
            row(f"{lab}, {c}: median (p10-p90)",
                lambda r, nm=nm, c=c: f"{r[nm][c]['median']:.3g} ({r[nm][c]['p10']:.3g}-"
                                      f"{r[nm][c]['p90']:.3g})")
    for c in ("land", "ocean"):
        row(f"per-cell RMS|a| / |C|, {c}: median",
            lambda r, c=c: f"{r['a_over_C_cell'][c]['median']:.3f}")
    L.append("")

    L += ["## Injection relative to the latent along a free default rollout", "",
          "token RMS = mean over cells of the per-token RMS over features; train noise = "
          "per-entry std of the training-mode impute noise, |z|_F * 1e-4.", ""]
    for t in tags:
        lat = res[t]["latent"]
        L += [f"### {t}", "",
              "| day | abs(z) all | abs(z) land / ocean | token RMS (all / ocean) "
              "| train noise std / token RMS | abs(f_n)/abs(z) | abs(C)/abs(z) "
              "| abs(a_n)/abs(z) |",
              "|---|---|---|---|---|---|---|---|"]
        for d in DAYS:
            if d >= len(lat["n"]):
                continue
            i = lat["n"].index(d)
            fz = lat["f_over_z"][i - 1] if d > 0 else np.nan
            cz = lat["C_over_z"][i - 1] if d > 0 else res[t]["C_norm"]["all"] / lat["z_all"][i]
            az = lat["a_over_z"][i - 1] if d > 0 else np.nan
            L.append(f"| {d} | {lat['z_all'][i]:.4g} | {lat['z_land'][i]:.4g} / "
                     f"{lat['z_ocean'][i]:.4g} | {lat['token_rms_all'][i]:.3f} / "
                     f"{lat['token_rms_ocean'][i]:.3f} | "
                     f"{lat['train_noise_per_entry_std'][i] / lat['token_rms_all'][i]:.3f} | "
                     f"{fz:.3f} | {cz:.3f} | {az:.3f} |")
        L.append("")
        r = res[t]
        cf, ca = np.array(r["cum_f"]), np.array(r["cum_a"])
        n = np.arange(1, len(cf) + 1)
        L += [f"Cumulative sums ({t}): |sum f_k| / n at n = 1, 30, 150, {len(cf)}: "
              + ", ".join(f"{cf[k - 1] / k:.4g}" for k in (1, 30, 150, len(cf)))
              + f"; |sum a_k| at n = 30, 150, {len(ca)}: "
              + ", ".join(f"{ca[k - 1]:.4g}" for k in (30, 150, len(ca)))
              + f" (sqrt(n) * RMS|a| = "
              + ", ".join(f"{np.sqrt(k) * r['rms_a_norm']['all']:.4g}"
                          for k in (30, 150, len(ca)))
              + "); log-log slope of |sum a_k| over n = 10..150: "
              + f"{np.polyfit(np.log(n[9:150]), np.log(ca[9:150]), 1)[0]:.2f}.", ""]
    out = DATA.parent / "decomp_tables.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))  # noqa: T201
    figures(tags, res)


def figures(tags, res):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(len(tags), 3, figsize=(18, 4.6 * len(tags)), squeeze=False,
                            constrained_layout=True)
    for i, t in enumerate(tags):
        z = np.load(DATA / f"forcing_decomp_{t}.npz")
        lat, lon = z["cell_lat"], ((z["cell_lon"] + 180) % 360) - 180
        for j, (k, lab) in enumerate((("C_cell", "per-cell |C_i|"),
                                      ("a_cell_rms", "per-cell RMS_n |a_n,i|"))):
            sc = axs[i, j].scatter(lon, lat, c=z[k], s=3, cmap="viridis")
            axs[i, j].set_title(f"{t}: {lab}")
            plt.colorbar(sc, ax=axs[i, j])
        ax = axs[i, 2]
        r = res[t]
        n = np.arange(1, len(r["cum_f"]) + 1)
        ax.loglog(n, r["cum_f"], label="|sum f_k|")
        ax.loglog(n, r["cum_a"], label="|sum a_k|")
        ax.loglog(n, n * r["C_norm"]["all"], "k:", lw=0.8, label="n |C|")
        ax.loglog(n, np.sqrt(n) * r["rms_a_norm"]["all"], "k--", lw=0.8, label="sqrt(n) RMS|a|")
        ax.set_xlabel("n (days)")
        ax.set_title(f"{t}: cumulative injection")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.savefig(FIG / "d1_decomp_maps.png", dpi=100)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for t in tags:
        r = res[t]
        n = np.arange(1, len(r["f_norm"]["all"]) + 1)
        axs[0].plot(n, r["f_norm"]["all"], label=f"{t} |f_n|")
        axs[0].plot(n, r["a_norm"]["all"], ls="--", label=f"{t} |a_n|")
        axs[0].axhline(r["C_norm"]["all"], color="gray", lw=0.6)
        lat = r["latent"]
        axs[1].plot(lat["n"][1:], lat["f_over_z"], label=f"{t} |f_n|/|z_n|")
        axs[1].plot(lat["n"][1:], lat["a_over_z"], ls="--", label=f"{t} |a_n|/|z_n|")
    axs[0].set_title("injection norm per day (gray: |C|)")
    axs[1].set_title("relative to the latent of a free default rollout")
    for ax in axs:
        ax.set_xlabel("day")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(FIG / "d1_decomp_series.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
