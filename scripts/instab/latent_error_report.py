#!/usr/bin/env python
"""
Tables and figures of the latent-error decomposition (latent_error.py output).

    python latent_error_report.py [--data DIR] [--fig DIR] [--out tables.md]

Reads DATA/latenterr_<model>_<start>.json, writes markdown tables (per run: days
1/10/20/40/60/80/100/150; noise floor; gains; decision readout) and figures.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DAYS = (1, 10, 20, 40, 60, 80, 100, 150)
CLS = ("land", "adj", "other")
MODE_NAME = {"none": "default", "freeze": "freeze"}


def load(data):
    runs = {}
    for f in sorted(Path(data).glob("latenterr_*.json")):
        d = json.loads(f.read_text())
        model, start = d["tag"].split("_", 1)
        runs[(model, start)] = d
    return runs


def at(T, key, day):
    n = T["n"]
    return T[key][n.index(day)] if day in n else float("nan")


def ri(a, i):
    return a[i] if i < len(a) else float("nan")


def f3(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def gain_row(g, name, prec="fp32", eps="1e-02"):
    r = g[prec][name][eps]
    return r


def tables(runs):
    L = []
    any_d = next(iter(runs.values()))
    cc = any_d["class_counts"]
    share = {c: cc[c] / cc["all"] for c in CLS}
    L.append("# Latent-error decomposition: tables\n")
    L.append(f"Cell classes (HEALPix level 5, equal-area cells, so cell share = area share): "
             f"land {cc['land']} ({share['land']:.3f}), land-adjacent ocean {cc['adj']} "
             f"({share['adj']:.3f}), other ocean {cc['other']} ({share['other']:.3f}).\n")

    L.append("## Checks\n")
    L.append("| model | start | |z_true_0 - z_0|/|z_0| | own step vs Model.forward, steps 1-5 (max rel), "
             "default / freeze | gain-day step check (max rel) | encode s |")
    L.append("|---|---|---|---|---|---|")
    for (m, s), d in sorted(runs.items()):
        chk = " / ".join(f3(max(d["runs"][mo]["step_check_rel"])) for mo in d["runs"])
        gchk = max((g["step_check_rel"] for mo in d["runs"]
                    for g in d["runs"][mo]["gains"].values()), default=float("nan"))
        L.append(f"| {m} | {s} | {f3(d['z_true0_vs_z0_rel'])} | {chk} | {f3(gchk)} | "
                 f"{d['encode_seconds']:.0f} |")
    L.append("")

    L.append("## Noise floor: one-day step from the encoded truth\n")
    L.append("nf = |z_true_(n) - S_n(z_true_(n-1))| / |z_true_n| (default step); nf_fz = same "
             "with the land tokens of z_true_(n-1) kept (freeze step), ocean cells; dtrue = "
             "|z_true_n - z_true_(n-1)| / |z_true_n| (day-to-day change of the encoded truth); "
             "Dtrue = |z_true_n - z_true_0| / |z_true_n|; nf2 = |S(S(z_true_(n-2))) - S(z_true_(n-1))| "
             "/ |z_true_n| (lead-2 minus lead-1 latent, all cells). Each relative to the class "
             "norm of z_true_n.\n")
    L.append("| model | start | day | nf all | nf ocean | nf land | nf adj | nf_fz ocean | "
             "dtrue ocean | Dtrue ocean | nf land share | nf adj share | nf2 all |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, s), d in sorted(runs.items()):
        NF = d["noise_floor"]
        for day in DAYS:
            if day not in NF["n"]:
                continue
            L.append(f"| {m} | {s} | {day} | " + " | ".join(f3(at(NF, k, day)) for k in (
                "all.nf_rel", "ocean.nf_rel", "land.nf_rel", "adj.nf_rel", "ocean.nf_fz_rel",
                "ocean.dtrue_rel", "ocean.Dtrue_rel", "land.nf_share", "adj.nf_share",
                "all.nf2_rel")) + " |")
        NF_mean = {k: np.nanmean(NF[k]) for k in ("all.nf_rel", "ocean.nf_rel")}
        L.append(f"| {m} | {s} | mean 1-{NF['n'][-1]} | {f3(NF_mean['all.nf_rel'])} | "
                 f"{f3(NF_mean['ocean.nf_rel'])} | | | | | | | | |")
    L.append("")

    L.append("## Latent error along the rollouts\n")
    L.append("e = z_n - z_true_n. e_rel = |e|/|z_true_n| (all cells; ocean cells). Shares: "
             "fraction of |e|^2 on the class, and in brackets the ratio to the cell share. "
             "cos(e,X): X = R (cumulative residual sum(dz - f)), F (cumulative forcing), dz "
             "(current increment), D (z_n - z_0); the fraction of |e|^2 on X is cos^2. "
             "span: fraction of |e|^2 in span{R, F} (= span{D, F}). ocean: the same restricted "
             "to ocean cells.\n")
    for (m, s), d in sorted(runs.items()):
        for mo, r in d["runs"].items():
            T = r["trajectory"]
            for pre, ek, title in (("", "e", "encoded-truth reference e = z_n - z_true_n"),
                                   ("g_", "g", "lead-1 reference g = z_n - S_n(z_true_(n-1))")):
                L.append(f"### {m} {MODE_NAME[mo]} from {s}: {title}\n")
                L.append("| day | err_rel | err_rel ocean | land share | adj share | other share | "
                         "cos(err,R) | cos(err,F) | cos(err,dz) | cos(err,D) | span RF | "
                         "ocean cos(err,R) | ocean cos(err,F) | ocean cos(err,dz) | ocean span RF | "
                         "adj cos(err,R) | cos(e,g) | |R|/|z_t| | |F|/|z_t| | |D|/|z_t| |")
                L.append("|" + "---|" * 20)
                for day in DAYS:
                    if day not in T["n"]:
                        continue
                    sh = []
                    for c in CLS:
                        v = at(T, f"{c}.{pre}ee_share", day)
                        sh.append(f"{f3(v)} ({v / share[c]:.2f})")
                    vals = [at(T, k, day) for k in (
                        f"all.{pre}cos_eR", f"all.{pre}cos_eF", f"all.{pre}cos_eI",
                        f"all.{pre}cos_eD", f"all.{pre}frac_span_RF", f"ocean.{pre}cos_eR",
                        f"ocean.{pre}cos_eF", f"ocean.{pre}cos_eI", f"ocean.{pre}frac_span_RF",
                        f"adj.{pre}cos_eR", "all.cos_eg", "all.R_rel", "all.F_rel", "all.D_rel")]
                    L.append(f"| {day} | {f3(at(T, f'all.{ek}_rel', day))} | "
                             f"{f3(at(T, f'ocean.{ek}_rel', day))} | " + " | ".join(sh) + " | "
                             + " | ".join(f3(v) for v in vals) + " |")
                L.append("")

    L.append("## Gain probes (one-day step at day N, fp32, eps = 1e-2 |z|)\n")
    L.append("q = <v, Jv>, g = |Jv| for unit v; q_oc = growth of the ocean part of v. "
             "Directions: next_inc = z_(N+1) - z_N, inc = dz_N, e = e_N, e_oc = e_N on ocean "
             "cells, g = g_N (lead-1 reference), g_oc, R = R_N, F = F_N, rnd = random. Ranges over eps 1e-3..1e-1 in the json.\n")
    names = ("next_increment", "increment", "e", "e_ocean", "g", "g_ocean", "R", "F", "random0")
    short = ("next_inc", "inc", "e", "e_oc", "g", "g_oc", "R", "F", "rnd")
    L.append("| model | start | mode | N | " + " | ".join(f"{n} q / g" for n in short)
             + " | cos(e,F) | cos(g,F) | cos(g,R) | cos(e,next_inc) | bf16 e g (eps 1e-2) |")
    L.append("|---|---|---|---|" + "---|" * (len(names) + 5))
    for (m, s), d in sorted(runs.items()):
        for mo, r in d["runs"].items():
            for N, g in sorted(r["gains"].items(), key=lambda x: int(x[0])):
                cells = []
                for n in names:
                    x = gain_row(g, n)
                    cells.append(f"{x['q']:.2f} / {x['gain']:.2f}")
                bf = g["bf16"]["e"]["1e-02"]["gain"]
                L.append(f"| {m} | {s} | {MODE_NAME[mo]} | {N} | " + " | ".join(cells)
                         + f" | {g['cosines']['cos_e_F']:.3f} | {g['cosines']['cos_g_F']:.3f} | "
                         f"{g['cosines']['cos_g_R']:.3f} | "
                         f"{g['cosines']['cos_e_next_increment']:.3f} | {bf:.2f} |")
    L.append("")
    L.append(decision(runs, share))
    return "\n".join(L)


def decision(runs, share):
    L = ["## Decision readout\n",
         "Computed for both references (e: encoded truth, as specified; g: lead-1 reference). "
         "A: cos(err,R) >= 0.5 at days 20/40/60 and growing (20 < 40 < 60). "
         "C: land-adjacent share of |err|^2 / cell share > 2, mean over days 1-59. "
         "B: gain |Jv| > 1 along the next increment, e_N or g_N (fp32, eps 1e-2) at day 40 or "
         "80, with |cos(err_N, F_N)| < 0.5 (no forcing alignment); 'marginal' when the largest "
         "such gain is in (1, 1.05]; q = <v, Jv> in brackets.\n",
         "| model | mode | start | ref | cos(err,R) d20 / d40 / d60 | A | ocean cos(err,R) "
         "d20/40/60 | cos(err,F) d20/40/60 | adj share / cell share (mean d1-59; d10, d30, d59) "
         "| C | gain next_inc g (q) d40 / d80 | gain err g (q) d40 / d80 | |cos(err,F)| d40/d80 "
         "| B |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (m, s), d in sorted(runs.items()):
        for mo, r in d["runs"].items():
            T = r["trajectory"]
            for pre, ek in (("", "e"), ("g_", "g")):
                cr = [at(T, f"all.{pre}cos_eR", k) for k in (20, 40, 60)]
                co = [at(T, f"ocean.{pre}cos_eR", k) for k in (20, 40, 60)]
                cf = [at(T, f"all.{pre}cos_eF", k) for k in (20, 40, 60)]
                A = all(c >= 0.5 for c in cr) and cr[0] < cr[1] < cr[2]
                sh = np.array(T[f"adj.{pre}ee_share"][:59], float)
                ratio = sh / share["adj"]
                rmean = np.nanmean(ratio[1:] if ek == "g" else ratio)  # g_1 = 0
                C = rmean > 2
                g = r["gains"]
                gi, ge, cfs, gmax = [], [], [], 0.0
                for N in ("40", "80"):
                    if N not in g:
                        gi.append("-"), ge.append("-"), cfs.append("-")
                        continue
                    a = g[N]["fp32"]["next_increment"]["1e-02"]
                    b = g[N]["fp32"][ek]["1e-02"]
                    gi.append(f"{a['gain']:.2f} ({a['q']:.2f})")
                    ge.append(f"{b['gain']:.2f} ({b['q']:.2f})")
                    c = abs(g[N]["cosines"][f"cos_{ek}_F"])
                    cfs.append(f"{c:.2f}")
                    if c < 0.5:
                        gmax = max(gmax, a["gain"], b["gain"])
                B = "yes" if gmax > 1.05 else ("marginal" if gmax > 1 else "no")
                rows.append(dict(model=m, mode=MODE_NAME[mo], start=s, ref=ek, A=A, C=C, B=B))
                L.append(
                    f"| {m} | {MODE_NAME[mo]} | {s} | {ek} | "
                    f"{' / '.join(f'{x:.2f}' for x in cr)} | {'yes' if A else 'no'} | "
                    f"{' / '.join(f'{x:.2f}' for x in co)} | "
                    f"{' / '.join(f'{x:.2f}' for x in cf)} | {rmean:.2f} ({ri(ratio, 9):.2f}, "
                    f"{ri(ratio, 29):.2f}, {ri(ratio, 58):.2f}) | {'yes' if C else 'no'} | "
                    f"{' / '.join(gi)} | {' / '.join(ge)} | {' / '.join(cfs)} | "
                    f"{B} |")
    return "\n".join(L) + "\n"


def figures(runs, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(runs)
    any_d = runs[keys[0]]
    cc = any_d["class_counts"]
    share = {c: cc[c] / cc["all"] for c in CLS}
    colors = {"none": "#c0392b", "freeze": "#2471a3"}
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))

    def grid(title):
        fig, axs = plt.subplots(nrow, ncol, figsize=(11, 3.8 * nrow), sharex=True,
                                constrained_layout=True, squeeze=False)
        fig.suptitle(title, fontsize=9)
        return fig, axs.ravel()

    def finish(fig, axs, name):
        for ax in axs[-ncol:]:
            ax.set_xlabel("day")
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)

    # 1: error growth with the noise floors
    fig, axs = grid("Latent error / |z_true_n|: e (encoded truth, solid), g (lead-1 reference, "
                    "dash-dot); grey: one-step noise floor |z_t - S(z_t)|, dotted: "
                    "|z_t,n - z_t,n-1|")
    for ax, k in zip(axs, keys):
        d = runs[k]
        for mo, r in d["runs"].items():
            T = r["trajectory"]
            ax.plot(T["n"], T["all.e_rel"], color=colors[mo], label=f"{MODE_NAME[mo]} e")
            ax.plot(T["n"], T["all.g_rel"], color=colors[mo], ls="-.",
                    label=f"{MODE_NAME[mo]} g")
        NF = d["noise_floor"]
        ax.plot(NF["n"], NF["all.nf_rel"], color="0.5", lw=1, label="noise floor")
        ax.plot(NF["n"], NF["all.dtrue_rel"], color="0.6", lw=1, ls=":", label="truth change")
        ax.set_title(f"{k[0]} from {k[1]}")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    finish(fig, axs, "err_rel.png")

    # 2: cosines, per reference
    for pre, ek in (("", "e"), ("g_", "g")):
        fig, axs = grid(f"cos({ek}_n, X), all cells: R = cumulative residual (solid), "
                        "F = cumulative forcing (dashed), dz = increment (dotted)")
        for ax, k in zip(axs, keys):
            for mo, r in runs[k]["runs"].items():
                T = r["trajectory"]
                ax.plot(T["n"], T[f"all.{pre}cos_eR"], color=colors[mo],
                        label=f"{MODE_NAME[mo]} R")
                ax.plot(T["n"], T[f"all.{pre}cos_eF"], color=colors[mo], ls="--", label="F")
                ax.plot(T["n"], T[f"all.{pre}cos_eI"], color=colors[mo], ls=":", label="dz")
            ax.axhline(0.5, color="k", lw=0.6)
            ax.axhline(0, color="k", lw=0.4)
            ax.set_title(f"{k[0]} from {k[1]}")
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7, ncol=2)
        finish(fig, axs, f"cosines_{ek}.png")

    # 3: class shares relative to the cell share, per reference
    for pre, ek in (("", "e"), ("g_", "g")):
        fig, axs = grid(f"Share of |{ek}_n|^2 / cell share: land (dotted), land-adjacent ocean "
                        "(solid), other ocean (dashed)")
        for ax, k in zip(axs, keys):
            for mo, r in runs[k]["runs"].items():
                T = r["trajectory"]
                for c, ls in (("land", ":"), ("adj", "-"), ("other", "--")):
                    ax.plot(T["n"], np.array(T[f"{c}.{pre}ee_share"], float) / share[c],
                            color=colors[mo], ls=ls, label=f"{MODE_NAME[mo]} {c}")
            ax.axhline(2, color="k", lw=0.6)
            ax.axhline(1, color="k", lw=0.4)
            ax.axvline(60, color="0.6", lw=0.6)
            ax.set_title(f"{k[0]} from {k[1]}")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7, ncol=2)
        finish(fig, axs, f"class_shares_{ek}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/a/a270225/claude_api_tools/latenterr/data")
    p.add_argument("--fig", default="/work/ab0995/a270225/WeatherGenerator2/paper_eval/figures/"
                   "latenterr")
    p.add_argument("--out", default="/home/a/a270225/claude_api_tools/latenterr/tables.md")
    a = p.parse_args()
    runs = load(a.data)
    Path(a.out).write_text(tables(runs))
    figures(runs, a.fig)
    print(f"{len(runs)} runs -> {a.out}, {a.fig}")


if __name__ == "__main__":
    main()
