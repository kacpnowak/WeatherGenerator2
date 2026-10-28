"""
Inference forcing-scale sweep of fvars003 with forecast.land_cells freeze
(landcells_report.md section 3, step 2): forcing injection multiplied by alpha in
{0.5, 0.75, 1.0, 1.25}, 150 days from 2209-01-01 and 2150-01-01. Scores with the code of
scripts/instab/analyse_multistart.py (onset, pre-onset e-fold, ac1, 2-day high-pass, RMSE at
fixed days), plus the gradients in alpha used for the readout.

    cd /work/ab0995/a270225/WeatherGenerator2
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_forcescale.py \
        compute RUN [RUN ...] [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_forcescale.py report

compute  daily series of a run (analyse_multistart.compute) -> OUT/data/part1_daily_RUN.nc
report   tables -> OUT/tables.md, figures -> paper_eval/figures/forcescale/
alpha = 1.0 from 2209-01-01 is lcfz003 (landcells/data/part1_daily_lcfz003.nc); the default
(no freeze, alpha 1) references are yr1fv003 and ms3a2150 (multistart/data).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_multistart as ams  # noqa: E402

af = ams.af
OUT = Path("/home/a/a270225/claude_api_tools/forcescale")
DATA = OUT / "data"
FIG = af.MAIN / "paper_eval/figures/forcescale"
ALPHAS = (0.5, 0.75, 1.0, 1.25)
STARTS = ("2209-01-01", "2150-01-01")
#: (alpha, start) -> run id
RUN = {}
for _a in ALPHAS:
    for _s in STARTS:
        RUN[(_a, _s)] = f"fs{round(100 * _a):03d}y{_s[2:4]}"
RUN[(1.0, "2209-01-01")] = "lcfz003"
#: default rollouts (no land_cells, alpha 1), for context only
DEFAULT = {"2209-01-01": "yr1fv003", "2150-01-01": "ms3a2150"}
for (_a, _s), _r in RUN.items():
    ams.RUNS.setdefault(_r, ("3", _s, "real"))
PATHS = {"lcfz003": Path("/home/a/a270225/claude_api_tools/landcells/data"),
         "yr1fv003": ams.DATA, "ms3a2150": ams.DATA}
TABLE_DAYS = (1, 15, 30, 60, 90, 150)
CH = ("temp_2", "salt_2", "salt_75", "ssh")
COL = {0.5: "tab:blue", 0.75: "tab:cyan", 1.0: "k", 1.25: "tab:red"}


def path(run):
    return PATHS.get(run, DATA) / f"part1_daily_{run}.nc"


def compute(runs, workers):
    ams.DATA = DATA
    DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        for r in runs:
            ams.compute(r, workers, tmp)


def fmt(x, nd=3):
    return ams.fmt(x, nd)


def onset_or_cap(x, key):
    """Onset day; None (no onset) -> the max lead + 1 for gradients, flagged with '>'."""
    v = x[key]
    return (float(v), False) if v is not None else (float(x["maxlead"] + 1), True)


def monotonic(vals):
    d = np.diff(vals)
    return bool(np.all(d > 0) or np.all(d < 0))


def readout(ms, dss, lines):
    """Gradients in alpha per start date and the spec's decision rule."""
    lines += ["## Gradients in alpha (readout)", "",
              "Spec (landcells_report.md 3.2): A is supported if onset or e-fold changes by more "
              "than 30 days monotonically in alpha with day-15 RMSE within 10 %; insensitive or "
              "symmetric degradation argues against A. Onset 'none<=150' is counted as 151 "
              "(a lower bound). Slopes are least-squares fits over the four alphas, in days "
              "per unit alpha; 'range' = max - min over alpha.", "",
              "| start | quantity | a=0.5 | a=0.75 | a=1.0 | a=1.25 | range | slope (d/alpha) "
              "| monotonic |",
              "|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for s in STARTS:
        have = [a for a in ALPHAS if (a, s) in ms]
        if len(have) < len(ALPHAS):
            lines.append(f"| {s} | (incomplete: {len(have)}/{len(ALPHAS)} runs) "
                         "| | | | | | | |")
            continue
        al = np.array(ALPHAS)
        rows = {}
        for key, name in (("on_s75", "onset salt_75>0.5"), ("on_t2", "onset temp_2>2")):
            v = [onset_or_cap(ms[(a, s)], key) for a in ALPHAS]
            rows[name] = (np.array([x for x, _ in v]), [c for _, c in v])
        rows["e-fold (20-0.3 crossing)"] = (np.array([ms[(a, s)]["efold"] for a in ALPHAS]),
                                            [False] * 4)
        rows["e-fold 30-80"] = (np.array([ms[(a, s)]["efold3080"] for a in ALPHAS]), [False] * 4)
        for name, (v, capped) in rows.items():
            cells = [(">" if c else "") + f"{x:.1f}" for x, c in zip(v, capped, strict=True)]
            ok = np.isfinite(v)
            rng = float(np.nanmax(v) - np.nanmin(v)) if ok.all() else np.nan
            slope = float(np.polyfit(al[ok], v[ok], 1)[0]) if ok.sum() >= 2 else np.nan
            lines.append(f"| {s} | {name} | " + " | ".join(cells)
                         + f" | {rng:.1f} | {slope:+.1f} | {monotonic(v) if ok.all() else '-'} |")
        # day-15 RMSE relative to alpha = 1
        d15 = {}
        for c in CH:
            ref = float(dss[(1.0, s)].rmse.sel(channel=c, lead=15))
            rel = [float(dss[(a, s)].rmse.sel(channel=c, lead=15)) / ref for a in ALPHAS]
            d15[c] = rel
            lines.append(f"| {s} | day-15 RMSE {c} / alpha=1 | "
                         + " | ".join(f"{r:.3f}" for r in rel)
                         + f" | {max(rel) - min(rel):.3f} | - | {monotonic(np.array(rel))} |")
        on = rows["onset salt_75>0.5"][0]
        ef = rows["e-fold 30-80"][0]
        big = [n for n in ("onset salt_75>0.5", "e-fold (20-0.3 crossing)", "e-fold 30-80")
               if monotonic(rows[n][0]) and np.ptp(rows[n][0]) > 30]
        d15_ok = all(abs(r - 1) <= 0.10 for c in ("salt_75", "temp_2") for r in d15[c])
        verdict[s] = (big, d15_ok, on, ef)
    lines.append("")
    for s, (big, d15_ok, on, _ef) in verdict.items():
        sign = "smaller alpha -> later onset" if on[0] > on[-1] else (
            "smaller alpha -> earlier onset" if on[0] < on[-1] else "no change")
        lines.append(f"- {s}: monotonic changes > 30 d: {big or 'none'}; day-15 salt_75/temp_2 "
                     f"RMSE within 10 % of alpha=1 for all alpha: {d15_ok}; onset trend: {sign}.")
    lines.append("")


def report():
    FIG.mkdir(parents=True, exist_ok=True)
    dss, ms = {}, {}
    for key, r in RUN.items():
        if path(r).exists():
            dss[key] = xr.open_dataset(path(r))
            ms[key] = ams.metrics(dss[key])
    ref = {s: xr.open_dataset(path(r)) for s, r in DEFAULT.items() if path(r).exists()}
    refm = {s: ams.metrics(d) for s, d in ref.items()}
    lines = ["# Forcing-scale sweep, fvars003, land_cells freeze, scores", "",
             "Rows: forcing injection scaled by alpha (forecast.forcing_scale), land cells "
             "frozen; last row per start: default rollout (no freeze, alpha 1) for context.", ""]
    lines += ["## Onset, pre-onset growth, alternation", "",
              "| start | alpha | run | max lead | onset salt_75>0.5 | salt_75>1.0 "
              "| onset temp_2>2 | temp_450>1 | salt_75 RMSE d(on-5..on-1) / on "
              "| e-fold (window, R2) | e-fold 30-80 | ac1 salt_75 err (window) | ac1 hc err "
              "| hp hc (ZJ) | hp salt_75 (psu) | day-1 vmT err (K) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(s, a, r, x):
        pre = (" ".join(f"{v:.3f}" for d, v in x["pre"].items() if d < x["on_s75"])
               + f" / {x['pre'][x['on_s75']]:.3f}") if x["on_s75"] else "-"
        ef = (f"{x['efold']:.1f} d (20-{x['ef_win'][1]}"
              f"{'' if x['d03'] else ', no 0.3 crossing'}, {x['ef_r2']:.2f})")
        return (f"| {s} | {a} | {r} | {x['maxlead']} | {fmt(x['on_s75'])} | {fmt(x['on_s75_1'])} "
                f"| {fmt(x['on_t2'])} | {fmt(x['on_t450'])} | {pre} | {ef} "
                f"| {x['efold3080']:.1f} | {x['ac1_s75']:.2f} ({x['pre_win'][0]}-"
                f"{x['pre_win'][1]}) | {x['ac1_hc']:.2f} | {x['hp_hc']:.3g} "
                f"| {x['hp_s75']:.3g} | {x['vmT1']:+.4f} |")

    for s in STARTS:
        for a in ALPHAS:
            if (a, s) in ms:
                lines.append(row(s, a, RUN[(a, s)], ms[(a, s)]))
        if s in refm:
            lines.append(row(s, "1 (default, no freeze)", DEFAULT[s], refm[s]))
    lines.append("")
    for c in CH:
        sc, unit = (1e3, "mm") if c == "ssh" else (1.0, "K" if c.startswith("temp") else "psu")
        lines += [f"## RMSE {c} ({unit})", "",
                  "| start | alpha | " + " | ".join(f"d{d}" for d in TABLE_DAYS) + " |",
                  "|---|---|" + "---|" * len(TABLE_DAYS)]
        for s in STARTS:
            items = [(a, dss[(a, s)]) for a in ALPHAS if (a, s) in dss]
            if s in ref:
                items.append(("1 (default)", ref[s]))
            for a, ds in items:
                cells = [f"{float(ds.rmse.sel(channel=c, lead=d)) * sc:.3g}"
                         if d <= int(ds.lead.max()) else "-" for d in TABLE_DAYS]
                lines.append(f"| {s} | {a} | " + " | ".join(cells) + " |")
        lines.append("")
    for title, key, f in (
        ("ac1 of daily increments of the salt_75 global-mean error, per window", "ac1_win",
         "{:.2f}"),
        ("2-day high-pass RMS of the heat-content error (ZJ), per window", "hp_win", "{:.3g}"),
    ):
        lines += [f"## {title}", "",
                  "| start | alpha | " + " | ".join(f"{a}-{b}" for a, b in ams.WINDOWS) + " |",
                  "|---|---|" + "---|" * len(ams.WINDOWS)]
        for s in STARTS:
            for a in ALPHAS:
                if (a, s) in ms:
                    lines.append(f"| {s} | {a} | "
                                 + " | ".join(f.format(v) for v in ms[(a, s)][key]) + " |")
        lines.append("")
    readout(ms, dss, lines)
    figures(dss, ms, ref)
    txt = "\n".join(lines) + "\n"
    (OUT / "tables.md").write_text(txt)
    print(txt)  # noqa: T201


def figures(dss, ms, ref):
    thr = {"salt_75": 0.5, "temp_2": 2.0}
    fig, axs = plt.subplots(len(STARTS), 5, figsize=(23, 4.3 * len(STARTS)),
                            constrained_layout=True)
    for i, s in enumerate(STARTS):
        for j, c in enumerate(CH):
            ax = axs[i, j]
            sc = 1e3 if c == "ssh" else 1.0
            for a in ALPHAS:
                if (a, s) in dss:
                    ds = dss[(a, s)]
                    ax.semilogy(ds.lead, ds.rmse.sel(channel=c) * sc, color=COL[a], lw=1.2,
                                label=f"alpha {a} ({RUN[(a, s)]})")
            if s in ref:
                ax.semilogy(ref[s].lead, ref[s].rmse.sel(channel=c) * sc, color="gray", lw=0.9,
                            ls="--", label=f"default, no freeze ({DEFAULT[s]})")
            if c in thr:
                ax.axhline(thr[c], color="gray", lw=0.8, ls=":")
            ax.set_title(f"{s}: RMSE {c}" + (" (mm)" if c == "ssh" else ""))
            ax.set_xlabel("lead (days)")
            ax.grid(alpha=0.3)
        axs[i, 0].legend(fontsize=7)
        ax = axs[i, 4]
        sc, unit = af.scale_unit("hc_0_2150")
        for a in ALPHAS:
            if (a, s) in dss:
                ds = dss[(a, s)]
                ax.plot(ds.lead, af.err(ds, "hc_0_2150") * sc, color=COL[a], lw=1)
        ax.set_title(f"{s}: global-mean error hc_0_2150 ({unit})")
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    fig.suptitle("fvars003, land_cells freeze: forcing injection scaled by alpha")
    fig.savefig(FIG / "forcescale_rmse.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    mk = {"2209-01-01": "o-", "2150-01-01": "s--"}
    for s in STARTS:
        have = [a for a in ALPHAS if (a, s) in ms]
        if not have:
            continue
        on = [onset_or_cap(ms[(a, s)], "on_s75")[0] for a in have]
        ont = [onset_or_cap(ms[(a, s)], "on_t2")[0] for a in have]
        axs[0].plot(have, on, mk[s], label=s)
        axs[1].plot(have, ont, mk[s], label=s)
        axs[2].plot(have, [ms[(a, s)]["efold3080"] for a in have], mk[s], label=s)
        axs[3].plot(have, [float(dss[(a, s)].rmse.sel(channel="salt_75", lead=15)) for a in have],
                    mk[s], label=f"{s} d15")
        axs[3].plot(have, [float(dss[(a, s)].rmse.sel(channel="salt_75", lead=60)) for a in have],
                    mk[s], alpha=0.5, label=f"{s} d60")
    titles = ("onset salt_75 > 0.5 (d; 151 = none)", "onset temp_2 > 2 (d)",
              "e-fold salt_75 RMSE, days 30-80 (d)", "salt_75 RMSE (psu)")
    for ax, t in zip(axs, titles, strict=True):
        ax.set_title(t)
        ax.set_xlabel("alpha (forcing scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("fvars003, land_cells freeze: scores vs forcing scale")
    fig.savefig(FIG / "forcescale_vs_alpha.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("runs", nargs="*")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    if a.cmd == "compute":
        compute(a.runs, a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
