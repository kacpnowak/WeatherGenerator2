"""
Additivity of land freeze, latent filter and forcing scale (Fable diagnostic D3): fvars003,
2209-01-01, land_cells freeze combined with second_difference nu=0.25 (d3a09), with
forcing_scale 1.1 (d3b09), and with both (d3c09). Scored with the forcing-scale / multi-start
code (analyse_multistart.metrics: onset, e-fold, ac1, 2-day high-pass) against the
single-mechanism runs: lcfz003 (freeze), ins3sd25 / ins3sd10 / ins3rx25 / ins3rx10 (filter
only), fs125y09 (freeze, alpha 1.25) and yr1fv003 (default).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_d3.py \
        compute RUN [RUN ...] [--workers N] [--max-lead 363]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_d3.py report

compute  daily series (analyse_multistart.compute, with the 2180-2209 climatology RMSE every
         5 days up to day 150) -> OUT/data/part1_daily_RUN.nc, or OUT/data/lead363/ with
         --max-lead 363
report   tables -> OUT/tables.md, figure -> paper_eval/figures/fable_diag/d3_rmse.png
Climatology ratios use the clim RMSE stored with yr1fv003 (same truth, same start date).
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
HOME = Path("/home/a/a270225/claude_api_tools")
OUT = HOME / "fable_diag/D3"
DATA = OUT / "data"
FIG = af.MAIN / "paper_eval/figures/fable_diag"
START = "2209-01-01"
#: run -> (label, directory of its 150-day series, directory of its extended series or None)
RUNS = {
    "yr1fv003": ("default", HOME / "multistart/data", None),
    "lcfz003": ("freeze", HOME / "landcells/data", None),
    "ins3rx10": ("relax 0.1", HOME / "instab_tests/data", None),
    "ins3rx25": ("relax 0.25", HOME / "instab_tests/data", None),
    "ins3sd10": ("sd 0.1", HOME / "instab_tests/data", None),
    "ins3sd25": ("sd 0.25", HOME / "instab_tests/data", HOME / "instab_tests/data"),
    "fs125y09": ("freeze + alpha 1.25", HOME / "forcescale/data",
                 HOME / "forcescale/data/lead363"),
    "d3a09": ("freeze + sd 0.25", DATA, DATA / "lead363"),
    "d3b09": ("freeze + alpha 1.1", DATA, DATA / "lead363"),
    "d3c09": ("freeze + sd 0.25 + alpha 1.1", DATA, DATA / "lead363"),
}
NEW = ("d3a09", "d3b09", "d3c09")
for _r in NEW:
    ams.RUNS.setdefault(_r, ("3", START, "real"))
CH = ("temp_2", "salt_2", "salt_75", "ssh")
DAYS = (1, 2, 15, 30, 60, 90, 120, 150)
EXT_DAYS = (150, 180, 240, 300, 363)


def compute(runs, workers, max_lead):
    ams.DATA = DATA if max_lead == 150 else DATA / f"lead{max_lead}"
    ams.MAX_LEAD = max_lead
    ams.DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ams.DATA) as tmp:
        for r in runs:
            ams.compute(r, workers, tmp)


def load(run, extended=False):
    _, d150, dext = RUNS[run]
    p = (dext if extended else d150)
    if p is None:
        return None
    p = p / f"part1_daily_{run}.nc"
    if not p.exists():
        return None
    ds = xr.open_dataset(p)
    if extended:
        return ds if int(ds.lead.max()) > 150 else None
    return ds.sel(lead=ds.lead <= 150)


def fmt(x, top):
    return ams.fmt(x).replace("none<=150", f"none<={top}") if x is None else ams.fmt(x)


def report():
    FIG.mkdir(parents=True, exist_ok=True)
    clim_ds = load("yr1fv003")
    have = {r: load(r) for r in RUNS if load(r) is not None}
    ext = {r: load(r, True) for r in RUNS if load(r, True) is not None}
    lines = ["# D3: additivity of land freeze, latent filter and forcing scale (fvars003, "
             f"{START})", "",
             "Onsets: first lead with RMSE salt_75 > 0.5 psu / temp_2 > 2 K; 'none<=N' = no "
             "onset within the run's N days. e-fold: salt_75 RMSE fit from day 20 to the 0.3 psu "
             "crossing (or the end) and over days 30-80. ac1 / hp: pre-onset window (16 to "
             "onset-1, or 16-150).", ""]
    lines += ["## Onset and pre-onset growth (150-day series; extended series where present)", "",
              "| run | options | onset salt_75>0.5 | onset temp_2>2 | onset salt_75>0.5 (ext.) "
              "| onset temp_2>2 (ext.) | e-fold 20-0.3 | e-fold 30-80 | ac1 salt_75 err "
              "| ac1 hc err | hp hc (ZJ) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    ms = {}
    for r, ds in have.items():
        x = ams.metrics(ds)
        ms[r] = x
        top = int(ds.lead.max())
        e = ext.get(r)
        if e is not None:
            xe, te = ams.metrics(e), int(e.lead.max())
            oe = [str(xe[k]) if xe[k] is not None else f"none<={te}" for k in ("on_s75", "on_t2")]
        else:
            oe = ["-", "-"]
        lines.append(
            f"| {r} | {RUNS[r][0]} | {fmt(x['on_s75'], top)} | {fmt(x['on_t2'], top)} "
            f"| {oe[0]} | {oe[1]} | {x['efold']:.1f} d (20-{x['ef_win'][1]}) "
            f"| {x['efold3080']:.1f} | {x['ac1_s75']:.2f} | {x['ac1_hc']:.2f} "
            f"| {x['hp_hc']:.3g} |")
    lines.append("")
    for c in CH:
        sc, unit = (1e3, "mm") if c == "ssh" else (1.0, "K" if c.startswith("temp") else "psu")
        lines += [f"## RMSE {c} ({unit})", "",
                  "| run | options | " + " | ".join(f"d{d}" for d in DAYS) + " |",
                  "|---|---|" + "---|" * len(DAYS)]
        for r, ds in have.items():
            top = int(ds.lead.max())
            cells = [f"{float(ds.rmse.sel(channel=c, lead=d)) * sc:.3g}" if d <= top else "-"
                     for d in DAYS]
            lines.append(f"| {r} | {RUNS[r][0]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Ratio to the 2180-2209 day-of-year climatology RMSE (same truth), days 60 and "
              "150", "",
              "| run | options | " + " | ".join(f"{c} d60 | {c} d150" for c in CH) + " |",
              "|---|---|" + "---|" * (2 * len(CH))]
    clim = {(c, d): float(clim_ds[f"clim_rmse_{c}"].sel(lead=d)) for c in CH for d in (60, 150)}
    for r, ds in have.items():
        cells = []
        for c in CH:
            for d in (60, 150):
                cells.append(f"{float(ds.rmse.sel(channel=c, lead=d)) / clim[(c, d)]:.2f}")
        lines.append(f"| {r} | {RUNS[r][0]} | " + " | ".join(cells) + " |")
    sc = {c: (1e3 if c == "ssh" else 1.0) for c in CH}
    lines.append("| (clim RMSE) | | " + " | ".join(
        f"{clim[(c, d)] * sc[c]:.3g}" for c in CH for d in (60, 150)) + " |")
    lines.append("")
    if ext:
        lines += ["## Extended runs (to 363 days)", "",
                  "| run | options | max lead | "
                  + " | ".join(f"salt_75 d{d}" for d in EXT_DAYS) + " | "
                  + " | ".join(f"temp_2 d{d}" for d in EXT_DAYS) + " | salt_75 max (day) |",
                  "|---|---|---|" + "---|" * (2 * len(EXT_DAYS) + 1)]
        for r, ds in ext.items():
            top = int(ds.lead.max())
            cells = [f"{float(ds.rmse.sel(channel=c, lead=d)):.3g}" if d <= top else "-"
                     for c in ("salt_75", "temp_2") for d in EXT_DAYS]
            s75 = ds.rmse.sel(channel="salt_75")
            imax = int(s75.argmax())
            lines.append(f"| {r} | {RUNS[r][0]} | {top} | " + " | ".join(cells)
                         + f" | {float(s75[imax]):.3g} ({int(ds.lead[imax])}) |")
        lines.append("")
    lines += ["## ac1 of daily increments of the salt_75 global-mean error, per window", "",
              "| run | " + " | ".join(f"{a}-{b}" for a, b in ams.WINDOWS) + " |",
              "|---|" + "---|" * len(ams.WINDOWS)]
    for r, x in ms.items():
        lines.append(f"| {r} | " + " | ".join(f"{v:.2f}" for v in x["ac1_win"]) + " |")
    lines += ["", "## 2-day high-pass RMS of the heat-content error (ZJ), per window", "",
              "| run | " + " | ".join(f"{a}-{b}" for a, b in ams.WINDOWS) + " |",
              "|---|" + "---|" * len(ams.WINDOWS)]
    for r, x in ms.items():
        lines.append(f"| {r} | " + " | ".join(f"{v:.3g}" for v in x["hp_win"]) + " |")
    lines.append("")
    txt = "\n".join(lines) + "\n"
    (OUT / "tables.md").write_text(txt)
    print(txt)  # noqa: T201

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    show = [r for r in ("yr1fv003", "lcfz003", "ins3sd25", "fs125y09", *NEW) if r in have]
    for ax, c in zip(axs, ("salt_75", "temp_2", "ssh"), strict=True):
        s = 1e3 if c == "ssh" else 1.0
        for r in show:
            ds = ext.get(r) or have[r]
            new = r in NEW
            ax.plot(ds.lead, ds.rmse.sel(channel=c) * s, lw=1.8 if new else 1,
                    ls="-" if new else "--", label=f"{r} ({RUNS[r][0]})")
        cl = clim_ds[f"clim_rmse_{c}"]
        ax.plot(clim_ds.lead, cl * s, "k.", ms=3, label="clim 2180-2209")
        ax.set_yscale("log")
        ax.set_xlabel("lead (days)")
        ax.set_title(f"RMSE {c}" + (" (mm)" if c == "ssh" else ""))
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "d3_rmse.png", dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("runs", nargs="*")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-lead", type=int, default=150)
    a = p.parse_args()
    if a.cmd == "compute":
        compute(a.runs, a.workers, a.max_lead)
    else:
        report()


if __name__ == "__main__":
    main()
