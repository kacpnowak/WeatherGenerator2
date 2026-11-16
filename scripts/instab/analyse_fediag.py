"""
Scores of the fable diagnostics D1 (forcing_split), D2 (fe_noise, training-noise regime) and
D9 (onset spread from first-step latent noise), fvars003 from 2209-01-01, 150 days. Uses the
code of scripts/instab/analyse_multistart.py (onset, pre-onset e-fold 20 -> 0.3, ac1,
2-day high-pass, RMSE at fixed days, 2180-2209 climatology RMSE vs the same truth), as
analyse_forcescale.py does.

    cd /work/ab0995/a270225/WeatherGenerator2
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_fediag.py \
        compute RUN [RUN ...] [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_fediag.py report

compute  daily series of a run (analyse_multistart.compute) -> OUT/data/part1_daily_RUN.nc
report   tables -> OUT/{D1,D2,D9}/tables.md, figures -> paper_eval/figures/fable_diag/

Year-long runs (YEAR: train_like-noise runs from 2209 / 2150 / 2200, 363 days):
    ... analyse_fediag.py compute nztl09 nztl50 nztl00 --max-lead 363
    ... analyse_fediag.py report YEAR
compute --max-lead N writes OUT/data/leadN/ and stores the climatology RMSE every 5 days up to
N (and at N); report YEAR -> OUT/YEAR_NOISE/tables.md and figure year_noise_rmse.png.
The climatology is the 2180-2209 day-of-year climatology, looked up at each valid date.
References: lcfz003 (freeze, landcells/data), yr1fv003 (default, multistart/data) and the
forcing_scale runs fs050y09 / fs075y09 / fs125y09 (forcescale/data).
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
OUT = Path("/home/a/a270225/claude_api_tools/fable_diag")
DATA = OUT / "data"
FIG = af.MAIN / "paper_eval/figures/fable_diag"
START = "2209-01-01"
TABLE_DAYS = (1, 15, 30, 60, 90, 150)
CLIM_DAYS = (30, 60, 90, 150)
CH = ("temp_2", "salt_2", "salt_75", "ssh")

#: diagnostic -> [(run, label)] in table order; references last
D1 = [
    ("spa050y9", "freeze, split: const 1, anom 0.5"),
    ("spa150y9", "freeze, split: const 1, anom 1.5"),
    ("spc080y9", "freeze, split: const 0.8, anom 1"),
    ("spc125y9", "freeze, split: const 1.25, anom 1"),
    ("lcfz003", "freeze, unscaled (const 1, anom 1) [ref]"),
    ("fs050y09", "freeze, forcing_scale 0.5 (both) [ref]"),
    ("fs075y09", "freeze, forcing_scale 0.75 (both) [ref]"),
    ("fs125y09", "freeze, forcing_scale 1.25 (both) [ref]"),
]
D2 = [
    ("nztld09", "train_like 1e-4 + dropout"),
    ("nztl09", "train_like 1e-4, no dropout"),
    ("nzr01y09", "token_rms 0.01"),
    ("nzr03y09", "token_rms 0.03"),
    ("nzr10y09", "token_rms 0.10"),
    ("yr1fv003", "deterministic default [ref]"),
]
D9 = {
    "default": [("d9def1", "seed 1"), ("d9def2", "seed 2"), ("d9def3", "seed 3"),
                ("yr1fv003", "unperturbed [ref]")],
    "freeze": [("d9frz1", "seed 1"), ("d9frz2", "seed 2"), ("d9frz3", "seed 3"),
               ("lcfz003", "unperturbed [ref]")],
}
REF_DIRS = {"lcfz003": Path("/home/a/a270225/claude_api_tools/landcells/data"),
            "yr1fv003": ams.DATA,
            **{r: Path("/home/a/a270225/claude_api_tools/forcescale/data")
               for r in ("fs050y09", "fs075y09", "fs125y09")}}
#: year-long train_like-noise runs: run -> (start, deterministic reference of that start)
YEAR = {"nztl09": ("2209-01-01", "yr1fv003"),
        "nztl50": ("2150-01-01", "ms3a2150"),
        "nztl00": ("2200-01-01", "ms3a2200")}
YEAR_LEAD = 363
YEAR_DAYS = (1, 15, 30, 60, 90, 150, 240, 363)
YEAR_CLIM_DAYS = (30, 60, 90, 150, 240, 363)
#: 363-day series of the deterministic 2209 run (diag_yr1; no climatology columns) and of the
#: old lineage's year (oldlineage_series.py; rmse only)
YEAR_REF = {"yr1fv003": Path("/home/a/a270225/claude_api_tools/diag_yr1/part1_daily_yr1fv003.nc"),
            "yr1win03": Path("/home/a/a270225/claude_api_tools/oldlineage/data/"
                             "series_yr1win03.nc")}
for _r, (_s, _) in YEAR.items():
    ams.RUNS.setdefault(_r, ("3", _s, "real"))
#: deterministic 2209 rollouts of the no-noise fine-tune fvnn003 (fe_impute_latent_noise_std 0)
NONOISE = [("fvnn0e0", "fvnn003 mini-epoch 0, deterministic"),
           ("fvnn1y09", "fvnn003 mini-epoch 1, deterministic, 363 d"),
           ("fvnnf09", "fvnn003 final checkpoint, deterministic, 363 d"),
           ("fvnnf00", "fvnn003 final checkpoint, deterministic, 363 d, from 2200"),
           ("fvnn10y", "fvnn003 final checkpoint, deterministic, 3650 d, from 2200"),
           ("fvzfe09", "fvzfe (forecast engine frozen), deterministic, 363 d"),
           ("fvzfe00", "fvzfe (forecast engine frozen), deterministic, 363 d, from 2200"),
           ("fvzal09", "fvzal (forecast, aggregation, global frozen), deterministic, 363 d"),
           ("fvzal00", "fvzal (forecast, aggregation, global frozen), deterministic, 363 d, from 2200"),
           ("fvofe09", "fvofe (only the forecast engine trains), deterministic, 363 d"),
           ("fvofe00", "fvofe (only the forecast engine trains), deterministic, 363 d, from 2200"),
           *[(f"srwig{e}_{y}", f"srwig (fvofe recipe + S2RoPE-Wigner L=10 in the forecast engine), "
              f"checkpoint {c}, deterministic, 363 d" + (", from 2200" if y == "00" else ""))
             for e, c in ((4, 4), (8, 8), (12, 10)) for y in ("09", "00")]]
_ALL = [r for r, _ in D1 + D2 + NONOISE] + [r for v in D9.values() for r, _ in v]
for _r in ("fvnnf00", "fvnn10y", "fvzfe00", "fvzal00", "fvofe00", "srwig4_00", "srwig8_00",
           "srwig12_00"):
    ams.RUNS.setdefault(_r, ("3", "2200-01-01", "real"))
for _r in _ALL:
    ams.RUNS.setdefault(_r, ("3", START, "real"))


def path(run):
    return REF_DIRS.get(run, DATA) / f"part1_daily_{run}.nc"


def compute(runs, workers, max_lead=150):
    ams.DATA = DATA if max_lead == 150 else DATA / f"lead{max_lead}"
    ams.MAX_LEAD = max_lead
    if max_lead > 150:
        # read by clim_extra in the (forked) workers
        ams.CLIM_LEADS = {1} | set(range(5, max_lead + 1, 5)) | {max_lead}
    ams.DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ams.DATA) as tmp:
        for r in runs:
            if r in REF_DIRS:
                raise ValueError(f"{r} is a reference run, scored elsewhere")
            ams.compute(r, workers, tmp)


def load(runs):
    dss, ms = {}, {}
    for r in runs:
        if path(r).exists():
            dss[r] = xr.open_dataset(path(r))
            ms[r] = ams.metrics(dss[r])
    return dss, ms


def fmt(x, nd=3):
    return ams.fmt(x, nd)


def onset_num(x, key):
    """Onset day, None (no onset by the max lead) -> max lead + 1 (a lower bound)."""
    v = x[key]
    return float(v) if v is not None else float(x["maxlead"] + 1)


def rmse(ds, c, d):
    sc = 1e3 if c == "ssh" else 1.0
    return float(ds.rmse.sel(channel=c, lead=d)) * sc if d <= int(ds.lead.max()) else np.nan


def clim_ratio(ds, c, d):
    clim = ds[f"clim_rmse_{c}"].sel(lead=d)
    return float(ds.rmse.sel(channel=c, lead=d) / clim)


def score_header():
    return ["| run | setting | onset salt_75>0.5 | onset temp_2>2 | e-fold 20-0.3 (R2) "
            "| e-fold 30-80 | ac1 salt_75 err (window) | ac1 hc err | hp hc (ZJ) "
            "| hp hc 76-90 (ZJ) | n non-finite |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]


def score_row(r, lab, x, ds):
    ef = (f"{x['efold']:.1f}{'' if x['d03'] else ' (no 0.3 crossing)'} "
          f"({x['ef_r2']:.2f})")
    hp7690 = x["hp_win"][ams.WINDOWS.index((76, 90))]
    nnf = int(np.nansum(ds.n_nonfinite_pred_wet.values))
    return (f"| {r} | {lab} | {fmt(x['on_s75'])} | {fmt(x['on_t2'])} | {ef} "
            f"| {x['efold3080']:.1f} | {x['ac1_s75']:.2f} ({x['pre_win'][0]}-{x['pre_win'][1]}) "
            f"| {x['ac1_hc']:.2f} | {x['hp_hc']:.3g} | {hp7690:.3g} | {nnf} |")


def rmse_tables(items, dss, lines):
    for c in CH:
        unit = "mm" if c == "ssh" else ("K" if c.startswith("temp") else "psu")
        lines += [f"### RMSE {c} ({unit})", "",
                  "| run | setting | " + " | ".join(f"d{d}" for d in TABLE_DAYS) + " |",
                  "|---|---|" + "---|" * len(TABLE_DAYS)]
        for r, lab in items:
            if r in dss:
                lines.append(f"| {r} | {lab} | "
                             + " | ".join(f"{rmse(dss[r], c, d):.3g}" for d in TABLE_DAYS) + " |")
        lines.append("")
    lines += ["### RMSE / climatology RMSE (2180-2209 day-of-year climatology vs the same truth)",
              "", "| run | " + " | ".join(f"{c} d{d}" for c in ("salt_75", "temp_2")
                                            for d in CLIM_DAYS) + " |",
              "|---|" + "---|" * (2 * len(CLIM_DAYS))]
    for r, _ in items:
        if r in dss:
            lines.append(f"| {r} | " + " | ".join(
                f"{clim_ratio(dss[r], c, d):.2f}" for c in ("salt_75", "temp_2")
                for d in CLIM_DAYS) + " |")
    lines.append("")


def figure(groups, dss, name, title):
    """RMSE salt_75 / temp_2 / ssh and the heat-content error vs lead for groups of runs."""
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.3), constrained_layout=True)
    thr = {"salt_75": 0.5, "temp_2": 2.0}
    for r, lab, sty in groups:
        if r not in dss:
            continue
        ds = dss[r]
        for ax, c in zip(axs[:3], ("salt_75", "temp_2", "ssh"), strict=True):
            sc = 1e3 if c == "ssh" else 1.0
            ax.semilogy(ds.lead, ds.rmse.sel(channel=c) * sc, label=f"{r}: {lab}", **sty)
        sc, _ = af.scale_unit("hc_0_2150")
        axs[3].plot(ds.lead, af.err(ds, "hc_0_2150") * sc, **sty)
    for ax, c in zip(axs[:3], ("salt_75", "temp_2", "ssh"), strict=True):
        if c in thr:
            ax.axhline(thr[c], color="gray", lw=0.8, ls=":")
        ax.set_title(f"RMSE {c}" + (" (mm)" if c == "ssh" else ""))
    axs[3].set_title(f"global-mean error hc_0_2150 ({af.scale_unit('hc_0_2150')[1]})")
    for ax in axs:
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


def report_d1(lines):
    dss, ms = load([r for r, _ in D1])
    lines += ["# D1: forcing injection split into constant and anomaly, fvars003, land freeze, "
              f"from {START}, 150 d", "",
              "Injected field a_c * C + a_a * (f_n - C), C = 2209 time mean of f_n "
              "(forecast.forcing_split); forcing_scale alpha scales both (a_c = a_a = alpha).", "",
              "## Onset, growth, alternation", ""] + score_header()
    for r, lab in D1:
        if r in ms:
            lines.append(score_row(r, lab, ms[r], dss[r]))
    lines.append("")
    rmse_tables(D1, dss, lines)
    # readout: onset change per unit change of each scale
    lines += ["## Readout: onset change against the unscaled freeze run (lcfz003)", "",
              "| run | a_c | a_a | d onset salt_75 (d) | d onset temp_2 (d) | d e-fold 30-80 (d) "
              "| d onset s75 per unit scale change |", "|---|---|---|---|---|---|---|"]
    scales = {"spa050y9": (1, 0.5), "spa150y9": (1, 1.5), "spc080y9": (0.8, 1),
              "spc125y9": (1.25, 1), "fs050y09": (0.5, 0.5), "fs075y09": (0.75, 0.75),
              "fs125y09": (1.25, 1.25)}
    if "lcfz003" in ms:
        ref = ms["lcfz003"]
        for r, (ac, aa) in scales.items():
            if r not in ms:
                continue
            x = ms[r]
            ds75 = onset_num(x, "on_s75") - onset_num(ref, "on_s75")
            dt2 = onset_num(x, "on_t2") - onset_num(ref, "on_t2")
            de = x["efold3080"] - ref["efold3080"]
            du = max(abs(ac - 1), abs(aa - 1))
            cap = ">=" if x["on_s75"] is None else ""
            lines.append(f"| {r} | {ac} | {aa} | {cap}{ds75:+.0f} | "
                         f"{'>=' if x['on_t2'] is None else ''}{dt2:+.0f} | {de:+.1f} "
                         f"| {cap}{ds75 / du:+.0f} |")
    lines.append("")
    styles = [("lcfz003", "unscaled", {"color": "k", "lw": 1.4}),
              ("spa050y9", "anom 0.5", {"color": "tab:blue"}),
              ("spa150y9", "anom 1.5", {"color": "tab:red"}),
              ("spc080y9", "const 0.8", {"color": "tab:blue", "ls": "--"}),
              ("spc125y9", "const 1.25", {"color": "tab:red", "ls": "--"}),
              ("fs050y09", "forcing_scale 0.5", {"color": "tab:blue", "ls": ":", "lw": 0.9}),
              ("fs125y09", "forcing_scale 1.25", {"color": "tab:red", "ls": ":", "lw": 0.9})]
    figure(styles, dss, "d1_forcing_split_rmse.png",
           "D1: fvars003 freeze, constant / anomaly of the forcing injection scaled separately")


def report_d2(lines):
    dss, ms = load([r for r, _ in D2])
    lines += ["# D2: inference in the training-noise regime, fvars003 default (no freeze), "
              f"from {START}, 150 d", "",
              "forecast.fe_noise at the FE input every step (seed 0). train_like: randn * "
              "|z|_F * std (the training expression; per-entry std ~ 0.5 x token RMS at 1e-4); "
              "token_rms: randn * rms(token) * std; dropout: FE blocks in train mode (p 0.1).",
              "", "## Onset, growth, alternation", ""] + score_header()
    for r, lab in D2:
        if r in ms:
            lines.append(score_row(r, lab, ms[r], dss[r]))
    lines.append("")
    rmse_tables(D2, dss, lines)
    styles = [("yr1fv003", "deterministic", {"color": "k", "lw": 1.4}),
              ("nztld09", "train_like + dropout", {"color": "tab:purple"}),
              ("nztl09", "train_like", {"color": "tab:purple", "ls": "--"}),
              ("nzr01y09", "token_rms 0.01", {"color": "tab:green"}),
              ("nzr03y09", "token_rms 0.03", {"color": "tab:orange"}),
              ("nzr10y09", "token_rms 0.10", {"color": "tab:red"})]
    figure(styles, dss, "d2_fe_noise_rmse.png",
           "D2: fvars003 default, latent noise / dropout at the FE input in inference")


def report_d9(lines):
    lines += ["# D9: onset spread, fvars003, token_rms noise 1e-3 at the first step only, "
              f"from {START}, 150 d", ""]
    allds = {}
    for grp, items in D9.items():
        dss, ms = load([r for r, _ in items])
        allds.update(dss)
        lines += [f"## {grp}", ""] + score_header()
        for r, lab in items:
            if r in ms:
                lines.append(score_row(r, lab, ms[r], dss[r]))
        mem = [r for r, _ in items if r.startswith("d9") and r in ms]
        if mem:
            lines += ["", "Members (seeds 1-3; the unperturbed run is not included in the "
                      "statistics). 'none' onsets are counted as 151 (a lower bound).", "",
                      "| quantity | members | mean | std (ddof 1) | range | unperturbed |",
                      "|---|---|---|---|---|---|"]
            unp = [r for r, _ in items if not r.startswith("d9")][0]
            for key, name in (("on_s75", "onset salt_75>0.5"), ("on_t2", "onset temp_2>2"),
                              ("efold", "e-fold 20-0.3"), ("efold3080", "e-fold 30-80"),
                              ("ac1_s75", "ac1 salt_75 err"), ("ac1_hc", "ac1 hc err"),
                              ("hp_hc", "hp hc (ZJ)")):
                if key.startswith("on_"):
                    v = np.array([onset_num(ms[r], key) for r in mem])
                    u = onset_num(ms[unp], key) if unp in ms else np.nan
                else:
                    v = np.array([ms[r][key] for r in mem], float)
                    u = ms[unp][key] if unp in ms else np.nan
                sd = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
                lines.append(f"| {name} | " + ", ".join(f"{x:.3g}" for x in v)
                             + f" | {v.mean():.3g} | {sd:.3g} | {np.ptp(v):.3g} | {u:.3g} |")
        lines.append("")
        rmse_tables(items, dss, lines)
    styles = [("yr1fv003", "default, unperturbed", {"color": "k", "lw": 1.4}),
              ("lcfz003", "freeze, unperturbed", {"color": "tab:blue", "lw": 1.4})]
    styles += [(f"d9def{i}", f"default seed {i}", {"color": "gray", "lw": 0.8}) for i in (1, 2, 3)]
    styles += [(f"d9frz{i}", f"freeze seed {i}", {"color": "tab:cyan", "lw": 0.8})
               for i in (1, 2, 3)]
    figure(styles, allds, "d9_onset_spread_rmse.png",
           "D9: fvars003, token_rms 1e-3 noise at the first forecast step, 3 seeds")


def _fmt_on(v, top):
    return f"none<={top}" if v is None else str(int(v))


def report_year(lines):
    """Year-long train_like-noise runs vs the deterministic runs of the same start and yr1win03."""
    top = YEAR_LEAD
    ydir = DATA / f"lead{top}"
    dss, ms = {}, {}
    for r in YEAR:
        p = ydir / f"part1_daily_{r}.nc"
        if p.exists():
            dss[r] = xr.open_dataset(p)
            ms[r] = ams.metrics(dss[r])
    for r, p in YEAR_REF.items():
        if p.exists():
            dss[r] = xr.open_dataset(p)
    for _, (_, ref) in YEAR.items():
        if ref not in dss and (REF_DIRS["yr1fv003"] / f"part1_daily_{ref}.nc").exists():
            dss[ref] = xr.open_dataset(REF_DIRS["yr1fv003"] / f"part1_daily_{ref}.nc")
    if "yr1fv003" in dss:
        ms["yr1fv003"] = ams.metrics(dss["yr1fv003"])
    for _, ref in YEAR.values():
        if ref in dss and ref not in ms:
            ms[ref] = ams.metrics(dss[ref])
    clim_src = {"yr1fv003": "nztl09", "yr1win03": "nztl09", "nztl09": "nztl09",
                "ms3a2150": "ms3a2150", "ms3a2200": "ms3a2200", "nztl50": "nztl50",
                "nztl00": "nztl00"}
    order = []
    for r, (st, ref) in YEAR.items():
        order += [(r, st, "train_like 1e-4, no dropout"), (ref, st, "deterministic default [ref]")]
    order.append(("yr1win03", "2209-01-01", "old lineage (frcwin03), deterministic [ref]"))

    def val(r, c, d):
        ds = dss.get(r)
        if ds is None or d > int(ds.lead.max()):
            return np.nan
        return float(ds.rmse.sel(channel=c, lead=d)) * (1e3 if c == "ssh" else 1.0)

    def clim(r, c, d):
        ds = dss.get(clim_src[r])
        if ds is None or f"clim_rmse_{c}" not in ds or d > int(ds.lead.max()):
            return np.nan
        return float(ds[f"clim_rmse_{c}"].sel(lead=d)) * (1e3 if c == "ssh" else 1.0)

    # truth identity of the references that borrow nztl09's climatology RMSE
    chk = []
    for r in ("yr1fv003",):
        if r in dss and "nztl09" in dss:
            a, b = dss[r], dss["nztl09"]
            n = min(int(a.lead.max()), int(b.lead.max()))
            ta = a.truth.sel(lead=slice(1, n)).values
            tb = b.truth.sel(lead=slice(1, n), quantity=a.quantity).values
            chk.append(f"{r} vs nztl09 truth (global means, leads 1-{n}): max |diff| = "
                       f"{float(np.nanmax(np.abs(ta - tb))):.3g}")
    lines += [f"# Year-long inference with train_like latent noise (fvars003, {top} days)", "",
              "forecast.fe_noise mode train_like, std 1e-4, no dropout, every step, seed 0, no "
              "land freeze. Deterministic references: yr1fv003 (2209, 363 d), ms3a2150 / ms3a2200 "
              "(150 d only). Old lineage: yr1win03 (frcwin03, 2209, 363 d, series from "
              "oldlineage_series.py). Climatology = 2180-2209 day-of-year climatology vs the "
              "same truth; references from 2209 use nztl09's climatology RMSE (same truth).", ""]
    if chk:
        lines += [f"- {x}" for x in chk] + [""]
    lines += ["## Onset, growth, alternation", "",
              "| run | start | setting | max lead | onset salt_75>0.5 | onset temp_2>2 "
              "| e-fold 20-0.3 (R2) | e-fold 30-80 | ac1 salt_75 err (window) | ac1 hc err "
              "| hp hc (ZJ) | n non-finite |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, st, lab in order:
        if r not in dss:
            continue
        ds = dss[r]
        ml = int(ds.lead.max())
        if r in ms:
            x = ms[r]
            ef = (f"{x['efold']:.1f}{'' if x['d03'] else ' (no 0.3 crossing)'} "
                  f"({x['ef_r2']:.2f})")
            nnf = int(np.nansum(ds.n_nonfinite_pred_wet.values))
            lines.append(f"| {r} | {st} | {lab} | {ml} | {_fmt_on(x['on_s75'], ml)} "
                         f"| {_fmt_on(x['on_t2'], ml)} | {ef} | {x['efold3080']:.1f} "
                         f"| {x['ac1_s75']:.2f} ({x['pre_win'][0]}-{x['pre_win'][1]}) "
                         f"| {x['ac1_hc']:.2f} | {x['hp_hc']:.3g} | {nnf} |")
        else:
            lead = ds.lead.values
            o75 = af.first_exceed(ds.rmse.sel(channel="salt_75").values, lead, 0.5)
            ot2 = af.first_exceed(ds.rmse.sel(channel="temp_2").values, lead, 2.0)
            lines.append(f"| {r} | {st} | {lab} | {ml} | {_fmt_on(o75, ml)} | {_fmt_on(ot2, ml)} "
                         "| - | - | - | - | - | - |")
    lines.append("")
    for c in CH:
        unit = "mm" if c == "ssh" else ("K" if c.startswith("temp") else "psu")
        lines += [f"## RMSE {c} ({unit})", "",
                  "| run | start | " + " | ".join(f"d{d}" for d in YEAR_DAYS) + " |",
                  "|---|---|" + "---|" * len(YEAR_DAYS)]
        for r, st, _ in order:
            if r in dss and c in dss[r].channel.values:
                lines.append(f"| {r} | {st} | " + " | ".join(
                    ams.fmt(val(r, c, d)) for d in YEAR_DAYS) + " |")
        for st, src in (("2209-01-01", "nztl09"), ("2150-01-01", "nztl50"),
                        ("2200-01-01", "nztl00")):
            if src in dss:
                lines.append(f"| (clim) | {st} | " + " | ".join(
                    ams.fmt(clim(src, c, d)) for d in YEAR_DAYS) + " |")
        lines.append("")
    lines += ["## RMSE / climatology RMSE", ""]
    for c in ("salt_75", "temp_2", "ssh", "salt_2"):
        lines += [f"### {c}", "", "| run | start | " + " | ".join(
            f"d{d}" for d in YEAR_CLIM_DAYS) + " |", "|---|---|" + "---|" * len(YEAR_CLIM_DAYS)]
        for r, st, _ in order:
            if r in dss and c in dss[r].channel.values:
                q = [val(r, c, d) / clim(r, c, d) for d in YEAR_CLIM_DAYS]
                lines.append(f"| {r} | {st} | " + " | ".join(
                    f"{v:.2f}" if np.isfinite(v) else "-" for v in q) + " |")
        lines.append("")
    # figure: RMSE vs lead with the climatology level
    cols = {"2209-01-01": "tab:purple", "2150-01-01": "tab:green", "2200-01-01": "tab:orange"}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for ax, c in zip(axs, ("salt_75", "temp_2", "ssh"), strict=True):
        sc = 1e3 if c == "ssh" else 1.0
        for r, st, lab in order:
            if r not in dss or c not in dss[r].channel.values:
                continue
            ds = dss[r]
            if r == "yr1win03":
                sty = {"color": "0.35", "lw": 1.1, "ls": "-."}
            elif r in YEAR:
                sty = {"color": cols[st], "lw": 1.6}
            else:
                sty = {"color": cols[st], "lw": 1.0, "ls": "--"}
            ax.semilogy(ds.lead, ds.rmse.sel(channel=c) * sc, label=f"{r} ({st[:4]}): {lab}",
                        **sty)
        for st, src in (("2209-01-01", "nztl09"), ("2150-01-01", "nztl50"),
                        ("2200-01-01", "nztl00")):
            if src in dss:
                cl = dss[src][f"clim_rmse_{c}"].dropna("lead")
                ax.plot(cl.lead, cl * sc, color=cols[st], lw=0.9, ls=":",
                        label=f"climatology ({st[:4]})")
        ax.set_title(f"RMSE {c}" + (" (mm)" if c == "ssh" else " (psu)" if "salt" in c
                                    else " (K)"))
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.suptitle("fvars003, train_like latent noise 1e-4 (solid) vs deterministic (dashed), "
                 "old lineage yr1win03 (dash-dot); dotted: 2180-2209 day-of-year climatology")
    fig.savefig(FIG / "year_noise_rmse.png", dpi=120)
    plt.close(fig)


def report(which):
    FIG.mkdir(parents=True, exist_ok=True)
    for name, fn in (("D1", report_d1), ("D2", report_d2), ("D9", report_d9),
                     ("YEAR", report_year)):
        if name == "YEAR" and name not in (which or []):
            continue
        if which and name not in which:
            continue
        lines = []
        fn(lines)
        txt = "\n".join(lines) + "\n"
        d = OUT / ("YEAR_NOISE" if name == "YEAR" else name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "tables.md").write_text(txt)
        print(txt)  # noqa: T201


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("runs", nargs="*",
                   help="compute: run ids; report: D1 / D2 / D9 (default all) / YEAR")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-lead", type=int, default=150)
    a = p.parse_args()
    if a.cmd == "compute":
        compute(a.runs, a.workers, a.max_lead)
    else:
        report(a.runs)


if __name__ == "__main__":
    main()
