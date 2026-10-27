"""
Multi-start unfiltered 150-day rollouts of fvars002 / fvars003 (combined_report.md section 5,
step 1): does the blow-up happen from initial dates inside / near the training period
(2010-2199, validation 2200) as from 2209-01-01, and at the same salt_75 error?

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_multistart.py \
        compute RUN [RUN ...] [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/analyse_multistart.py report

compute  daily global series of a run (leads 1-150) with analyse_filter.compute, i.e.
         diag_fvars_year's per-lead code, plus the RMSE of the 2180-2209 day-of-year
         climatology against the same truth (temp_2, salt_2, salt_75, ssh) every 5 days
         -> DATA/part1_daily_RUN.nc and DATA/perlead_RUN.csv; checks that the initial time
         read from the output is the run's start date.
report   onset days, pre-onset growth, ac1 / 2-day high-pass, day-1 volume-mean T error and
         RMSE at fixed days per model and start date -> OUT/tables.md, figures -> FIG/.
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
import analyse_filter as af  # noqa: E402

dfy = af.dfy
from paper_eval.climatology import ChannelClimLookup  # noqa: E402

OUT = Path("/home/a/a270225/claude_api_tools/multistart")
DATA = OUT / "data"
FIG = af.MAIN / "paper_eval/figures/multistart"
CLIM_DIR = af.MAIN / "paper_eval/cache_oe3i7whz/climatology"
CLIM_PATHS = [CLIM_DIR / "clim_FESOM_NODE_2180-2209.zarr",
              CLIM_DIR / "clim_FESOM_NODE_extra_2180-2209.zarr"]
CLIM_CH = ("temp_2", "salt_2", "salt_75", "ssh")
CLIM_STREAM = {"temp_2": "TEMP", "salt_2": "SALT", "salt_75": "SALT", "ssh": "SURFACE"}
CLIM_LEADS = {1} | set(range(5, 151, 5))
MAX_LEAD = 150

#: run id -> (model digit, initial date, forcing label)
RUNS = {}
for _m in "32":
    for _tag, _d in (("a2150", "2150-01-01"), ("g2190", "2190-07-01"), ("a2200", "2200-01-01"),
                     ("a2202", "2202-01-01"), ("a2205", "2205-01-01"), ("g2209", "2209-07-01")):
        RUNS[f"ms{_m}{_tag}"] = (_m, _d, "real")
    RUNS[f"yr1fv00{_m}"] = (_m, "2209-01-01", "real")
    RUNS[f"yr1fc00{_m}"] = (_m, "2209-01-01", "clim 2170-2199")

TABLE_DAYS = (1, 15, 30, 60, 90, 150)
RMSE_TABLE_CH = ("temp_2", "salt_2", "salt_75", "ssh")
WINDOWS = ((2, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 105), (106, 120),
           (121, 150))


def runs_of(m):
    """Runs of model m in initial-date order (2209-01-01 clim-forcing run last)."""
    rs = [r for r, v in RUNS.items() if v[0] == m and v[2] == "real"]
    return sorted(rs, key=lambda r: RUNS[r][1]) + [f"yr1fc00{m}"]


def label(r):
    m, d, f = RUNS[r]
    return f"{d} ({r})" + ("" if f == "real" else ", clim forcing")


# ── compute ──────────────────────────────────────────────────────────────────────────
_C = {}


def clim_extra(lead):
    """RMSE of the 2180-2209 climatology vs the run's truth at CLIM_LEADS (NaN elsewhere)."""
    out = {f"clim_rmse_{c}": np.nan for c in CLIM_CH}
    if lead not in CLIM_LEADS:
        return out
    if not _C:
        _C["clim"] = ChannelClimLookup(CLIM_PATHS, list(CLIM_CH))
    root, s, geo = dfy._W["root"], dfy._W["sample"], dfy._W["geo"]
    valid = np.datetime64(root[f"{s}/TEMP/{lead}/target/times"][0])
    clim = np.asarray(_C["clim"].get(valid), np.float64)
    if clim.shape != (geo.n, len(CLIM_CH)):
        raise ValueError(f"climatology shape {clim.shape}")
    for j, c in enumerate(CLIM_CH):
        tg = root[f"{s}/{CLIM_STREAM[c]}/{lead}/target"]
        t = np.asarray(tg["data"])[:, list(tg.attrs["channels"]).index(c)].astype(np.float64)
        mask, w = geo.weight(c)
        w = w * mask
        e = np.where(mask, clim[:, j] - t, 0.0)
        out[f"clim_rmse_{c}"] = float(np.sqrt((w * e ** 2).sum() / w.sum()))
    return out


def compute(run, workers, tmp):
    m, start, _ = RUNS[run]
    path = af.compute(run, workers, tmp, model=m, data=DATA, max_lead=MAX_LEAD, extra=clim_extra)
    ds = xr.open_dataset(path)
    want = f"{start}T23:30"
    if not ds.attrs["init_time"].startswith(want):
        raise ValueError(f"{run}: init_time {ds.attrs['init_time']} != expected {want}")
    lead = ds.lead.values
    cols = {"lead": lead, "valid_time": ds.valid_time.values.astype("datetime64[m]").astype(str)}
    for c in ds.channel.values:
        cols[f"rmse_{c}"] = ds.rmse.sel(channel=c).values
    for q in ds.quantity.values:
        cols[f"pred_{q}"] = ds.pred.sel(quantity=q).values
        cols[f"truth_{q}"] = ds.truth.sel(quantity=q).values
    for c in CLIM_CH:
        cols[f"clim_rmse_{c}"] = ds[f"clim_rmse_{c}"].values
    cols["n_nonfinite_pred_wet"] = ds.n_nonfinite_pred_wet.values
    import pandas as pd

    pd.DataFrame(cols).to_csv(DATA / f"perlead_{run}.csv", index=False, float_format="%.6g")
    print(f"{run}: init {ds.attrs['init_time']} OK, csv -> {DATA / f'perlead_{run}.csv'}")


# ── report ───────────────────────────────────────────────────────────────────────────
def load(run):
    return xr.open_dataset(DATA / f"part1_daily_{run}.nc")


def fmt(x, nd=3):
    if x is None:
        return "none<=150"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not np.isfinite(x):
        return "-"
    return f"{x:.{nd}g}"


def metrics(ds):
    lead = ds.lead.values
    s75 = ds.rmse.sel(channel="salt_75").values
    t2 = ds.rmse.sel(channel="temp_2").values
    r = {"maxlead": int(lead.max()),
         "on_s75": af.first_exceed(s75, lead, 0.5), "on_s75_1": af.first_exceed(s75, lead, 1.0),
         "on_t2": af.first_exceed(t2, lead, 2.0),
         "on_t450": af.first_exceed(ds.rmse.sel(channel="temp_450").values, lead, 1.0),
         "d03": af.first_exceed(s75, lead, 0.3)}
    on = r["on_s75"]
    r["pre"] = {d: float(s75[lead == d][0]) for d in range(on - 5, on + 1)} if on else {}
    b = r["d03"] if r["d03"] is not None else int(lead.max())
    r["ef_win"] = (20, b)
    sel = (lead >= 20) & (lead <= b)
    if sel.sum() >= 3:
        x, y = lead[sel], np.log(s75[sel])
        coef = np.polyfit(x, y, 1)
        r["efold"] = 1.0 / coef[0] if coef[0] > 0 else np.inf
        r["ef_r2"] = 1 - ((y - np.polyval(coef, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    else:
        r["efold"], r["ef_r2"] = np.nan, np.nan
    r["efold3080"] = af.efold(s75, lead, 30, 80)
    end = (on - 1) if on else int(lead.max())
    r["pre_win"] = (16, end)
    ds_s = af.err(ds, "salt_75")
    hc = af.err(ds, "hc_0_2150") * 1e-21
    r["ac1_s75"] = af.ac1(ds_s, lead, 16, end) if end > 20 else np.nan
    r["ac1_hc"] = af.ac1(hc, lead, 16, end) if end > 20 else np.nan
    r["hp_hc"] = af.hp_rms(hc, lead, 16, end) if end > 20 else np.nan
    r["hp_s75"] = af.hp_rms(ds_s, lead, 16, end) if end > 20 else np.nan
    r["ac1_win"] = [af.ac1(ds_s, lead, a, b2) for a, b2 in WINDOWS]
    r["ac1_win_hc"] = [af.ac1(hc, lead, a, b2) for a, b2 in WINDOWS]
    r["hp_win"] = [af.hp_rms(hc, lead, a, b2) for a, b2 in WINDOWS]
    pre_w = [v for (a, b2), v in zip(WINDOWS, r["ac1_win"]) if a >= 16 and b2 <= end]
    r["ac1_prewin_range"] = (min(pre_w), max(pre_w)) if pre_w else None
    r["vmT1"] = float(af.err(ds, "vmT_0_2150")[lead == 1][0])
    r["hc1"] = float(hc[lead == 1][0])
    r["vmS1"] = float(af.err(ds, "vmS_0_2150")[lead == 1][0])
    return r


def consistency(lines):
    """Recomputed yr1fv00X vs the diag_yr1 series (same code, leads 1-150)."""
    lines += ["## Consistency check", ""]
    for m in "32":
        run = f"yr1fv00{m}"
        if not (DATA / f"part1_daily_{run}.nc").exists():
            continue
        a = load(run)
        b = xr.open_dataset(af.REF / f"part1_daily_{run}.nc").sel(lead=a.lead)
        d_r = float(np.nanmax(np.abs(a.rmse.values - b.rmse.values)))
        d_p = float(np.nanmax(np.abs(a.pred.values - b.pred.values) / (np.abs(b.pred.values) + 1e-30)))
        lines.append(f"- {run} leads 1-{int(a.lead.max())}: max |RMSE diff| vs "
                     f"diag_yr1/part1_daily_{run}.nc = {d_r:.3g}, max relative diff of the global "
                     f"means = {d_p:.3g}")
    lines.append("")


def report():
    FIG.mkdir(parents=True, exist_ok=True)
    lines = []
    allm = {}
    for m in "32":
        runs = [r for r in runs_of(m) if (DATA / f"part1_daily_{r}.nc").exists()]
        dss = {r: load(r) for r in runs}
        ms = {r: metrics(dss[r]) for r in runs}
        allm[m] = (runs, dss, ms)
        lines += [f"## fvars00{m}", ""]
        lines += ["### Onset, pre-onset growth, alternation, day-1 offset", "",
                  "| start (run) | onset salt_75>0.5 | salt_75>1.0 | onset temp_2>2 | temp_450>1 "
                  "| salt_75 RMSE d(on-5..on-1) / on | e-fold (window, R2) | e-fold 30-80 "
                  "| ac1 salt_75 err (window) | ac1 hc err | hp hc (ZJ) | day-1 vmT err (K) "
                  "| day-1 hc err (ZJ) | day-1 vmS err (psu) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in runs:
            x = ms[r]
            pre = (" ".join(f"{v:.3f}" for d, v in x["pre"].items() if d < x["on_s75"]) +
                   f" / {x['pre'][x['on_s75']]:.3f}") if x["on_s75"] else "-"
            ef = f"{x['efold']:.1f} d (20-{x['ef_win'][1]}{'' if x['d03'] else ', no 0.3 crossing'}, {x['ef_r2']:.2f})"
            win = f"{x['pre_win'][0]}-{x['pre_win'][1]}"
            lines.append(
                f"| {label(r)} | {fmt(x['on_s75'])} | {fmt(x['on_s75_1'])} | {fmt(x['on_t2'])} "
                f"| {fmt(x['on_t450'])} | {pre} | {ef} | {x['efold3080']:.1f} "
                f"| {x['ac1_s75']:.2f} ({win}) | {x['ac1_hc']:.2f} | {x['hp_hc']:.3g} "
                f"| {x['vmT1']:+.4f} | {x['hc1']:+.0f} | {x['vmS1']:+.5f} |")
        lines.append("")
        lines += ["### ac1 of daily increments of the salt_75 global-mean error, per window", "",
                  "| start (run) | " + " | ".join(f"{a}-{b}" for a, b in WINDOWS) + " |",
                  "|---|" + "---|" * len(WINDOWS)]
        for r in runs:
            lines.append(f"| {label(r)} | " + " | ".join(f"{v:.2f}" for v in ms[r]["ac1_win"]) + " |")
        lines += ["", "### ac1 of daily increments of the heat-content (0-2150 m) error, per window", "",
                  "| start (run) | " + " | ".join(f"{a}-{b}" for a, b in WINDOWS) + " |",
                  "|---|" + "---|" * len(WINDOWS)]
        for r in runs:
            lines.append(f"| {label(r)} | " + " | ".join(f"{v:.2f}" for v in ms[r]["ac1_win_hc"]) + " |")
        lines += ["", "### 2-day high-pass RMS of the heat-content error (ZJ), per window", "",
                  "| start (run) | " + " | ".join(f"{a}-{b}" for a, b in WINDOWS) + " |",
                  "|---|" + "---|" * len(WINDOWS)]
        for r in runs:
            lines.append(f"| {label(r)} | " + " | ".join(f"{v:.3g}" for v in ms[r]["hp_win"]) + " |")
        lines.append("")
        for c in RMSE_TABLE_CH:
            sc, unit = (1e3, "mm") if c == "ssh" else (1.0, "K" if c.startswith("temp") else "psu")
            lines += [f"### RMSE {c} ({unit}); in brackets: 2180-2209 climatology RMSE vs the same truth", "",
                      "| start (run) | " + " | ".join(f"d{d}" for d in TABLE_DAYS) + " |",
                      "|---|" + "---|" * len(TABLE_DAYS)]
            for r in runs:
                ds = dss[r]
                cells = []
                for d in TABLE_DAYS:
                    v = float(ds.rmse.sel(channel=c, lead=d)) * sc
                    cl = float(ds[f"clim_rmse_{c}"].sel(lead=d)) * sc
                    cells.append(f"{v:.3g} ({cl:.3g})")
                lines.append(f"| {label(r)} | " + " | ".join(cells) + " |")
            lines.append("")
        figure_model(m, runs, dss)
    consistency(lines)
    figure_dates(allm)
    txt = "\n".join(lines) + "\n"
    (OUT / "tables.md").write_text(txt)
    print(txt)
    return allm


COLORS = {"2150-01-01": "tab:blue", "2190-07-01": "tab:cyan", "2200-01-01": "tab:green",
          "2202-01-01": "tab:olive", "2205-01-01": "tab:orange", "2209-01-01": "k",
          "2209-07-01": "tab:red"}


def _style(r):
    _, d, f = RUNS[r]
    return dict(color=COLORS[d], ls="--" if f != "real" else "-", lw=1.2)


def figure_model(m, runs, dss):
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    thr = {"salt_75": 0.5, "temp_2": 2.0}
    for ax, c in zip(axs.flat[:4], RMSE_TABLE_CH):
        sc = 1e3 if c == "ssh" else 1.0
        for r in runs:
            ds = dss[r]
            ax.semilogy(ds.lead, ds.rmse.sel(channel=c) * sc, label=label(r), **_style(r))
            cl = ds[f"clim_rmse_{c}"]
            ok = np.isfinite(cl.values)
            ax.semilogy(ds.lead[ok], cl[ok] * sc, ".", ms=3, color=_style(r)["color"])
        if c in thr:
            ax.axhline(thr[c], color="gray", lw=0.8, ls=":")
        ax.set_title(f"RMSE {c}" + (" (mm)" if c == "ssh" else "") + "; dots: clim 2180-2209")
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=7)
    for ax, q in zip(axs.flat[4:6], ("salt_75", "hc_0_2150")):
        sc, unit = af.scale_unit(q)
        for r in runs:
            ds = dss[r]
            ax.plot(ds.lead, af.err(ds, q) * sc, **_style(r))
        ax.set_title(f"global-mean error {q} ({unit})")
        ax.set_xlabel("lead (days)")
        ax.grid(alpha=0.3)
    ax = axs.flat[6]
    for r in runs:
        ds = dss[r]
        ax.plot(ds.lead, af.err(ds, "vmT_0_2150"), **_style(r))
    ax.set_title("volume-mean T error 0-2150 m (K)")
    ax.set_xlabel("lead (days)")
    ax.grid(alpha=0.3)
    ax = axs.flat[7]
    mids = [0.5 * (a + b) for a, b in WINDOWS]
    for r in runs:
        ds = dss[r]
        ax.plot(mids, [af.ac1(af.err(ds, "salt_75"), ds.lead.values, a, b) for a, b in WINDOWS],
                "o-", ms=3, **_style(r))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("ac1 of daily increments, salt_75 mean error")
    ax.set_xlabel("lead window centre (days)")
    ax.grid(alpha=0.3)
    fig.suptitle(f"fvars00{m}: unfiltered 150-day rollouts from several initial dates "
                 "(dashed: 2209-01-01 with climatological forcing)")
    fig.savefig(FIG / f"multistart_fvars00{m}.png", dpi=110)
    plt.close(fig)


def figure_dates(allm):
    """One row per initial date: RMSE of the four channels, fvars003 vs fvars002."""
    dates = sorted({RUNS[r][1] for r in RUNS})
    fig, axs = plt.subplots(len(dates), 4, figsize=(18, 2.6 * len(dates)), constrained_layout=True,
                            sharex=True)
    for i, d in enumerate(dates):
        for j, c in enumerate(RMSE_TABLE_CH):
            ax = axs[i, j]
            sc = 1e3 if c == "ssh" else 1.0
            for m, col in (("3", "tab:blue"), ("2", "tab:red")):
                runs, dss, _ = allm[m]
                for r in runs:
                    if RUNS[r][1] != d:
                        continue
                    ds = dss[r]
                    ls = "-" if RUNS[r][2] == "real" else "--"
                    ax.semilogy(ds.lead, ds.rmse.sel(channel=c) * sc, color=col, ls=ls, lw=1.1,
                                label=f"fvars00{m} {r}")
                    cl = ds[f"clim_rmse_{c}"]
                    ok = np.isfinite(cl.values)
                    ax.semilogy(ds.lead[ok], cl[ok] * sc, ".", ms=3, color="gray")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"RMSE {c}" + (" (mm)" if c == "ssh" else ""))
            if j == 0:
                ax.set_ylabel(d)
                ax.legend(fontsize=6)
    for ax in axs[-1]:
        ax.set_xlabel("lead (days)")
    fig.suptitle("RMSE vs lead per initial date (grey dots: 2180-2209 climatology)")
    fig.savefig(FIG / "multistart_by_date.png", dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("runs", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "compute":
        DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=DATA) as tmp:
            for r in a.runs:
                compute(r, a.workers, tmp)
    else:
        report()


if __name__ == "__main__":
    main()
