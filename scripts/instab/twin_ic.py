"""
IC-memory twin runs (synth2_report.md section 3, "Cheap first test"): fvars003 with
land_cells freeze and forcing_scale alpha, started on 2208-12-01 (twin) and on 2209-01-01
(reference). From 2209-01-01 on both see identical forcing and differ only by the twin's
31-day forecast error. Does the twin - reference difference decay (the model forgets its
initial condition), and does the decay time tau shrink with alpha?

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/twin_ic.py \
        compute TWIN REF [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/twin_ic.py report

compute  per common valid date (reference lead L = 0..150, twin lead L + offset; the offset
         is read from the target times and the targets are checked to be identical): the
         area-weighted RMS of (twin - reference) predictions and each run's RMSE vs truth,
         for CH, in physical units -> DATA/twin_TWIN_REF.nc. Reference lead 0 is the
         reference's initial state, taken as the truth (the difference is then the twin's
         error vs truth).
report   tables (OUT/tables.md), decay-time fits and figures (FIG).
Weights and wet masks are diag_fvars_year's Geometry (nod_area per level, wet iff
k < nlevels_nod2D - 1; 2-D fields: nod_area[0], all nodes).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import diag_fvars_year as dfy  # noqa: E402
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402

MAIN = Path("/work/ab0995/a270225/WeatherGenerator2")
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
OUT = Path("/home/a/a270225/claude_api_tools/twin")
DATA = OUT / "data"
FIG = MAIN / "paper_eval/figures/twin"
CH = ("salt_75", "temp_2", "ssh", "salt_2", "temp_450")
STREAM = {"salt_75": "SALT", "salt_2": "SALT", "temp_2": "TEMP", "temp_450": "TEMP",
          "ssh": "SURFACE"}
#: alpha -> (twin, reference)
PAIRS = {0.75: ("tw075", "fs075y09"), 1.0: ("tw100", "lcfz003"), 1.25: ("tw125", "fs125y09")}
COL = {0.75: "tab:cyan", 1.0: "k", 1.25: "tab:red"}
VALID_S75 = 0.3  # fits use dates before either run's salt_75 RMSE exceeds this (psu)
TABLE_DAYS = (0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 135, 150)

_W = {}


def _init(twin, ref):
    _W["t"] = open_merged_ranks(MAIN / "results" / twin, 0)
    _W["r"] = open_merged_ranks(MAIN / "results" / ref, 0)
    _W["geo"] = dfy.Geometry(MESH)


def _fields(root, stream, lead, kind):
    s = root.store.samples[0]
    g = root[f"{s}/{stream}/{lead}/{kind}"]
    a = np.asarray(g["data"]).astype(np.float64)
    if kind == "prediction":
        a = a[..., 0]
    return list(g.attrs["channels"]), a


def times(root, lead):
    s = root.store.samples[0]
    return np.datetime64(root[f"{s}/TEMP/{lead}/target/times"][0])


def _pair(args):
    """Scores for one common date: twin lead lt, reference lead lr (0 = reference IC)."""
    lt, lr = args
    geo = _W["geo"]
    out = {}
    for stream in sorted(set(STREAM.values())):
        ch_t, tg = _fields(_W["t"], stream, lt, "target")
        _, pt = _fields(_W["t"], stream, lt, "prediction")
        if lr > 0:
            ch_r, tr = _fields(_W["r"], stream, lr, "target")
            _, pr = _fields(_W["r"], stream, lr, "prediction")
            if ch_r != ch_t:
                raise ValueError(f"{stream}: channel lists differ")
        for c in (c for c in CH if STREAM[c] == stream):
            j = ch_t.index(c)
            mask, w = geo.weight(c)
            w = w * mask
            t = np.where(mask, tg[:, j], 0.0)
            p_t = np.where(mask, pt[:, j], 0.0)
            if lr > 0:
                if not np.array_equal(np.where(mask, tr[:, j], 0.0), t):
                    raise ValueError(f"{c}: targets differ at twin lead {lt} / ref lead {lr}")
                p_r = np.where(mask, pr[:, j], 0.0)
            else:
                p_r = t
            rms = lambda e: float(np.sqrt((w * e ** 2).sum() / w.sum()))  # noqa: E731
            et, er = p_t - t, p_r - t
            out[c] = (rms(p_t - p_r), rms(et), rms(er),
                      float((w * et * er).sum() / np.sqrt((w * et ** 2).sum() * (w * er ** 2).sum()))
                      if lr > 0 else np.nan)
    return lt, lr, out


def compute(twin, ref, workers):
    rt = open_merged_ranks(MAIN / "results" / twin, 0)
    rr = open_merged_ranks(MAIN / "results" / ref, 0)
    st, sr = rt.store.samples[0], rr.store.samples[0]
    leads_t = sorted(int(k) for k in rt[f"{st}/TEMP"].group_keys() if int(k) > 0)
    leads_r = sorted(int(k) for k in rr[f"{sr}/TEMP"].group_keys() if int(k) > 0)
    init_t = times(rt, 1) - np.timedelta64(1, "D")
    init_r = times(rr, 1) - np.timedelta64(1, "D")
    off = int((init_r - init_t) / np.timedelta64(1, "D"))
    if init_r - init_t != np.timedelta64(off, "D"):
        raise ValueError(f"non-integer day offset {init_t} -> {init_r}")
    print(f"{twin}: init {init_t}, leads {leads_t[0]}-{leads_t[-1]}; {ref}: init {init_r}, "
          f"leads {leads_r[0]}-{leads_r[-1]}; offset {off} d", flush=True)
    pairs = [(lr + off, lr) for lr in [0] + leads_r if lr + off in leads_t]
    for lt, lr in pairs[1:]:
        if times(rt, lt) != times(rr, lr):
            raise ValueError(f"valid time mismatch twin lead {lt} vs ref lead {lr}")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(twin, ref)) as ex:
        res = list(ex.map(_pair, pairs))
    lr = np.array([r[1] for r in res])
    arr = np.array([[r[2][c] for c in CH] for r in res])  # (date, ch, 4)
    ds = xr.Dataset(
        {"diff": (("day", "channel"), arr[..., 0]), "rmse_twin": (("day", "channel"), arr[..., 1]),
         "rmse_ref": (("day", "channel"), arr[..., 2]), "err_corr": (("day", "channel"), arr[..., 3])},
        coords={"day": lr, "channel": list(CH), "twin_lead": ("day", lr + off),
                "valid_time": ("day", np.array([init_r + np.timedelta64(int(x), "D") for x in lr],
                                               "datetime64[ns]"))},
        attrs={"twin": twin, "ref": ref, "twin_init": str(init_t), "ref_init": str(init_r),
               "offset_days": off,
               "definition": "day = reference lead (days after the reference initial time); "
                             "diff = area-weighted RMS of twin - reference prediction over wet "
                             "nodes; day 0: reference = truth"})
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"twin_{twin}_{ref}.nc"
    ds.to_netcdf(out)
    print(f"-> {out} ({len(lr)} dates)", flush=True)


# ── report ───────────────────────────────────────────────────────────────────────────
def loglin(day, y, a, b):
    sel = (day >= a) & (day <= b) & np.isfinite(y) & (y > 0)
    if sel.sum() < 3:
        return np.nan, np.nan, np.nan
    x, ly = day[sel], np.log(y[sel])
    k, c = np.polyfit(x, ly, 1)
    r2 = 1 - ((ly - (k * x + c)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return k, c, r2


def floorfit(day, y, a, b):
    """D = A exp(-(t - a) / tau) + C over days a..b: (tau, C, R2) or None if it fails."""
    from scipy.optimize import curve_fit

    sel = (day >= a) & (day <= b) & np.isfinite(y)
    x, yy = day[sel] - a, y[sel]
    if x.size < 5:
        return None
    try:
        (A, tau, C), _ = curve_fit(lambda t, A, tau, C: A * np.exp(-t / tau) + C, x, yy,
                                   p0=(yy[0] - yy[-1], max(x[-1] / 3, 1.0), yy[-1]),
                                   bounds=([0, 0.1, 0], [np.inf, 1e4, np.inf]), maxfev=20000)
    except RuntimeError:
        return None
    fit = A * np.exp(-x / tau) + C
    return tau, C, 1 - ((yy - fit) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def valid_end(ds):
    """Last day before either run's salt_75 RMSE exceeds VALID_S75."""
    day = ds.day.values
    bad = (ds.rmse_twin.sel(channel="salt_75").values > VALID_S75) | \
          (ds.rmse_ref.sel(channel="salt_75").values > VALID_S75)
    return int(day[np.argmax(bad)] - 1) if bad.any() else int(day.max()), bool(bad.any())


def fits(ds, c):
    """Decay-time fits of the twin - reference difference of channel c.

    main: log-linear from day 1 to the minimum of the difference within the valid window
    (the decaying part); if the minimum is at day <= 5 the difference does not decay and the
    trend over the whole valid window is reported instead. Also fixed windows 1-30, 1-60 and
    the whole valid window (log-linear, any sign).
    """
    day = ds.day.values
    d = ds["diff"].sel(channel=c).values
    end, _ = valid_end(ds)
    v = (day >= 1) & (day <= end)
    imin = int(day[v][np.nanargmin(d[v])])
    r = {"end": end, "dmin_day": imin, "d0": float(d[day == 0][0]), "d1": float(d[day == 1][0]),
         "dmin": float(d[day == imin][0]), "dend": float(d[day == end][0])}
    a, b = 1, imin
    k, _, r2 = loglin(day, d, a, b) if b > 5 else (np.nan, np.nan, np.nan)
    r["main"] = (a, b, -1 / k if k < 0 else np.inf, r2) if b > 5 else None
    r["floor"] = floorfit(day, d, 1, imin) if imin > 5 else None
    for name, (a, b) in (("w20", (1, min(20, end))), ("w30", (1, min(30, end))),
                         ("w60", (1, min(60, end))),
                         ("wall", (1, end))):
        k, _, r2 = loglin(day, d, a, b)
        r[name] = (a, b, k, r2)
    return r


def tau_str(k):
    """Decay time from a log slope k: tau = -1/k (positive = decaying); growth -> e-fold."""
    if not np.isfinite(k):
        return "-"
    return f"tau {-1 / k:.0f} d" if k < 0 else f"grows, e-fold {1 / k:.0f} d"


def report():
    FIG.mkdir(parents=True, exist_ok=True)
    dss = {a: xr.open_dataset(DATA / f"twin_{t}_{r}.nc") for a, (t, r) in PAIRS.items()
           if (DATA / f"twin_{t}_{r}.nc").exists()}
    L = ["# IC-memory twin runs, fvars003, land_cells freeze: tables", "",
         "day = days after 2209-01-01 (reference lead); twin lead = day + offset. "
         "diff = area-weighted RMS(twin - reference); twin / ref = RMSE vs truth; "
         "corr = area-weighted pattern correlation of the two error fields. "
         "ssh in mm, temp in K, salt in psu. Day 0: reference = truth (its initial state).", ""]
    L += ["## Runs", "", "| alpha | twin | twin init | reference | ref init | offset (d) | days "
          "| valid window (both salt_75 RMSE <= 0.3) |", "|---|---|---|---|---|---|---|---|"]
    for a, ds in dss.items():
        end, cut = valid_end(ds)
        L.append(f"| {a} | {ds.attrs['twin']} | {ds.attrs['twin_init']} | {ds.attrs['ref']} | "
                 f"{ds.attrs['ref_init']} | {ds.attrs['offset_days']} | 0-{int(ds.day.max())} "
                 f"| 1-{end}{'' if cut else ' (no crossing)'} |")
    L.append("")
    # decay fits
    L += ["## Decay fits of the twin - reference difference", "",
          "main = log-linear fit from day 1 to the minimum of the difference within the valid "
          "window (the decaying part); 'none' if the minimum is at day <= 5. Fixed windows: "
          "log-linear over days 1-20, 1-30, 1-60 and 1-end of valid window (each capped at the "
          "end of the valid window); tau = -1/slope (decay) or e-fold = 1/slope (growth). "
          "floor fit = A exp(-t/tau) + C over the main window; C/d1 = share of the day-1 "
          "difference that does not decay.", "",
          "| channel | alpha | diff d0 / d1 / min (day) / end of window | main fit: window, tau, R2 "
          "| floor fit: tau, C/d1, R2 | 1-20: tau, R2 | 1-30: tau, R2 | 1-60: tau, R2 "
          "| 1-end: tau, R2 | min / d1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    allfits = {}
    for c in CH:
        sc = 1e3 if c == "ssh" else 1.0
        for a, ds in dss.items():
            f = fits(ds, c)
            allfits[(c, a)] = f
            m = f["main"]
            ms = (f"{m[0]}-{m[1]}, tau {m[2]:.0f} d, {m[3]:.2f}" if m else "none (no decay)")
            fx = " | ".join(f"{tau_str(f[w][2])}, {f[w][3]:.2f}" if np.isfinite(f[w][2]) else "-"
                            for w in ("w20", "w30", "w60", "wall"))
            fl = f["floor"]
            fs = f"{fl[0]:.0f} d, {fl[1] / f['d1']:.2f}, {fl[2]:.2f}" if fl else "-"
            L.append(f"| {c} | {a} | {f['d0'] * sc:.3g} / {f['d1'] * sc:.3g} / "
                     f"{f['dmin'] * sc:.3g} ({f['dmin_day']}) / {f['dend'] * sc:.3g} ({f['end']}) "
                     f"| {ms} | {fs} | {fx} | {f['dmin'] / f['d1']:.2f} |")
    L.append("")
    # onset dates
    L += ["## Onset dates (salt_75 RMSE > 0.5 psu, temp_2 > 2 K), in days after 2209-01-01", "",
          "Taken from the common-date series (day >= 1); '-' = none within the overlap.", "",
          "| alpha | twin salt_75 | ref salt_75 | twin temp_2 | ref temp_2 | twin 0.3 psu | ref 0.3 psu |",
          "|---|---|---|---|---|---|---|"]
    for a, ds in dss.items():
        day = ds.day.values

        def first(v, x):
            i = np.nonzero((v > x) & (day >= 1))[0]
            return str(int(day[i[0]])) if i.size else "-"
        L.append(f"| {a} | {first(ds.rmse_twin.sel(channel='salt_75').values, 0.5)} "
                 f"| {first(ds.rmse_ref.sel(channel='salt_75').values, 0.5)} "
                 f"| {first(ds.rmse_twin.sel(channel='temp_2').values, 2.0)} "
                 f"| {first(ds.rmse_ref.sel(channel='temp_2').values, 2.0)} "
                 f"| {first(ds.rmse_twin.sel(channel='salt_75').values, 0.3)} "
                 f"| {first(ds.rmse_ref.sel(channel='salt_75').values, 0.3)} |")
    L.append("")
    # full series tables
    for c in CH:
        sc = 1e3 if c == "ssh" else 1.0
        L += [f"## {c}: diff | twin RMSE | ref RMSE | error corr, by day", "",
              "| day | " + " | ".join(f"a={a} diff | twin | ref | corr" for a in dss) + " |",
              "|---|" + "---|---|---|---|" * len(dss)]
        for d in TABLE_DAYS:
            cells = []
            for a, ds in dss.items():
                if d in ds.day.values:
                    x = ds.sel(day=d, channel=c)
                    cells += [f"{float(x['diff']) * sc:.3g}", f"{float(x.rmse_twin) * sc:.3g}",
                              f"{float(x.rmse_ref) * sc:.3g}", f"{float(x.err_corr):.2f}"]
                else:
                    cells += ["-"] * 4
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        L.append("")
    txt = "\n".join(L) + "\n"
    (OUT / "tables.md").write_text(txt)
    print(txt)
    figure(dss, allfits)


def figure(dss, allfits):
    fig, axs = plt.subplots(3, len(CH), figsize=(4.2 * len(CH), 10.5), constrained_layout=True)
    for j, c in enumerate(CH):
        sc = 1e3 if c == "ssh" else 1.0
        ax, ax2, ax3 = axs[0, j], axs[1, j], axs[2, j]
        for a, ds in dss.items():
            day = ds.day.values
            end, _ = valid_end(ds)
            d = ds["diff"].sel(channel=c).values * sc
            ax.semilogy(day, d, color=COL[a], lw=1.4, label=f"alpha {a} twin-ref")
            ax.semilogy(day, ds.rmse_ref.sel(channel=c).values * sc, color=COL[a], lw=0.8, ls="--")
            ax.semilogy(day, ds.rmse_twin.sel(channel=c).values * sc, color=COL[a], lw=0.8, ls=":")
            ax.axvline(end, color=COL[a], lw=0.6, alpha=0.5)
            m = allfits[(c, a)]["main"]
            if m:
                k, b0, _ = loglin(day, d, m[0], m[1])
                x = np.arange(m[0], m[1] + 1)
                ax.semilogy(x, np.exp(b0 + k * x), color=COL[a], lw=2.5, alpha=0.35)
            ax2.plot(day, d / (ds.rmse_ref.sel(channel=c).values * sc), color=COL[a], lw=1.2)
            ax3.plot(day, ds.err_corr.sel(channel=c).values, color=COL[a], lw=1.2)
        ax.set_title(f"{c}" + (" (mm)" if c == "ssh" else "")
                     + ": twin-ref (solid), ref RMSE (--), twin RMSE (:)", fontsize=8)
        ax2.set_title(f"{c}: diff / ref RMSE", fontsize=9)
        ax3.set_title(f"{c}: pattern corr of twin and ref errors", fontsize=9)
        ax3.set_ylim(-0.1, 1.02)
        for x in (ax, ax2, ax3):
            x.grid(alpha=0.3)
            x.set_xlim(0, 150)
        ax3.set_xlabel("days after 2209-01-01")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("fvars003, land freeze: twin from 2208-12-01 vs reference from 2209-01-01 "
                 "(vertical lines: end of valid window, salt_75 RMSE <= 0.3; thick: decay fit)")
    fig.savefig(FIG / "twin_diff.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("runs", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "compute":
        compute(a.runs[0], a.runs[1], a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
