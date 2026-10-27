"""
Old-lineage comparison (combined_report section 5 step 2, task 3): drift vs day-to-day
alternation of the global-mean errors of the stable one-year rollout yr1win03 (frcwin03,
FESOM_NODE / FESOM_ELEM layout, levels to 450 m) next to yr1fv002 / yr1fv003.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/oldlineage_series.py \
        compute [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/oldlineage_series.py report

compute  daily global means of prediction and truth and RMSE of yr1win03 (FESOM_NODE), with
         diag_fvars_year's definitions (mesh wet mask, nod_area weights, layer thicknesses,
         rho0, cp, *_0_200 integrals) -> DATA/series_yr1win03.nc
report   ac1 (lag-1 autocorrelation of the daily increments of the global-mean error) and hp
         (RMS of the 2-day high-pass d(L) - (d(L-1) + d(L+1))/2), the same functions as
         analyse_filter.py, per lead window, for yr1win03 and the unfiltered yr1fv002 /
         yr1fv003 series (diag_yr1/part1_daily_*.nc) -> OUT/series_report.txt, FIG/
Comparable quantities: temp_2, salt_75, salt_125, temp_450, ssh_mean, hc_0_200, vmS_0_200
(the old layout ends at 450 m: no 0-2150 m integrals, no sea ice).
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
from paper_eval.config import load_config  # noqa: E402
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402

MAIN = Path("/work/ab0995/a270225/WeatherGenerator2")
OUT = Path("/home/a/a270225/claude_api_tools/oldlineage")
DATA = OUT / "data"
FIG = MAIN / "paper_eval/figures/oldlineage"
REF = Path("/home/a/a270225/claude_api_tools/diag_yr1")
OLD = "yr1win03"
NEW = ("yr1fv002", "yr1fv003")
LEVEL_Q = ("temp_2", "temp_75", "temp_450", "salt_2", "salt_75", "salt_125", "salt_450")
SURF_Q = ("sst", "sss")
Q = ("temp_2", "salt_75", "salt_125", "temp_450", "ssh_mean", "hc_0_200", "vmS_0_200")
RMSE_Q = ("temp_2", "salt_75", "salt_125", "temp_450", "ssh")
WINDOWS = ((2, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 120), (121, 180),
           (181, 270), (271, 363))
PRE_ONSET = {"yr1fv002": (16, 60), "yr1fv003": (16, 90), OLD: (16, 90)}
DAYS = (1, 5, 10, 15, 30, 45, 60, 75, 90, 120, 150, 180, 270, 363)

_W = {}


def _init(cfg_path):
    cfg = load_config(cfg_path)
    _W["root"] = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    _W["s"] = str(_W["root"].store.samples[0])
    _W["geo"] = dfy.Geometry(cfg["mesh_path"])


def _lead(lead):
    root, s, geo = _W["root"], _W["s"], _W["geo"]
    tg = root[f"{s}/FESOM_NODE/{lead}/target"]
    pg = root[f"{s}/FESOM_NODE/{lead}/prediction"]
    ch = list(tg.attrs["channels"])
    assert list(pg.attrs["channels"]) == ch
    t = np.asarray(tg["data"]).astype(np.float64)
    p = np.asarray(pg["data"])[..., 0].astype(np.float64)
    if t.shape[0] != geo.n:
        raise ValueError(f"lead {lead}: {t.shape[0]} points, mesh {geo.n}")
    coord_err = None
    if lead == 1:  # node order check against the mesh (lat, lon columns)
        c = np.asarray(tg["coords"]).astype(np.float64)
        dlon = np.abs(((c[:, 1] - geo.lon) + 180.0) % 360.0 - 180.0)
        coord_err = (float(np.abs(c[:, 0] - geo.lat).max()), float(dlon.max()))
    f = {c: (p[:, j], t[:, j]) for j, c in enumerate(ch)}
    temp = sorted((c for c in f if c.startswith("temp_")), key=geo.level)
    salt = sorted((c for c in f if c.startswith("salt_")), key=geo.level)
    assert [geo.level(c) for c in temp] == list(range(len(temp))) == [geo.level(c) for c in salt]
    nlev = len(temp)
    wet = np.stack([geo.wet(k) for k in range(nlev)])
    w = geo.area[:nlev] * wet
    dz_up = np.clip(np.minimum(geo.nz[1:nlev + 1], dfy.UPPER) - geo.nz[:nlev], 0, None)
    vol_up = (w * dz_up[:, None]).sum()
    a0 = geo.area[0].astype(np.float64)
    pm, tm, rm = [], [], []
    for c in LEVEL_Q:
        k = geo.level(c)
        pp, tt = np.where(wet[k], f[c][0], 0.0), np.where(wet[k], f[c][1], 0.0)
        pm.append((w[k] * pp).sum() / w[k].sum())
        tm.append((w[k] * tt).sum() / w[k].sum())
    for c in SURF_Q + ("ssh",):
        pm.append((a0 * f[c][0]).sum() / a0.sum())
        tm.append((a0 * f[c][1]).sum() / a0.sum())
    for idx, dst in ((0, pm), (1, tm)):
        T = np.stack([np.where(wet[k], f[c][idx], 0.0) for k, c in enumerate(temp)])
        S = np.stack([np.where(wet[k], f[c][idx], 0.0) for k, c in enumerate(salt)])
        dst += [dfy.RHO0 * dfy.CP * (w * dz_up[:, None] * T).sum(),
                (w * dz_up[:, None] * S).sum() / vol_up]
    for c in RMSE_Q + ("sst",):
        k = geo.level(c) if c not in ("ssh", "sst") else None
        ww = w[k] if k is not None else a0
        m = wet[k] if k is not None else np.ones(geo.n, bool)
        e = np.where(m, f[c][0] - f[c][1], 0.0)
        rm.append(np.sqrt((ww * e**2).sum() / ww.sum()))
    return lead, np.array(pm), np.array(tm), np.array(rm), coord_err


def compute(workers):
    cfg_path = WT / f"paper_eval/configs/{OLD}.yaml"
    cfg = load_config(cfg_path)
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    s = str(root.store.samples[0])
    leads = sorted(int(k) for k in root[f"{s}/FESOM_NODE"].group_keys() if int(k) > 0)
    qn = list(LEVEL_Q) + ["sst", "sss", "ssh_mean", "hc_0_200", "vmS_0_200"]
    rn = list(RMSE_Q) + ["sst"]
    P = np.full((len(leads), len(qn)), np.nan)
    T, R = P.copy(), np.full((len(leads), len(rn)), np.nan)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(cfg_path),)) as ex:
        for i, (lead, p, t, r, ce) in enumerate(ex.map(_lead, leads)):
            assert lead == leads[i]
            P[i], T[i], R[i] = p, t, r
            if ce is not None:
                print(f"lead 1 coords vs mesh: max |dlat| {ce[0]:.2e}, |dlon| {ce[1]:.2e}")
    ds = xr.Dataset({"pred": (("lead", "quantity"), P), "truth": (("lead", "quantity"), T),
                     "rmse": (("lead", "channel"), R)},
                    coords={"lead": leads, "quantity": qn, "channel": rn},
                    attrs={"run": OLD, "definitions": dfy.DEFINITIONS})
    DATA.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(DATA / f"series_{OLD}.nc")
    print(f"{OLD}: {len(leads)} leads -> {DATA / f'series_{OLD}.nc'}")


# ── report ───────────────────────────────────────────────────────────────────────────
def load(run):
    path = DATA / f"series_{run}.nc" if run == OLD else REF / f"part1_daily_{run}.nc"
    return xr.open_dataset(path)


def err(ds, q):
    return (ds.pred.sel(quantity=q) - ds.truth.sel(quantity=q)).values


def scale_unit(q):
    if q.startswith("hc_"):
        return 1e-21, "ZJ"
    if q.startswith("ssh"):
        return 1e3, "mm"
    if q.startswith(("temp", "vmT", "sst")):
        return 1.0, "K"
    return 1.0, "psu"


def ac1(d, lead, a, b):  # as analyse_filter.ac1
    sel = (lead >= a) & (lead <= b)
    inc = np.diff(d[sel])
    return np.corrcoef(inc[:-1], inc[1:])[0, 1] if inc.size > 3 else np.nan


def hp_rms(d, lead, a, b):  # as analyse_filter.hp_rms
    hp = np.full_like(d, np.nan)
    hp[1:-1] = d[1:-1] - 0.5 * (d[:-2] + d[2:])
    sel = (lead >= a) & (lead <= b)
    return np.sqrt(np.nanmean(hp[sel] ** 2))


def report():
    dss = {r: load(r) for r in (OLD,) + NEW}
    L = ["Old lineage vs fesom_vars: drift vs day-to-day alternation of the global-mean errors",
         "  runs: yr1win03 (frcwin03 latest, FESOM_NODE/FESOM_ELEM, levels 2-450 m), yr1fv002,",
         "  yr1fv003 (TEMP/SALT/U/V/SURFACE, levels 2-2035 m); all free rollouts from 2209-01-01.",
         "  d = prediction - truth of the wet-area-weighted global mean (diag_fvars_year",
         "  definitions; hc_0_200 = heat content 0-200 m, vmS_0_200 = volume-mean S 0-200 m).",
         "  ac1 = lag-1 autocorrelation of the daily increments d(L)-d(L-1) in the window",
         "  (-0.5: white alternation; -1: pure period-2 alternation; > 0: smooth drift);",
         "  hp = RMS over the window of d(L) - (d(L-1)+d(L+1))/2 (2-day high-pass);",
         "  hpT = the same for the truth itself (natural day-to-day variability).",
         "  Functions as scripts/instab/analyse_filter.py. Series of yr1fv00X from",
         f"  {REF}/part1_daily_*.nc; yr1win03 from {DATA}/series_yr1win03.nc.", ""]
    L.append("RMSE (wet-area weighted) at selected leads (ssh in mm)")
    L.append(f"  {'channel':<10}{'run':<10}" + "".join(f"{d:>8}" for d in DAYS))
    for c in RMSE_Q:
        sc = 1e3 if c == "ssh" else 1.0
        for r, ds in dss.items():
            lead = ds.lead.values
            v = ds.rmse.sel(channel=c).values * sc
            L.append(f"  {c:<10}{r:<10}" + "".join(
                f"{v[lead == d][0]:>8.3g}" if d in lead else f"{'':>8}" for d in DAYS))
    if "sst" in dss[OLD].channel.values:
        ds = dss[OLD]
        v, lead = ds.rmse.sel(channel="sst").values, ds.lead.values
        L.append(f"  {'sst':<10}{OLD:<10}" + "".join(f"{v[lead == d][0]:>8.3g}" for d in DAYS))
    L.append("")
    for name, fn in (("ac1", ac1), ("hp", hp_rms)):
        L.append(f"{name} of the global-mean error per lead window"
                 + (" (display units)" if name == "hp" else ""))
        L.append(f"  {'quantity':<22}{'run':<10}" + "".join(f"{f'{a}-{b}':>9}" for a, b in WINDOWS)
                 + f"{'pre-onset':>11}")
        for q in Q:
            sc, unit = scale_unit(q)
            for r, ds in dss.items():
                lead = ds.lead.values
                d = err(ds, q) * sc
                a, b = PRE_ONSET[r]
                L.append(f"  {q + (f' ({unit})' if name == 'hp' else ''):<22}{r:<10}"
                         + "".join(f"{fn(d, lead, a_, b_):>9.3g}" for a_, b_ in WINDOWS)
                         + f"{fn(d, lead, a, b):>11.3g}")
            if name == "hp":
                ds = dss[OLD]
                t = ds.truth.sel(quantity=q).values * sc
                L.append(f"  {q + ' truth hpT':<22}{OLD:<10}" + "".join(
                    f"{hp_rms(t, ds.lead.values, a_, b_):>9.3g}" for a_, b_ in WINDOWS))
        L.append(f"  pre-onset window: yr1fv002 16-60, yr1fv003 16-90, yr1win03 16-90 (no onset)")
        L.append("")
    # new models: 0-2150 m heat content for reference (not available for the old layout)
    L.append("Reference (fesom_vars only): hc_0_2150 error, ac1 / hp (ZJ) per window")
    for r in NEW:
        ds = dss[r]
        lead = ds.lead.values
        d = err(ds, "hc_0_2150") * 1e-21
        L.append(f"  {r:<10} ac1 " + " ".join(f"{ac1(d, lead, a, b):>6.2f}" for a, b in WINDOWS))
        L.append(f"  {r:<10} hp  " + " ".join(f"{hp_rms(d, lead, a, b):>6.3g}" for a, b in WINDOWS))
    L.append("")
    L.append("Mean error d at selected leads")
    L.append(f"  {'quantity':<22}{'run':<10}" + "".join(f"{d:>9}" for d in DAYS))
    for q in Q:
        sc, unit = scale_unit(q)
        for r, ds in dss.items():
            lead = ds.lead.values
            d = err(ds, q) * sc
            L.append(f"  {q + f' ({unit})':<22}{r:<10}" + "".join(
                f"{d[lead == x][0]:>9.3g}" if x in lead else f"{'':>9}" for x in DAYS))
    OUT.mkdir(parents=True, exist_ok=True)
    (DATA / "series_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure(dss)


def figure(dss):
    FIG.mkdir(parents=True, exist_ok=True)
    qs = ("temp_2", "salt_75", "ssh_mean", "hc_0_200")
    fig, axes = plt.subplots(3, len(qs), figsize=(4.2 * len(qs), 9.5), constrained_layout=True)
    col = {OLD: "k", "yr1fv002": "tab:blue", "yr1fv003": "tab:red"}
    for j, q in enumerate(qs):
        sc, unit = scale_unit(q)
        for r, ds in dss.items():
            lead = ds.lead.values
            d = err(ds, q) * sc
            hp = np.full_like(d, np.nan)
            hp[1:-1] = d[1:-1] - 0.5 * (d[:-2] + d[2:])
            axes[0, j].plot(lead, d, color=col[r], lw=0.8, label=r)
            axes[1, j].plot(lead, hp, color=col[r], lw=0.6, label=r)
            mids = [(a + b) / 2 for a, b in WINDOWS]
            axes[2, j].plot(mids, [ac1(d, lead, a, b) for a, b in WINDOWS], "o-", color=col[r],
                            label=r)
        axes[0, j].set_title(f"{q} mean error ({unit})")
        axes[1, j].set_title(f"{q} 2-day high-pass ({unit})")
        axes[2, j].set_title(f"{q} ac1 of daily increments")
        axes[2, j].axhline(0, color="0.6", lw=0.5)
        axes[0, j].set_xlim(0, 365)
        axes[1, j].set_xlim(0, 365)
        if q == "hc_0_200":
            axes[0, j].set_ylim(-400, 400)
            axes[1, j].set_ylim(-60, 60)
        if q == "salt_75":
            axes[0, j].set_ylim(-0.3, 0.3)
            axes[1, j].set_ylim(-0.03, 0.03)
        for i in range(3):
            axes[i, j].set_xlabel("lead (days)")
    axes[0, 0].legend()
    fig.savefig(FIG / "series_ac1_hp.png", dpi=130)
    print(f"figure -> {FIG / 'series_ac1_hp.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compute", "report"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "compute":
        compute(a.workers)
    else:
        report()


if __name__ == "__main__":
    main()
