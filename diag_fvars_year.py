"""
Onset diagnostics of the one-year per-variable (fesom_vars) rollouts, e.g. yr1fv002/yr1fv003:
daily global means and integrals (drift vs noise), then where and when the blow-up starts.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python diag_fvars_year.py compute RUN [RUN ...] [--workers N]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python diag_fvars_year.py table   RUN [RUN ...]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python diag_fvars_year.py analyse RUN [RUN ...]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python diag_fvars_year.py osc     RUN [RUN ...]
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python diag_fvars_year.py verify  RUN [RUN ...]

The run is read with paper_eval's reader (`open_merged_ranks`) from the config
paper_eval/configs/<RUN>.yaml (results_dir, mesh_path). Target and prediction of the TEMP,
SALT and SURFACE streams are stored in physical units (only fstep-0 `source` is normalized,
and it is not used here). Wet/dry comes from the mesh only: node i is wet at level k when
k < nlevels_nod2D[i] - 1; values at dry points (NaN in the target, unconstrained in the
prediction) are never used.

compute  one parallel pass over leads 1..363 -> OUT/part1_daily_<RUN>.{nc,csv} (daily global
         means / integrals of prediction and truth, per-channel RMSE) and a cache of
         prediction - truth for a few channels plus the truth a_ice at every lead
         (CACHE/<RUN>_err.npy, CACHE/<RUN>_aice_true.npy) for `analyse`.
table    Part 1 text table, prediction - truth at fixed days, and drift vs day-to-day noise per
         lead window -> OUT/part1_table_<RUN>.txt; FIG/part1_global_diffs.png for all RUNs given
analyse  Part 2: error maps (FIG), hotspot statistics, top-point error shares, onset days and
         early-onset clusters, sign behaviour at the clusters -> OUT/part2_<RUN>.txt; then `osc`
osc      Part 2b: per-node day-to-day (period-2) alternation of the error in the pre-blow-up
         window, binned like the hotspots -> OUT/part2_osc_<RUN>.txt, FIG/<RUN>_osc_maps.png
verify   target == truth archive for a few lead/channel pairs -> OUT/truth_check_<RUN>.txt

Constants and definitions are written into every output (see DEFINITIONS below).
"""

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))  # paper_eval is a plain folder at the repo root

from paper_eval.config import load_config  # noqa: E402
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402
from paper_eval.meshtools import load_mesh, parse_channel  # noqa: E402
from weathergen.readers_extra.fesom_mesh import read_mesh_entity  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path("/home/a/a270225/claude_api_tools/diag_yr1")
FIG = REPO / "paper_eval/figures/yr1fv_diag"
CACHE = REPO / "paper_eval/work/yr1fv_diag"

RHO0 = 1025.0  # kg m-3, reference density
CP = 3992.0  # J kg-1 K-1, specific heat of sea water (TEOS-10 cp0 = 3991.87, rounded)
UPPER = 200.0  # m, depth of the "upper" integrals
STREAMS = ("TEMP", "SALT", "SURFACE")

DAYS_TABLE = (1, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 270, 363)
MAP_DAYS = {"yr1fv003": tuple(range(30, 121, 10)), "yr1fv002": tuple(range(20, 91, 10))}
MAP_CHANNELS = ("salt_75", "salt_125", "temp_75", "temp_2", "ssh")
HOT_CHANNELS = ("salt_75", "salt_125", "temp_2")
#: prediction - truth cached at every lead for `analyse` (NaN at dry points)
CACHE_CHANNELS = ("salt_2", "salt_75", "salt_107", "salt_125", "salt_175",
                  "temp_2", "temp_75", "temp_125", "temp_450", "ssh", "a_ice")

ONSET_FACTOR = 5.0  # onset: first lead with |err| > ONSET_FACTOR * mean(|err|, leads 1..5)
EARLY_QUANTILE = 0.01  # early-onset set: onset day <= this quantile of all wet points' onsets
SHALLOW = 200.0  # m, "shallow column" coastal proxy
DEPTH_BANDS = (0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, np.inf)
LAT_BANDS = tuple(range(-90, 91, 20))

DEFINITIONS = f"""\
Definitions and constants
  wet mask        node i wet at level k iff k < nlevels_nod2D[i]-1 (mesh fesom.mesh.diag.nc); dry values never used
  area weight     level k: nod_area[k, i] (mesh, m^2); 2-D fields (ssh, a_ice, m_ice): nod_area[0]
  global mean     sum_i w_i x_i / sum_i w_i over wet nodes
  layer dz        dz_k = nz[k+1]-nz[k] from the mesh interfaces nz (zbar), full cells (partial bottom
                  cells ignored); the 31 levels (mid-depths 2.5..2035 m) span 0-2150 m
  heat content    HC = rho0*cp*sum_k sum_i w_ki dz_k T_ki  [J, T in degC], rho0 = {RHO0:g} kg m-3,
                  cp = {CP:g} J kg-1 K-1; *_0_2150: all 31 levels; *_0_200: dz clipped at {UPPER:g} m
                  (levels 0-14 full = 0-190 m, plus 10 m of level 15 [190-230 m])
  vol-mean S, T   sum w dz S / sum w dz over the same wet volume
  ice area/volume sum_i a_ice_i * nod_area[0,i], sum_i m_ice_i * nod_area[0,i] (raw; *_clip: a_ice
                  clipped to [0,1], m_ice to >= 0); NH: lat >= 0
  lead L          valid 2209-01-01T23:30 + L days (= truth archive day index L of 2209)
"""


# ── Mesh ──────────────────────────────────────────────────────────────────────────────
class Geometry:
    """Node geometry of the CORE2 mesh: coordinates, per-level areas, wet layers, bottom
    depth, layer interfaces and the node adjacency graph (mesh edges)."""

    def __init__(self, mesh_path, graph=False):
        m = load_mesh(mesh_path)
        self.lon, self.lat, self.tri = m.lon, m.lat, m.tri_elem
        self.area = m.nod_area  # (48, N)
        self.nz1 = m.nz1
        self.n = m.lon.size
        ent = read_mesh_entity(mesh_path, self.n)  # the training reader's own mesh view
        if not np.array_equal(ent.n_wet_layers, m.n_wet_layers_nod):
            raise ValueError("readers_extra and paper_eval disagree on wet layers")
        self.n_wet = m.n_wet_layers_nod
        self.bottom = ent.bottom_depth  # m, positive (-zbar_n_bottom)
        with xr.open_dataset(mesh_path) as ds:
            self.nz = np.abs(ds["nz"].values.astype(np.float64))  # (48,) interfaces, m
        if graph:
            self._build_graph()

    def wet(self, k):
        return self.n_wet > k

    def level(self, channel):
        """Level index of a channel, None for 2-D (surface) fields."""
        return parse_channel(channel, self.nz1)[2]

    def weight(self, channel):
        """(mask, area weight) of a channel's points."""
        k = self.level(channel)
        if k is None:
            return np.ones(self.n, bool), self.area[0].astype(np.float64)
        return self.wet(k), self.area[k].astype(np.float64)

    def _build_graph(self):
        e = np.concatenate([self.tri[:, [0, 1]], self.tri[:, [1, 2]], self.tri[:, [2, 0]]])
        e.sort(axis=1)
        edges, counts = np.unique(e, axis=0, return_counts=True)
        self.boundary = np.zeros(self.n, bool)  # nodes on the mesh's lateral (land) boundary
        self.boundary[edges[counts == 1].ravel()] = True
        a = sparse.coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])),
                              shape=(self.n, self.n))
        self.adj = (a + a.T).tocsr()

    def wall(self, k):
        """Nodes wet at level k next to land at that level: on the lateral mesh boundary, or
        with at least one edge-neighbour that is dry at level k."""
        dry_nb = self.adj @ (~self.wet(k)).astype(np.int32) > 0
        return self.wet(k) & (self.boundary | dry_nb)


def basins(lon, lat):
    """Crude lon/lat-box ocean basins (labels index into BASINS)."""
    lab = np.full(lon.shape, 2)  # Pacific by default
    mid = (lat >= -35) & (lat < 66)
    atl = mid & (
        ((lat < 9) & (lon > -77.2) & (lon < 20))
        | ((lat >= 9) & (lat < 17) & (lon > -84) & (lon < 20))
        | ((lat >= 17) & (lat < 30) & (lon > -100) & (lon < 20))
        | ((lat >= 30) & (lat < 48) & (lon > -100) & (lon < 42))
        | ((lat >= 48) & (lon > -100) & (lon < 32))
    )
    ind = mid & ~atl & (lon >= 20) & (
        ((lat < -20) & (lon < 140)) | ((lat >= -20) & (lat < -5) & (lon < 130))
        | ((lat >= -5) & (lat < 30) & (lon < 100))
    )
    lab[atl] = 0
    lab[ind] = 1
    lab[lat < -35] = 3
    lab[lat >= 66] = 4
    return lab


BASINS = ("Atlantic(+Med,Hudson)", "Indian", "Pacific", "Southern(<35S)", "Arctic(>=66N)")


# ── compute: one pass over the leads ─────────────────────────────────────────────────
_W = {}


def _init_worker(cfg_path):
    cfg = load_config(cfg_path)
    _W["root"] = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    _W["sample"] = _W["root"].store.samples[0]
    _W["geo"] = Geometry(cfg["mesh_path"])


def _read(stream, lead):
    """{channel: (prediction, truth)} float64 for one stream and lead."""
    root, s = _W["root"], _W["sample"]
    tg, pg = root[f"{s}/{stream}/{lead}/target"], root[f"{s}/{stream}/{lead}/prediction"]
    channels = list(tg.attrs["channels"])
    if list(pg.attrs["channels"]) != channels:
        raise ValueError(f"{stream} lead {lead}: target/prediction channels differ")
    t = np.asarray(tg["data"]).astype(np.float64)
    p = np.asarray(pg["data"])
    if p.shape[-1] != 1:
        raise ValueError(f"{stream} lead {lead}: expected one ensemble member, got {p.shape}")
    p = p[..., 0].astype(np.float64)
    if t.shape[0] != _W["geo"].n:
        raise ValueError(f"{stream} lead {lead}: {t.shape[0]} points, mesh has {_W['geo'].n}")
    return {c: (p[:, j], t[:, j]) for j, c in enumerate(channels)}


def quantity_names(level_channels):
    names = list(level_channels)
    names += ["hc_0_2150", "hc_0_200", "vmT_0_2150", "vmT_0_200", "vmS_0_2150", "vmS_0_200",
              "ssh_mean"]
    for kind in ("ice_area", "ice_area_clip", "ice_vol", "ice_vol_clip"):
        names += [f"{kind}_{h}" for h in ("tot", "nh", "sh")]
    return names


def _lead(lead):
    geo = _W["geo"]
    f = {}
    for stream in STREAMS:
        f |= _read(stream, lead)
    temp = sorted((c for c in f if c.startswith("temp_")), key=geo.level)
    salt = sorted((c for c in f if c.startswith("salt_")), key=geo.level)
    if [geo.level(c) for c in temp] != list(range(len(temp))) or len(temp) != len(salt):
        raise ValueError(f"unexpected level set: {temp} / {salt}")
    nlev = len(temp)
    wet = np.stack([geo.wet(k) for k in range(nlev)])  # (nlev, N)
    w = geo.area[:nlev] * wet
    dz = np.diff(geo.nz)[:nlev]
    dz_up = np.clip(np.minimum(geo.nz[1:nlev + 1], UPPER) - geo.nz[:nlev], 0, None)

    out_p, out_t, rmse, nonfinite = [], [], [], 0
    fields = {}
    for name in temp + salt:
        k = geo.level(name)
        p, t = f[name]
        if not np.isfinite(t[wet[k]]).all():
            raise ValueError(f"lead {lead} {name}: non-finite truth at wet points")
        nonfinite += int((~np.isfinite(p[wet[k]])).sum())
        fields[name] = (np.where(wet[k], p, 0.0), np.where(wet[k], t, 0.0))
    for name in temp + salt:  # level means
        k = geo.level(name)
        p, t = fields[name]
        out_p.append((w[k] * p).sum() / w[k].sum())
        out_t.append((w[k] * t).sum() / w[k].sum())
        rmse.append(np.sqrt((w[k] * (p - t) ** 2).sum() / w[k].sum()))

    tp = np.stack([fields[c][0] for c in temp])
    tt = np.stack([fields[c][1] for c in temp])
    sp = np.stack([fields[c][0] for c in salt])
    st = np.stack([fields[c][1] for c in salt])
    vol, vol_up = (w * dz[:, None]).sum(), (w * dz_up[:, None]).sum()
    integ_p, integ_t = [], []
    for T, S, dst in ((tp, sp, integ_p), (tt, st, integ_t)):
        hc = RHO0 * CP * (w * dz[:, None] * T).sum()
        hc_up = RHO0 * CP * (w * dz_up[:, None] * T).sum()
        dst += [hc, hc_up, hc / (RHO0 * CP * vol), hc_up / (RHO0 * CP * vol_up),
                (w * dz[:, None] * S).sum() / vol, (w * dz_up[:, None] * S).sum() / vol_up]

    a0 = geo.area[0].astype(np.float64)
    nh = geo.lat >= 0
    for idx, dst in ((0, integ_p), (1, integ_t)):
        dst.append((a0 * f["ssh"][idx]).sum() / a0.sum())
        ai, mi = f["a_ice"][idx], f["m_ice"][idx]
        for x in (ai, np.clip(ai, 0, 1), mi, np.clip(mi, 0, None)):
            dst += [(a0 * x).sum(), (a0 * x * nh).sum(), (a0 * x * ~nh).sum()]
    for c in ("ssh", "a_ice", "m_ice"):
        p, t = f[c]
        nonfinite += int((~np.isfinite(p)).sum())
        rmse.append(np.sqrt((a0 * (p - t) ** 2).sum() / a0.sum()))

    err = np.empty((len(CACHE_CHANNELS), geo.n), np.float32)
    for j, c in enumerate(CACHE_CHANNELS):
        mask, _ = geo.weight(c)
        p, t = f[c]
        err[j] = np.where(mask, p - t, np.nan)
    return (lead, np.array(out_p + integ_p), np.array(out_t + integ_t), np.array(rmse),
            nonfinite, err, f["a_ice"][1].astype(np.float32), temp + salt)


def compute(run, workers):
    cfg_path = REPO / f"paper_eval/configs/{run}.yaml"
    cfg = load_config(cfg_path)
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    s = root.store.samples[0]
    init = np.datetime64(root[f"{s}/TEMP/1/target/times"][0]) - np.timedelta64(1, "D")
    leads = sorted(int(k) for k in root[f"{s}/TEMP"].group_keys() if int(k) > 0)
    n = Geometry(cfg["mesh_path"]).n
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    err_mm = np.lib.format.open_memmap(CACHE / f"{run}_err.npy", mode="w+", dtype=np.float32,
                                       shape=(len(CACHE_CHANNELS), len(leads), n))
    aice_mm = np.lib.format.open_memmap(CACHE / f"{run}_aice_true.npy", mode="w+",
                                        dtype=np.float32, shape=(len(leads), n))
    pred = truth = rmse = None
    nonfinite = np.zeros(len(leads), np.int64)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cfg_path),)) as ex:
        for i, (lead, p, t, r, nf, err, aice, level_ch) in enumerate(ex.map(_lead, leads)):
            assert lead == leads[i]
            if pred is None:
                qnames = quantity_names(level_ch)
                rnames = level_ch + ["ssh", "a_ice", "m_ice"]
                pred = np.full((len(leads), len(qnames)), np.nan)
                truth, rmse = pred.copy(), np.full((len(leads), len(rnames)), np.nan)
            pred[i], truth[i], rmse[i], nonfinite[i] = p, t, r, nf
            err_mm[:, i] = err
            aice_mm[i] = aice
            if lead % 30 == 0:
                print(f"  {run}: lead {lead}", flush=True)
    err_mm.flush()
    aice_mm.flush()
    units = [("degC" if q.startswith("temp_") else "psu") for q in level_ch]
    units += ["J", "J", "degC", "degC", "psu", "psu", "m"] + ["m2"] * 6 + ["m3"] * 6
    valid = np.array([init + np.timedelta64(L, "D") for L in leads], "datetime64[ns]")
    ds = xr.Dataset(
        {"pred": (("lead", "quantity"), pred), "truth": (("lead", "quantity"), truth),
         "rmse": (("lead", "channel"), rmse),
         "n_nonfinite_pred_wet": (("lead",), nonfinite)},
        coords={"lead": leads, "quantity": qnames, "channel": rnames,
                "units": ("quantity", units), "valid_time": ("lead", valid)},
        attrs={"run": run, "init_time": str(init), "definitions": DEFINITIONS,
               "rmse": "wet-area-weighted RMSE of prediction-truth per channel (units of channel)"},
    )
    ds.to_netcdf(OUT / f"part1_daily_{run}.nc")
    df = {"lead": leads, "valid_date": [str(v)[:10] for v in valid]}
    for j, q in enumerate(qnames):
        df[f"{q}_pred"], df[f"{q}_truth"] = pred[:, j], truth[:, j]
    for j, c in enumerate(rnames):
        df[f"rmse_{c}"] = rmse[:, j]
    import pandas as pd

    pd.DataFrame(df).to_csv(OUT / f"part1_daily_{run}.csv", index=False, float_format="%.9g")
    print(f"{run}: {len(leads)} leads -> {OUT / f'part1_daily_{run}.nc'}, cache {CACHE}")


# ── table: Part 1 text table ──────────────────────────────────────────────────────────
#: display scale and format per quantity kind
def _display(q):
    if q.startswith("temp_"):
        return 1.0, "{:+.3f}", "K"
    if q.startswith("salt_"):
        return 1.0, "{:+.4f}", "psu"
    if q.startswith("hc_"):
        return 1e-21, "{:+.2f}", "ZJ"
    if q.startswith("vmT_"):
        return 1.0, "{:+.4f}", "K"
    if q.startswith("vmS_"):
        return 1.0, "{:+.4f}", "psu"
    if q == "ssh_mean":
        return 1e3, "{:+.1f}", "mm"
    if q.startswith("ice_area"):
        return 1e-12, "{:+.3f}", "1e6km2"
    if q.startswith("ice_vol"):
        return 1e-12, "{:+.3f}", "1e3km3"
    raise ValueError(q)


NOISE_QUANTITIES = ("temp_2", "temp_75", "temp_450", "salt_2", "salt_75", "salt_125", "hc_0_2150",
                    "vmS_0_200", "ssh_mean", "ice_area_tot", "ice_vol_tot")
NOISE_WINDOWS = ((1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 105),
                 (106, 120), (121, 180), (181, 270), (271, 363))


def _drift_noise(ds):
    """Per lead window: mean of d = prediction - truth (drift), RMS of its 2-day high-pass
    hp(L) = d(L) - (d(L-1) + d(L+1))/2 (day-to-day noise / odd-even oscillation) next to the
    truth's own hp RMS, and the lag-1 autocorrelation of the daily increments of d."""
    lead = ds.lead.values
    lines = ["Drift vs day-to-day noise of the global quantities, per lead window",
             "  mean  = window mean of d = prediction - truth (display units)",
             "  hp    = RMS over the window of hp(L) = d(L) - (d(L-1)+d(L+1))/2 (2-day high-pass of d)",
             "  hpT   = the same high-pass RMS of the truth itself (its own day-to-day variability)",
             "  ac1   = lag-1 autocorrelation of the daily increments d(L)-d(L-1) (-0.5: alternating"
             " noise; > 0: smooth drift)", ""]
    head = f"  {'quantity':<28}" + "".join(f"{f'{a}-{b}':>10}" for a, b in NOISE_WINDOWS)
    lines += [head, "  " + "-" * (len(head) - 2)]
    for q in NOISE_QUANTITIES:
        scale, _, unit = _display(q)
        p = ds.pred.sel(quantity=q).values * scale
        t = ds.truth.sel(quantity=q).values * scale
        d = p - t
        hp = np.full_like(d, np.nan)
        hp[1:-1] = d[1:-1] - 0.5 * (d[:-2] + d[2:])
        hpt = np.full_like(t, np.nan)
        hpt[1:-1] = t[1:-1] - 0.5 * (t[:-2] + t[2:])
        rows = {"mean": [], "hp": [], "hpT": [], "ac1": []}
        for a, b in NOISE_WINDOWS:
            sel = (lead >= a) & (lead <= b)
            rows["mean"].append(np.nanmean(d[sel]))
            rows["hp"].append(np.sqrt(np.nanmean(hp[sel] ** 2)))
            rows["hpT"].append(np.sqrt(np.nanmean(hpt[sel] ** 2)))
            inc = np.diff(d[sel])
            rows["ac1"].append(np.corrcoef(inc[:-1], inc[1:])[0, 1] if inc.size > 3 else np.nan)
        for k, vals in rows.items():
            lines.append(f"  {f'{q} {k} ({unit})' if k != 'ac1' else f'{q} ac1':<28}"
                         + "".join(f"{v:>10.3g}" for v in vals))
        lines.append("")
    return lines


def table(run):
    ds = xr.open_dataset(OUT / f"part1_daily_{run}.nc")
    days = [d for d in DAYS_TABLE if d in ds.lead.values]
    lines = [f"Part 1, run {run}: prediction - truth of daily global means/integrals "
             f"(init {ds.attrs['init_time'][:16]})", ""]
    lines += [DEFINITIONS]
    head = f"{'quantity':<15}{'unit':>7}" + "".join(f"{'d' + str(d):>9}" for d in days)
    head += f"{'truth_d1':>11}{'truthD363-1':>12}{'truthRange':>11}"
    lines += [head, "-" * len(head)]
    for q in ds.quantity.values:
        q = str(q)
        if q.startswith("ice_") and q.endswith(("_nh", "_sh")):
            continue  # hemispheric ice split stays in the netCDF/CSV
        scale, fmt, unit = _display(q)
        d = (ds.pred.sel(quantity=q) - ds.truth.sel(quantity=q)) * scale
        tr = ds.truth.sel(quantity=q) * scale
        row = f"{q:<15}{unit:>7}" + "".join(f"{fmt.format(float(d.sel(lead=x))):>9}" for x in days)
        t1 = float(tr.sel(lead=days[0]))
        absfmt = "{:.3f}" if abs(t1) < 1e4 else "{:.4g}"
        row += f"{absfmt.format(t1):>11}{fmt.format(float(tr.sel(lead=days[-1])) - t1):>12}"
        row += f"{fmt.format(float(tr.max() - tr.min())).lstrip('+'):>11}"
        lines.append(row)
    lines += ["", f"non-finite predictions at wet points, all leads: {int(ds.n_nonfinite_pred_wet.sum())}",
              "", "truth_d1 = truth at day 1 (display units); truthD363-1 = truth(day 363) - truth(day 1);",
              "truthRange = max - min of the truth over leads 1..363.", ""]
    lines += _drift_noise(ds)
    txt = "\n".join(lines) + "\n"
    (OUT / f"part1_table_{run}.txt").write_text(txt)
    print(txt)


# ── verify: target == archive ─────────────────────────────────────────────────────────
def verify(run):
    import zarr

    cfg = load_config(REPO / f"paper_eval/configs/{run}.yaml")
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    s = root.store.samples[0]
    geo = Geometry(cfg["mesh_path"])
    lines = [f"truth check, run {run}: target vs archive {cfg['truth']['TEMP']}"]
    for stream, lead, ch in (("SALT", 30, "salt_75"), ("TEMP", 1, "temp_2"),
                             ("TEMP", 120, "temp_125"), ("SURFACE", 200, "ssh")):
        tg = root[f"{s}/{stream}/{lead}/target"]
        vt = np.datetime64(tg["times"][0])
        t = np.asarray(tg["data"])[:, list(tg.attrs["channels"]).index(ch)]
        year = str(vt)[:4]
        g = zarr.open_group(f"{cfg['truth'][stream]}/ocean_node_{year}.zarr", mode="r")
        cols = list(g["data"].attrs["colnames"])
        d0 = np.datetime64(g["dates"][0][0])
        d = int((vt - d0) / np.timedelta64(1, "D"))
        stamp = np.datetime64(g["dates"][d * geo.n][0])
        a = np.asarray(g["data"][d * geo.n:(d + 1) * geo.n, cols.index(ch)])
        mask, _ = geo.weight(ch)
        diff = np.abs(a[mask] - t[mask]).max()
        lines.append(
            f"  {stream:8s} lead {lead:3d} {ch:9s} valid {str(vt)[:16]} archive day {d} "
            f"({str(stamp)[:16]}): max|target-archive| over {mask.sum()} wet nodes = {diff:.3g}; "
            f"target NaN exactly at mesh-dry nodes: {np.array_equal(~np.isfinite(t), ~mask)}")
    txt = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"truth_check_{run}.txt").write_text(txt)
    print(txt)


# ── analyse: Part 2 ──────────────────────────────────────────────────────────────────
def _wstats(e, w, masks):
    """[(rms, SE share %, area share %, n)] per boolean mask; e NaN outside the wet set."""
    ok = np.isfinite(e)
    se = np.where(ok, w * np.nan_to_num(e) ** 2, 0.0)
    wt = np.where(ok, w, 0.0)
    tot_se, tot_w = se.sum(), wt.sum()
    res = []
    for m in masks:
        mm = m & ok
        sw = wt[mm].sum()
        res.append((np.sqrt(se[mm].sum() / sw) if sw > 0 else np.nan,
                    100 * se[mm].sum() / tot_se, 100 * sw / tot_w, int(mm.sum())))
    return res


def _bin_table(title, names, per_day, days):
    """per_day[d] = _wstats output; cells: RMS/SE%/area%."""
    lines = [title, "  cells: area-weighted RMS / % of total area-weighted squared error / % of wet area"]
    head = f"  {'bin':<26}{'n':>7}" + "".join(f"{'day ' + str(d):>19}" for d in days)
    lines.append(head)
    for b, name in enumerate(names):
        row = f"  {name:<26}{per_day[0][b][3]:>7}"
        for st in per_day:
            rms, se, ar, nn = st[b]
            cell = "-" if nn == 0 else f"{rms:.3g}/{se:.1f}/{ar:.1f}"
            row += f"{cell:>19}"
        lines.append(row)
    return lines + [""]


def _top_share(e, w):
    ok = np.isfinite(e)
    e2, ww = e[ok] ** 2, w[ok]
    order = np.argsort(-e2)
    cum = np.cumsum((ww * e2)[order])
    n = e2.size
    return [cum[max(1, int(np.ceil(f * n))) - 1] / cum[-1] * 100 for f in (0.001, 0.01, 0.1)]


def _circmean(lon, w):
    r = np.deg2rad(lon)
    return float(np.rad2deg(np.arctan2((w * np.sin(r)).sum(), (w * np.cos(r)).sum())))


def _onset(abs_e, base, factor):
    """First lead (1-based) with |e| > factor*base, searched from lead 6; 0 = never."""
    exceed = abs_e[5:] > factor * base[None, :]
    first = exceed.argmax(axis=0)
    hit = exceed[first, np.arange(abs_e.shape[1])]
    return np.where(hit, first + 6, 0)


def _sign_stats(e_cl, w_cl, d0, nlead):
    """Sign behaviour of a cluster's error from its first onset day d0 to d0+30."""
    E = (e_cl * w_cl[None, :]).sum(1) / w_cl.sum()  # cluster-mean error per lead
    R = np.sqrt((e_cl ** 2 * w_cl[None, :]).sum(1) / w_cl.sum())
    a, b = d0 - 1, min(d0 + 30, nlead)  # 0-based slice [d0, d0+30]
    win = E[a:b]
    sgn = np.sign(win)
    changes = int((sgn[1:] != sgn[:-1]).sum())
    same_end = float((sgn == sgn[-1]).mean())
    node_sgn = np.sign(e_cl[a:b])
    node_changes = np.median((node_sgn[1:] != node_sgn[:-1]).sum(0))
    node_major = np.median(np.maximum((node_sgn > 0).mean(0), (node_sgn < 0).mean(0)))
    dE = np.diff(win)
    ac1 = float(np.corrcoef(dE[:-1], dE[1:])[0, 1]) if dE.size > 3 and dE.std() > 0 else np.nan
    pick = [d for d in (d0 - 5, d0, d0 + 5, d0 + 10, d0 + 20, d0 + 30) if 1 <= d <= nlead]
    return E, R, changes, b - a - 1, same_end, node_changes, node_major, ac1, pick


def _clusters(onset, valid, geo, w, k, nlead, e, base, label, nshow=12, nsign=5):
    """Early-onset set, its connected components on the mesh graph, sign statistics."""
    lines = []
    ons = np.where(onset > 0, onset, np.inf)[valid]
    d_early = float(np.quantile(ons, EARLY_QUANTILE, method="inverted_cdf"))
    note = f"the {EARLY_QUANTILE:.0%} quantile over wet points"
    if not np.isfinite(d_early):  # fewer than EARLY_QUANTILE ever reach onset: take them all
        if not np.isfinite(ons).any():
            return [f"  [{label}] no wet point ever reaches onset", ""], []
        d_early = float(ons[np.isfinite(ons)].max())
        note = f"all nodes that ever reach onset; fewer than {EARLY_QUANTILE:.0%} do"
    early = valid & (onset > 0) & (onset <= d_early)
    idx = np.flatnonzero(early)
    sub = geo.adj[idx][:, idx]
    ncomp, lab = connected_components(sub, directed=False)
    comps = sorted((idx[lab == c] for c in range(ncomp)), key=len, reverse=True)
    wall = geo.wall(k) if k is not None else geo.boundary
    lines.append(f"  [{label}] early-onset set: onset day <= {d_early:g} ({note}): {early.sum()} "
                 f"nodes in {ncomp} mesh-connected clusters ({sum(len(c) == 1 for c in comps)} "
                 "singletons)")
    lines.append(f"  {'#':>3}{'nodes':>6}{'area km2':>10}{'lon':>8}{'lat':>7}{'lon range':>16}"
                 f"{'lat range':>14}{'bottom m med (min-max)':>24}{'onset min/med':>14}"
                 f"{'nwet-k med':>11}{'wall%':>6}{'base med':>10}{'|e| d30 med':>12}")
    shown = []
    for i, c in enumerate(comps[:nshow]):
        if len(c) < 3:
            break
        ww = geo.area[0 if k is None else k][c]
        lo = geo.lon[c]
        clon = _circmean(lo, ww)
        dlon = (lo - clon + 180) % 360 - 180
        nf = (geo.n_wet[c] - (0 if k is None else k))
        lines.append(
            f"  {i + 1:>3}{len(c):>6}{ww.sum() / 1e6:>10.0f}{clon:>8.1f}"
            f"{(geo.lat[c] * ww).sum() / ww.sum():>7.1f}"
            f"{f'{clon + dlon.min():.1f}..{clon + dlon.max():.1f}':>16}"
            f"{f'{geo.lat[c].min():.1f}..{geo.lat[c].max():.1f}':>14}"
            f"{f'{np.median(geo.bottom[c]):.0f} ({geo.bottom[c].min():.0f}-{geo.bottom[c].max():.0f})':>24}"
            f"{f'{onset[c].min()}/{np.median(onset[c]):.0f}':>14}{np.median(nf):>11.0f}"
            f"{100 * wall[c].mean():>6.0f}{np.median(base[c]):>10.3g}"
            f"{np.median(np.abs(e[29, c])):>12.3g}")
        shown.append(c)
    lines.append("")
    lines.append(f"  [{label}] sign behaviour of the {min(nsign, len(shown))} largest clusters, window "
                 "[first onset d0, d0+30]: E = cluster area-mean error")
    lines.append(f"  {'#':>3}{'d0':>5}{'E signchg':>10}{'/n':>4}{'frac E sign=end':>16}"
                 f"{'node signchg med':>17}{'node majority-sign frac med':>28}{'lag1 ac(dE)':>12}"
                 "   E (RMS) at d0-5, d0, d0+5, d0+10, d0+20, d0+30")
    series = []
    for i, c in enumerate(shown[:nsign]):
        ww = geo.area[0 if k is None else k][c]
        d0 = int(onset[c].min())
        E, R, ch, nt, same, nch, nmaj, ac1, pick = _sign_stats(e[:, c], ww, d0, nlead)
        vals = ", ".join(f"{E[d - 1]:+.3g} ({R[d - 1]:.3g})" for d in pick)
        lines.append(f"  {i + 1:>3}{d0:>5}{ch:>10}{nt:>4}{same:>16.2f}{nch:>17.0f}{nmaj:>28.2f}"
                     f"{ac1:>12.2f}   {vals}")
        series.append((i + 1, E, R))
    return lines + [""], series


def _background_sign(e, valid, nlead):
    """Reference: per-node sign changes of the error over leads 1..31 at all wet points."""
    s = np.sign(e[:31, valid])
    return float(np.median((s[1:] != s[:-1]).sum(0)))


def _map_figure(run, channel, days, fields, geo, mask, norm_kind):
    from paper_eval.plotting.fesom_maps import add_coastline

    vals = np.concatenate([f[mask] for f in fields])
    vals = vals[np.isfinite(vals)]
    vmax = float(np.percentile(np.abs(vals), 99.5))
    if norm_kind == "lin":
        norm = mcolors.Normalize(-vmax, vmax)
        scale_txt = f"linear, +/-{vmax:.3g} (99.5th pct of |err| over all panels)"
    else:
        first = fields[0][mask]
        lin = float(np.nanmedian(np.abs(first)))
        vmax = float(np.percentile(np.abs(vals), 99.9))
        norm = mcolors.SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10)
        scale_txt = (f"symlog, linear within +/-{lin:.2g} (median |err| day {days[0]}), "
                     f"+/-{vmax:.3g} (99.9th pct)")
    ncol = 2
    nrow = int(np.ceil(len(days) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.3 * nrow), constrained_layout=True)
    lon, lat = geo.lon[mask], geo.lat[mask]
    for ax, d, f in zip(axes.flat, days, fields, strict=False):
        v = f[mask]
        o = np.argsort(np.abs(v))  # largest errors drawn on top
        sc = ax.scatter(lon[o], lat[o], c=v[o], s=0.6, lw=0, cmap="RdBu_r", norm=norm,
                        rasterized=True)
        add_coastline(ax)
        ok = np.isfinite(v)
        w = geo.weight(channel)[1][mask]
        rms = np.sqrt((w[ok] * v[ok] ** 2).sum() / w[ok].sum())
        ax.set_title(f"day {d}   RMS {rms:.3g}   max|e| {np.nanmax(np.abs(v)):.3g}", loc="left",
                     fontsize=9)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 90)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(days):]:
        ax.set_visible(False)
    unit = "psu" if channel.startswith("salt") else ("m" if channel == "ssh" else "K")
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01, extend="both")
    cb.set_label(f"{channel} prediction - truth ({unit})")
    fig.suptitle(f"{run}: {channel} prediction - truth, wet mesh nodes; colour scale {scale_txt}",
                 x=0.01, ha="left", fontsize=10)
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{run}_map_{channel}_{norm_kind}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _onset_figure(run, onsets, geo):
    from paper_eval.plotting.fesom_maps import add_coastline

    fig, axes = plt.subplots(len(onsets), 2, figsize=(13, 3.4 * len(onsets)),
                             constrained_layout=True)
    cmap = plt.get_cmap("viridis", 12)
    norm = mcolors.BoundaryNorm([6, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 270, 364], cmap.N)
    for r, (ch, variants) in enumerate(onsets.items()):
        for c, (label, onset, mask) in enumerate(variants):
            ax = axes[r, c]
            never = mask & (onset == 0)
            hit = mask & (onset > 0)
            ax.scatter(geo.lon[never], geo.lat[never], s=0.2, lw=0, c="0.85", rasterized=True)
            o = np.flatnonzero(hit)
            o = o[np.argsort(-onset[o])]  # earliest on top
            sc = ax.scatter(geo.lon[o], geo.lat[o], c=onset[o], s=0.6, lw=0, cmap=cmap, norm=norm,
                            rasterized=True)
            add_coastline(ax)
            ax.set_title(f"{ch}, onset ({label}); grey = never", loc="left", fontsize=9)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-80, 90)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.tick_params(labelsize=7)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("onset lead day (first |err| > 5 x own mean |err| of days 1-5)")
    fig.suptitle(f"{run}: per-node onset day", x=0.01, ha="left", fontsize=10)
    path = FIG / f"{run}_onset_maps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _series_figure(run, series):
    fig, axes = plt.subplots(len(series), 2, figsize=(12, 2.8 * len(series)),
                             constrained_layout=True, squeeze=False)
    cols = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    for r, (ch, variants) in enumerate(series.items()):
        for c, (label, ss) in enumerate(variants):
            ax = axes[r, c]
            for (i, E, R), col in zip(ss, cols, strict=False):
                L = np.arange(1, E.size + 1)
                ax.plot(L, E, color=col, lw=1.2, label=f"cluster {i} mean err")
                ax.plot(L, R, color=col, lw=0.8, ls=":")
            ax.axhline(0, color="0.3", lw=0.6)
            ax.set_xlim(0, 150)
            ax.set_title(f"{ch}, clusters ({label}): mean error (solid), RMS (dotted)", loc="left",
                         fontsize=9)
            ax.set_xlabel("lead day")
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{run}: error at the largest early-onset clusters", x=0.01, ha="left",
                 fontsize=10)
    path = FIG / f"{run}_cluster_series.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def static_bins(geo, ch):
    """{binning: [(name, mask)]} of the wet points of channel `ch` (graph geometry needed)."""
    k = geo.level(ch) or 0
    mask, _ = geo.weight(ch)
    db = [(f"{DEPTH_BANDS[i]:g}-{DEPTH_BANDS[i + 1]:g} m",
           mask & (geo.bottom >= DEPTH_BANDS[i]) & (geo.bottom < DEPTH_BANDS[i + 1]))
          for i in range(len(DEPTH_BANDS) - 1)]
    nf = geo.n_wet - k
    nfb = [("nwet-k = 1", nf == 1), ("nwet-k = 2", nf == 2),
           ("nwet-k = 3-5", (nf >= 3) & (nf <= 5)), ("nwet-k > 5", nf > 5)]
    wall = geo.wall(k)
    shallow = mask & ~wall & (geo.bottom < SHALLOW)
    opn = mask & ~wall & ~shallow
    cb = [("wall", wall), ("shallow", shallow), ("open", opn)]
    cross = [(f"{a} & {b}", ma & mb & mask) for a, ma in nfb for b, mb in cb]
    floor = [(n, m & mask) for n, m in nfb] + cb + cross
    lb = [(f"{LAT_BANDS[i]}..{LAT_BANDS[i + 1]}", mask & (geo.lat >= LAT_BANDS[i])
           & (geo.lat < LAT_BANDS[i + 1] + (LAT_BANDS[i + 1] == 90)))
          for i in range(len(LAT_BANDS) - 1)]
    bl = basins(geo.lon, geo.lat)
    bb = [(BASINS[i], mask & (bl == i)) for i in range(len(BASINS))]
    return {"depth": db, "floor": floor, "lat": lb, "basin": bb}


def ice_bins(mask, a):
    """Truth sea-ice state bins (ice and edge overlap for 0.15 < a <= 0.5)."""
    return [("ice (a>0.15)", mask & (a > 0.15)), ("edge (0.05-0.5)", mask & (a >= 0.05) & (a <= 0.5)),
            ("free (a<0.05)", mask & (a < 0.05))]


def analyse(run):
    cfg = load_config(REPO / f"paper_eval/configs/{run}.yaml")
    geo = Geometry(cfg["mesh_path"], graph=True)
    err = np.load(CACHE / f"{run}_err.npy", mmap_mode="r")  # (nch, nlead, N)
    aice = np.load(CACHE / f"{run}_aice_true.npy", mmap_mode="r")  # (nlead, N)
    nlead = err.shape[1]
    days = MAP_DAYS[run]
    ci = {c: j for j, c in enumerate(CACHE_CHANNELS)}
    figs = []

    for ch in MAP_CHANNELS:
        mask, _ = geo.weight(ch)
        fields = [np.asarray(err[ci[ch], d - 1]) for d in days]
        for kind in ("lin", "symlog"):
            figs.append(_map_figure(run, ch, days, fields, geo, mask, kind))
        print(f"  {run}: maps {ch}", flush=True)

    lines = [f"Part 2, run {run}: where the blow-up starts (prediction - truth)", "",
             DEFINITIONS,
             "Part 2 definitions",
             "  bottom depth   -zbar_n_bottom of the mesh (m)",
             "  near-floor     n_wet_layers - level index (1 = the level is the bottom wet layer)",
             "  coastal proxy  'wall' = node wet at the level and on the mesh lateral boundary (land",
             "                 edge) or with an edge-neighbour node dry at that level; 'shallow' =",
             f"                 not wall and bottom depth < {SHALLOW:g} m; 'open' = neither",
             "  ice state      truth a_ice of that day: ice > 0.15, edge 0.05-0.5, free < 0.05",
             "                 (ice and edge overlap for 0.15-0.5)",
             "  basins         crude lon/lat boxes (Atlantic incl. Med/Hudson, Indian, Pacific,",
             "                 Southern < 35S, Arctic >= 66N)",
             "  top-k% share   % of the total area-weighted squared error carried by the k% of wet",
             "                 points with the largest |error|",
             f"  onset          first lead L >= 6 with |e(L)| > {ONSET_FACTOR:g} x mean(|e(1..5)|) of that",
             "                 node; 'floored': the baseline mean(|e(1..5)|) is raised to at least",
             "                 the median baseline over wet nodes (suppresses near-zero baselines)",
             f"  clusters       nodes with onset <= the {EARLY_QUANTILE:.0%} quantile of onset days over",
             "                 all wet nodes (never = inf), connected through mesh edges",
             "  sign stats     E = cluster area-mean error; 'E signchg/n' = sign changes of E over",
             "                 the n day-to-day steps in [d0, d0+30]; 'node signchg med' = median",
             "                 over the cluster's nodes of their own sign changes in that window;",
             "                 lag1 ac(dE) = lag-1 autocorrelation of the daily increments of E",
             "                 (about -0.5 for alternating noise, near +1 for a smooth drift)",
             ""]
    onset_maps, series_all = {}, {}
    for ch in HOT_CHANNELS:
        k = geo.level(ch)
        mask, w = geo.weight(ch)
        e_all = np.asarray(err[ci[ch]])  # (nlead, N)
        e_days = [e_all[d - 1] for d in days]
        lines += ["=" * 100, f"{run}  {ch}  (level {k}, mid-depth {geo.nz1[k]:g} m, "
                  f"{mask.sum()} wet nodes)", "=" * 100, ""]

        lines.append("Global RMS and top-point shares of the squared error")
        lines.append(f"  {'day':>5}{'RMS':>10}{'top0.1%':>9}{'top1%':>8}{'top10%':>8}{'max|e|':>9}")
        for d, e in zip(days, e_days, strict=True):
            ok = np.isfinite(e)
            rms = np.sqrt((w[ok] * e[ok] ** 2).sum() / w[ok].sum())
            t01, t1, t10 = _top_share(e, w)
            lines.append(f"  {d:>5}{rms:>10.4g}{t01:>8.1f}%{t1:>7.1f}%{t10:>7.1f}%"
                         f"{np.nanmax(np.abs(e)):>9.3g}")
        lines.append("")

        bins = static_bins(geo, ch)
        titles = {"depth": "(a) bottom-depth bands",
                  "floor": "(b) closeness to the sea floor, coastal proxy, and their cross-tab",
                  "lat": "(c) latitude bands (deg)", "basin": "(e) basins (crude boxes)"}
        for key in ("depth", "floor", "lat"):
            lines += _bin_table(titles[key], [n for n, _ in bins[key]],
                                [_wstats(e, w, [m for _, m in bins[key]]) for e in e_days], days)
        ice_rows = []
        for d, e in zip(days, e_days, strict=True):
            ice_rows.append(_wstats(e, w, [m for _, m in ice_bins(mask, np.asarray(aice[d - 1]))]))
        lines += _bin_table("(d) truth sea-ice state of the same day",
                            [n for n, _ in ice_bins(mask, np.zeros(1))], ice_rows, days)
        lines += _bin_table(titles["basin"], [n for n, _ in bins["basin"]],
                            [_wstats(e, w, [m for _, m in bins["basin"]]) for e in e_days], days)

        # onset
        abs_e = np.abs(e_all)
        base = abs_e[:5].mean(0)
        floor = float(np.median(base[mask]))
        on_strict = np.where(mask, _onset(abs_e, base, ONSET_FACTOR), 0)
        base_f = np.maximum(base, floor)
        on_floor = np.where(mask, _onset(abs_e, base_f, ONSET_FACTOR), 0)
        edges = [6, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 151, 181, 271, 364]
        lines.append(f"Onset day histogram (wet nodes; floored baseline = {floor:.3g})")
        lines.append(f"  {'lead days':>10}{'strict n':>10}{'area%':>7}{'cum%':>7}"
                     f"{'floored n':>11}{'area%':>7}{'cum%':>7}")
        cums = [0.0, 0.0]
        for i in range(len(edges) - 1):
            row = f"  {f'{edges[i]}-{edges[i + 1] - 1}':>10}"
            for j, on in enumerate((on_strict, on_floor)):
                sel = mask & (on >= edges[i]) & (on < edges[i + 1])
                ar = 100 * w[sel].sum() / w[mask].sum()
                cums[j] += ar
                row += f"{sel.sum():>{10 if j == 0 else 11}}{ar:>7.2f}{cums[j]:>7.1f}"
            lines.append(row)
        row = f"  {'never':>10}"
        for j, on in enumerate((on_strict, on_floor)):
            sel = mask & (on == 0)
            row += f"{sel.sum():>{10 if j == 0 else 11}}{100 * w[sel].sum() / w[mask].sum():>7.2f}{'':>7}"
        lines += [row, ""]

        lines.append("Early-onset clusters (locations) and their sign behaviour")
        lines.append(f"  background: median per-node sign changes of e over leads 1..31 at all wet "
                     f"nodes = {_background_sign(e_all, mask, nlead):.0f} of 30")
        lc, s_strict = _clusters(on_strict, mask, geo, w, k, nlead, e_all, base, "strict")
        lines += lc
        lc, s_floor = _clusters(on_floor, mask, geo, w, k, nlead, e_all, base, "floored")
        lines += lc
        onset_maps[ch] = [("strict", on_strict, mask), ("floored", on_floor, mask)]
        series_all[ch] = [("strict", s_strict), ("floored", s_floor)]
        print(f"  {run}: stats {ch}", flush=True)

    figs.append(_onset_figure(run, onset_maps, geo))
    figs.append(_series_figure(run, series_all))
    txt = "\n".join(lines) + "\n"
    (OUT / f"part2_{run}.txt").write_text(txt)
    print(f"{run}: -> {OUT / f'part2_{run}.txt'}; figures:\n  " + "\n  ".join(map(str, figs)))


#: pre-blow-up windows in which the global means alternate from day to day (Part 1 ac1 < 0)
HP_WINDOW = {"yr1fv002": (16, 60), "yr1fv003": (16, 90)}
OSC_CHANNELS = ("salt_75", "salt_125", "temp_2", "temp_75", "ssh", "a_ice")


def oscillation(run):
    """Where the day-to-day (period-2) alternation of the error lives, per node, in the
    pre-blow-up window HP_WINDOW[run] -> OUT/part2_osc_<run>.txt, FIG/<run>_osc_maps.png."""
    from paper_eval.plotting.fesom_maps import add_coastline

    cfg = load_config(REPO / f"paper_eval/configs/{run}.yaml")
    geo = Geometry(cfg["mesh_path"], graph=True)
    err = np.load(CACHE / f"{run}_err.npy", mmap_mode="r")
    aice = np.load(CACHE / f"{run}_aice_true.npy", mmap_mode="r")
    ci = {c: j for j, c in enumerate(CACHE_CHANNELS)}
    a, b = HP_WINDOW[run]
    leads = np.arange(a, b + 1)
    alt = (-1.0) ** leads
    lines = [f"Part 2b, run {run}: day-to-day alternation of the error, leads {a}-{b}", "",
             "Per node i, over leads L in the window (e = prediction - truth):",
             "  hp_i(L) = e_i(L) - (e_i(L-1) + e_i(L+1))/2      2-day high-pass of the error",
             "  c_i     = mean_L (-1)^L e_i(L)                   period-2 (odd/even) amplitude",
             "  C       = area-weighted mean of c_i = the period-2 amplitude of the global-mean error",
             "Columns of the binned tables: hpRMS = area-weighted RMS of hp over nodes and leads;",
             "hp% = share of the total hp energy (area-weighted sum of hp^2); area% = share of wet area;",
             "C% = the bin's signed contribution to C (sum_bin w c / sum_all w c, in %); c_mean = the",
             "bin's area-weighted mean c. Ice state: truth a_ice averaged over the window.", ""]
    cmaps = []
    for ch in OSC_CHANNELS:
        mask, w = geo.weight(ch)
        e = np.asarray(err[ci[ch], a - 2:b + 1]).astype(np.float64)  # leads a-1 .. b+1
        hp = e[1:-1] - 0.5 * (e[:-2] + e[2:])
        c = (alt[:, None] * e[1:-1]).mean(0)
        hp2 = (hp ** 2).mean(0)
        e2 = (e[1:-1] ** 2).mean(0)
        ok = mask & np.isfinite(c)
        ww = np.where(ok, w, 0.0)
        C = (ww * np.nan_to_num(c)).sum() / ww.sum()
        rms = lambda x: np.sqrt((ww * np.nan_to_num(x)).sum() / ww.sum())  # noqa: E731
        hp_rms = np.sqrt(hp2)
        t01, t1, t10 = _top_share(np.where(ok, hp_rms, np.nan), w)
        lines += ["=" * 100, f"{run}  {ch}  leads {a}-{b}", "=" * 100,
                  f"  global: C = {C:+.4g}; RMS(c) = {np.sqrt((ww * np.nan_to_num(c) ** 2).sum() / ww.sum()):.4g}; "
                  f"RMS(hp) = {rms(hp2):.4g}; RMS(e) = {rms(e2):.4g}; RMS(hp)/RMS(e) = {rms(hp2) / rms(e2):.3f}",
                  f"  top-point shares of the hp energy: top0.1% {t01:.1f}%, top1% {t1:.1f}%, "
                  f"top10% {t10:.1f}%",
                  f"  median per-node RMS(hp)/RMS(e) = {np.median(np.sqrt(hp2[ok] / e2[ok])):.3f}", ""]
        bins = static_bins(geo, ch)
        bins["ice"] = ice_bins(mask, np.asarray(aice[a - 1:b]).mean(0))
        tot_hp = (ww * np.nan_to_num(hp2)).sum()
        tot_c = (ww * np.nan_to_num(c)).sum()
        for key in ("lat", "basin", "depth", "floor", "ice"):
            lines.append(f"  [{key}] {'bin':<26}{'n':>7}{'hpRMS':>10}{'hp%':>7}{'area%':>7}{'C%':>8}"
                         f"{'c_mean':>11}")
            for name, m in bins[key]:
                m = m & ok
                if not m.any():
                    continue
                wm = ww[m]
                lines.append(f"  {'':<{len(key) + 2}} {name:<26}{m.sum():>7}"
                             f"{np.sqrt((wm * hp2[m]).sum() / wm.sum()):>10.3g}"
                             f"{100 * (wm * hp2[m]).sum() / tot_hp:>7.1f}{100 * wm.sum() / ww.sum():>7.1f}"
                             f"{100 * (wm * c[m]).sum() / tot_c:>8.1f}{(wm * c[m]).sum() / wm.sum():>11.3g}")
            lines.append("")
        cmaps.append((ch, np.where(ok, c, np.nan), mask))
        print(f"  {run}: oscillation {ch}", flush=True)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), constrained_layout=True)
    for ax, (ch, c, mask) in zip(axes.flat, cmaps, strict=True):
        v = c[mask]
        vmax = float(np.nanpercentile(np.abs(v), 99.5))
        o = np.argsort(np.abs(v))
        sc = ax.scatter(geo.lon[mask][o], geo.lat[mask][o], c=v[o], s=0.6, lw=0, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, rasterized=True)
        add_coastline(ax)
        ax.set_title(f"{ch}: period-2 amplitude c = mean (-1)^L e(L), leads {a}-{b}", loc="left",
                     fontsize=9)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 90)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=7)
        fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01, extend="both")
    fig.suptitle(f"{run}: where the error alternates from day to day (own symmetric scale per panel, "
                 "99.5th pct of |c|)", x=0.01, ha="left", fontsize=10)
    path = FIG / f"{run}_osc_maps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    (OUT / f"part2_osc_{run}.txt").write_text(txt)
    print(f"{run}: -> {OUT / f'part2_osc_{run}.txt'}, {path}")


def part1_figure(runs):
    """prediction - truth of the main global quantities vs lead, all runs -> FIG."""
    qs = ("temp_2", "temp_75", "temp_450", "salt_2", "salt_75", "salt_125", "hc_0_2150",
          "vmS_0_200", "ssh_mean", "ice_area_tot", "ice_vol_tot", "vmT_0_2150")
    cols = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    for ax, q in zip(axes.flat, qs, strict=True):
        scale, _, unit = _display(q)
        for run, col in zip(runs, cols, strict=False):
            ds = xr.open_dataset(OUT / f"part1_daily_{run}.nc")
            d = (ds.pred.sel(quantity=q) - ds.truth.sel(quantity=q)) * scale
            ax.plot(ds.lead, d, color=col, lw=0.9, label=run)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_title(f"{q} ({unit})", loc="left", fontsize=9)
        ax.set_xlim(0, 365)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("lead day")
    fig.suptitle("Global means/integrals: prediction - truth, daily", x=0.01, ha="left", fontsize=10)
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "part1_global_diffs.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("compute", "table", "analyse", "osc", "verify"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for run in args.runs:
        if args.action == "compute":
            compute(run, args.workers)
        elif args.action == "table":
            table(run)
        elif args.action == "analyse":
            analyse(run)
            oscillation(run)
        elif args.action == "osc":
            oscillation(run)
        else:
            verify(run)
    if args.action == "table":
        part1_figure(args.runs)


if __name__ == "__main__":
    main()
