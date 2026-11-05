"""Regional diagnostics for the 2209 issues of the fvars models (deep error from 2209 starts,
the Aug-Sep 2209 step in the ten-year rollout fvnn10y).

    python scripts/instab/regional_diag.py rollout fvnn10y   # daily series from the output zip
    python scripts/instab/regional_diag.py forcing           # IFS forcing, 2190-2209
    python scripts/instab/regional_diag.py truth             # AWI-CM3 ocean, 2010-2209
        [--workers 24] [--out ~/claude_api_tools/y2209/data]

rollout: per forecast step, region and level the area-weighted mean of prediction and truth and
the RMSE for TEMP and SALT (31 levels, wet nodes), and for U/V the vector RMSE and the mean
kinetic energy 0.5 (u^2 + v^2) of prediction and truth (wet elements).
forcing: per sampled 18 UTC snapshot (every 2nd day) the regional mean of the nine forcing
variables the models use. Regular grids are cos(lat) weighted, reduced grids unweighted.
truth: per sampled day (15th of Jan, Apr, Jul, Oct) the regional mean T and S per level.
Regions are lon/lat boxes (REGIONS); they may overlap.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import zarr
from zarr.storage import ZipStore

MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
ARCHIVE = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
FORCING_VARS = ["10u", "10v", "2t", "2d", "msl", "cp", "lsp", "tcc", "tsr"]


def _box(lon, lat, lo0, lo1, la0, la1):
    return (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)


def regions(lon, lat):
    """name -> boolean mask; lon in [-180, 180]."""
    lon = np.where(lon > 180, lon - 360, lon)
    med = _box(lon, lat, -5.5, 36.5, 30, 46) & ~_box(lon, lat, 26, 42, 40.8, 48)
    return {
        "global": np.ones_like(lat, dtype=bool),
        "arctic": lat > 66,
        "n_atlantic": _box(lon, lat, -80, 0, 20, 66) & ~med,
        "mediterranean": med,
        "red_sea": _box(lon, lat, 32, 44, 12, 30),
        "trop_atlantic": _box(lon, lat, -60, 15, -20, 20),
        "s_atlantic": _box(lon, lat, -60, 20, -50, -20),
        "indian": _box(lon, lat, 20, 120, -50, 25) & ~_box(lon, lat, 32, 44, 12, 30),
        "n_pacific": ((lon > 120) | (lon < -100)) & (lat >= 0) & (lat <= 60),
        "s_pacific": ((lon > 150) | (lon < -70)) & (lat >= -50) & (lat < 0),
        "southern": lat < -50,
    }


def _channels(stream):
    return yaml.safe_load(open(f"config/streams/fesom_vars/{stream.lower()}.yml"))[stream]["target"]


_G = {}


def _init_rollout(run):
    _G["z"] = zarr.open_group(store=ZipStore(
        f"results/{run}/validation_chkpt00000_rank0000.zip", mode="r", read_only=True), mode="r")
    m = xr.open_dataset(MESH)
    nz = len(_channels("TEMP"))
    wet_n = np.arange(nz)[:, None] < (np.asarray(m.nlevels_nod2D) - 1)[None, :]
    wet_e = np.arange(nz)[:, None] < (np.asarray(m.nlevels) - 1)[None, :]
    _G["w_n"] = np.asarray(m.nod_area)[:nz] * wet_n  # (lev, nod)
    _G["w_e"] = np.asarray(m.elem_area)[None, :] * wet_e  # (lev, elem)
    _G["reg_n"] = regions(np.asarray(m.lon), np.asarray(m.lat))
    # runs written without U/V (e.g. TEMP,SALT,SURFACE only) skip the velocity statistics
    _G["has_uv"] = "U" in _G["z"]["0"]
    if _G["has_uv"]:
        c_e = np.asarray(_G["z"]["0/U/1/target/coords"][...])
        _G["reg_e"] = regions(c_e[:, 1], c_e[:, 0])


def _wmean(x, w, mask):
    """Area-weighted mean over points of x (pts, lev) with weights w (lev, pts) inside mask."""
    ww = w[:, mask]
    return (x[mask].T * ww).sum(1) / ww.sum(1)


def _rollout_step(fstep):
    g = _G["z"]
    out = {}
    for st in ("TEMP", "SALT"):
        p = np.asarray(g[f"0/{st}/{fstep}/prediction/data"][...], dtype=np.float64)[..., 0]
        t = np.asarray(g[f"0/{st}/{fstep}/target/data"][...], dtype=np.float64)
        p, t = np.nan_to_num(p), np.nan_to_num(t)
        for r, mask in _G["reg_n"].items():
            out[(st, "pred", r)] = _wmean(p, _G["w_n"], mask)
            out[(st, "truth", r)] = _wmean(t, _G["w_n"], mask)
            out[(st, "rmse", r)] = np.sqrt(_wmean((p - t) ** 2, _G["w_n"], mask))
    if not _G["has_uv"]:
        date = np.datetime64(str(np.asarray(g[f"0/TEMP/{fstep}/target/times"][0]))[:10])
        return fstep, date, out
    uv = {}
    for st in ("U", "V"):
        for kind in ("prediction", "target"):
            x = np.asarray(g[f"0/{st}/{fstep}/{kind}/data"][...], dtype=np.float64)
            uv[(st, kind)] = np.nan_to_num(x[..., 0] if x.ndim == 3 else x)
    err2 = (uv["U", "prediction"] - uv["U", "target"]) ** 2 + \
        (uv["V", "prediction"] - uv["V", "target"]) ** 2
    ke_p = 0.5 * (uv["U", "prediction"] ** 2 + uv["V", "prediction"] ** 2)
    ke_t = 0.5 * (uv["U", "target"] ** 2 + uv["V", "target"] ** 2)
    for r, mask in _G["reg_e"].items():
        out[("UV", "rmse", r)] = np.sqrt(_wmean(err2, _G["w_e"], mask))
        out[("KE", "pred", r)] = _wmean(ke_p, _G["w_e"], mask)
        out[("KE", "truth", r)] = _wmean(ke_t, _G["w_e"], mask)
    date = np.datetime64(str(np.asarray(g[f"0/TEMP/{fstep}/target/times"][0]))[:10])
    return fstep, date, out


def rollout(run, workers, out):
    _init_rollout(run)
    # branched / resumed runs start their output after step 1: use the steps present
    steps = sorted(int(k) for k in _G["z"]["0/TEMP"].group_keys() if int(k) >= 1)  # 0 = target-only
    _G["z"].store.close()  # workers open their own handles
    with Pool(workers, initializer=_init_rollout, initargs=(run,)) as pool:
        res = pool.map(_rollout_step, steps, chunksize=8)
    res.sort(key=lambda x: x[0])
    keys = sorted(res[0][2])
    reg = list(_G["reg_n"])
    nlev = len(_channels("TEMP"))
    depth = [float(c.split("_")[1]) for c in _channels("TEMP")]
    ds = xr.Dataset(coords={"lead": [r[0] for r in res], "region": reg, "depth": depth})
    ds = ds.assign_coords(valid_time=("lead", np.array([r[1] for r in res])))
    for st, stat in sorted({(k[0], k[1]) for k in keys}):
        arr = np.stack([[r[2][(st, stat, rr)] for rr in reg] for r in res])  # lead, region, lev
        ds[f"{st}_{stat}"] = (("lead", "region", "depth"), arr[..., :nlev])
    ds.attrs.update(run=run, description=__doc__)
    path = out / f"rollout_regional_{run}.nc"
    ds.to_netcdf(path)
    print("wrote", path, dict(ds.sizes))


def _forcing_snap(args):
    year, idx = args
    z = zarr.open(str(ARCHIVE / "atmos_all" / f"{year}.zarr"), mode="r")
    n = _G["npts"]
    block = np.asarray(z["data"][idx * n:(idx + 1) * n])
    date = np.asarray(z["dates"][idx * n, 0])
    # column order differs between yearly archives: look the variables up by name
    cols = list(z["data"].attrs["colnames"])
    fcols = [cols.index(v) for v in FORCING_VARS]
    lat = block[:, cols.index("lat")]
    assert np.allclose(lat, _G["flat"]), f"forcing grid differs in {year}"
    out = {}
    for r, mask in _G["freg"].items():
        w = _G["fw"][mask]
        out[r] = (block[mask][:, fcols] * w[:, None]).sum(0) / w.sum()
    return date, out


def _init_forcing():
    z = zarr.open(str(ARCHIVE / "atmos_all" / "2209.zarr"), mode="r")
    cols = list(z["data"].attrs["colnames"])
    _G["npts"] = int(z["data"].chunks[0])
    first = np.asarray(z["data"][: _G["npts"]])
    lon, lat = first[:, cols.index("lon")], first[:, cols.index("lat")]
    _, counts = np.unique(np.round(lat, 4), return_counts=True)
    regular = counts.min() == counts.max()
    _G["fw"] = np.cos(np.deg2rad(lat)) if regular else np.ones_like(lat)
    _G["freg"] = {k: v for k, v in regions(lon, lat).items()
                  if k in ("global", "arctic", "n_atlantic", "mediterranean", "southern",
                           "trop_atlantic", "indian", "n_pacific")}
    _G["flat"] = lat
    _G["regular"] = regular


def forcing(workers, out):
    _init_forcing()
    tasks = []
    for year in range(2190, 2210):
        z = zarr.open(str(ARCHIVE / "atmos_all" / f"{year}.zarr"), mode="r")
        nsnap = z["dates"].shape[0] // _G["npts"]
        # 18 UTC snapshot of every 2nd day (snapshots are 00, 06, 12, 18 UTC; the accumulated
        # fields cp, lsp, tsr read zero at 12 UTC)
        dates = np.asarray(z["dates"][: 4 * _G["npts"]: _G["npts"], 0])
        off = int(np.argmax(dates.astype("datetime64[h]").astype(int) % 24 == 18))
        tasks += [(year, i) for i in range(off, nsnap, 8)]
    with Pool(workers, initializer=_init_forcing) as pool:
        res = pool.map(_forcing_snap, tasks, chunksize=4)
    res.sort(key=lambda x: x[0])
    reg = list(_G["freg"])
    ds = xr.Dataset(coords={"time": np.array([r[0] for r in res]), "region": reg,
                            "var": FORCING_VARS})
    ds["mean"] = (("time", "region", "var"), np.stack([[r[1][rr] for rr in reg] for r in res]))
    ds.attrs.update(weights="cos(lat)" if _G["regular"] else "uniform", description=__doc__)
    path = out / "forcing_regional_2190_2209.nc"
    ds.to_netcdf(path)
    print("wrote", path, dict(ds.sizes))


def _init_truth():
    m = xr.open_dataset(MESH)
    nz = len(_channels("TEMP"))
    wet = np.arange(nz)[:, None] < (np.asarray(m.nlevels_nod2D) - 1)[None, :]
    _G["w_n"] = np.asarray(m.nod_area)[:nz] * wet
    _G["reg_n"] = regions(np.asarray(m.lon), np.asarray(m.lat))
    _G["npts"] = len(m.lon)


def _truth_day(args):
    year, day = args
    z = zarr.open(str(ARCHIVE / "ocean_node" / f"ocean_node_{year}.zarr"), mode="r")
    n = _G["npts"]
    block = np.asarray(z["data"][day * n:(day + 1) * n], dtype=np.float64)
    date = np.asarray(z["dates"][day * n, 0])
    # column order differs between yearly archives: look the channels up by name
    names = list(z["data"].attrs["colnames"])
    out = {}
    for st in ("TEMP", "SALT"):
        x = np.nan_to_num(block[:, [names.index(c) for c in _channels(st)]])
        for r, mask in _G["reg_n"].items():
            out[(st, r)] = _wmean(x, _G["w_n"], mask)
    return date, out


def truth(workers, out):
    _init_truth()
    tasks = []
    for year in range(2010, 2210):
        start = np.datetime64(f"{year}-01-01")
        tasks += [(year, int((np.datetime64(f"{year}-{m:02d}-15") - start).astype(int)))
                  for m in (1, 4, 7, 10)]
    with Pool(workers, initializer=_init_truth) as pool:
        res = pool.map(_truth_day, tasks, chunksize=2)
    res.sort(key=lambda x: x[0])
    reg = list(_G["reg_n"])
    depth = [float(c.split("_")[1]) for c in _channels("TEMP")]
    ds = xr.Dataset(coords={"time": np.array([r[0] for r in res]), "region": reg,
                            "depth": depth})
    for st in ("TEMP", "SALT"):
        ds[st] = (("time", "region", "depth"),
                  np.stack([[r[1][(st, rr)] for rr in reg] for r in res]))
    ds.attrs.update(description=__doc__)
    path = out / "truth_regional_2010_2209.nc"
    ds.to_netcdf(path)
    print("wrote", path, dict(ds.sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rollout", "forcing", "truth"])
    ap.add_argument("run", nargs="?", default="fvnn10y")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(Path.home() / "claude_api_tools/y2209/data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "rollout":
        rollout(a.run, a.workers, out)
    elif a.what == "forcing":
        forcing(a.workers, out)
    else:
        truth(a.workers, out)


if __name__ == "__main__":
    main()
