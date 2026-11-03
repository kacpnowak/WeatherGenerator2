"""Animated maps of a long free-running rollout: AWI-CM3 truth, emulator and emulator - truth
side by side, one mp4 per field, on the native FESOM mesh.

Reads the rollout's output zip (results/RUN/validation_chkpt00000_rank0000.zip); its points
are in mesh order (nodes for TEMP/SALT/SURFACE, elements for U/V). Dry points of a level are
masked with the mesh's nlevels, triangles across the dateline are dropped. Colour limits are
fixed per field (1st/99th percentile of the truth at three leads; the difference scale from
the last lead), so frames are comparable. The panel titles carry the area-weighted global RMSE.
Sea-ice concentration (POLAR) is drawn on Arctic and Antarctic caps with the 15 % ice edges.

    python scripts/instab/animate_rollout.py fvnn10y [--fields sst sss ssh speed_2 temp_107 a_ice]
        [--stride 5] [--fps 20] [--workers 24] [--out paper_eval/figures/fvnn10y/animations]
"""

import argparse
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402
from zarr.storage import ZipStore  # noqa: E402

MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
STREAM_CFG = {s: f"config/streams/fesom_vars/{s.lower()}.yml" for s in ("TEMP", "SALT", "SURFACE",
                                                                         "U", "V")}
# field -> (streams, channels, title, unit, colormap); two channels = speed of (u, v)
FIELDS = {
    "sst": (("TEMP",), ("temp_2",), "Sea surface temperature", "°C", "RdYlBu_r"),
    "sss": (("SALT",), ("salt_2",), "Sea surface salinity", "psu", "viridis"),
    "ssh": (("SURFACE",), ("ssh",), "Sea surface height", "m", "Spectral_r"),
    "temp_107": (("TEMP",), ("temp_107",), "Temperature, 107 m", "°C", "RdYlBu_r"),
    "speed_2": (("U", "V"), ("u_2", "v_2"), "Surface current speed, 2 m", "m/s", "magma"),
    "a_ice": (("SURFACE",), ("a_ice",), "Sea-ice concentration", "1", "Blues_r"),
}
#: fields drawn on two polar caps (azimuthal equidistant, NH and SH) instead of the global map
POLAR = {"a_ice"}
CAP_DEG = 40.0  # cap radius in degrees of latitude from the pole
ICE_EDGE = 0.15

_G = {}  # per-worker state: zip group, mesh, triangulations


def channels(stream):
    return yaml.safe_load(open(STREAM_CFG[stream]))[stream]["target"]


def init(run):
    _G["z"] = zarr.open_group(
        store=ZipStore(f"results/{run}/validation_chkpt00000_rank0000.zip", mode="r",
                       read_only=True), mode="r")
    m = xr.open_dataset(MESH)
    _G["lon"], _G["lat"] = np.asarray(m.lon), np.asarray(m.lat)
    _G["tri"] = np.asarray(m.face_nodes).T - 1
    _G["nlev_nod"], _G["nlev_elem"] = np.asarray(m.nlevels_nod2D), np.asarray(m.nlevels)
    _G["nod_area"], _G["elem_area"] = np.asarray(m.nod_area), np.asarray(m.elem_area)
    _G["nz1"] = np.asarray(m.nz1)
    lon_t = _G["lon"][_G["tri"]]
    _G["dateline"] = lon_t.max(1) - lon_t.min(1) > 180.0
    _G["cache"] = {}


def geometry(field):
    """(triangulation, wet mask of the points, area weights, on_elements) for a field."""
    if field in _G["cache"]:
        return _G["cache"][field]
    streams, chans, *_ = FIELDS[field]
    layer = int(np.argmin(np.abs(_G["nz1"] - float(chans[0].split("_")[1])))) \
        if "_" in chans[0] else 0
    on_elem = streams[0] in ("U", "V")
    if on_elem:
        wet = _G["nlev_elem"] - 1 > layer
        mask = _G["dateline"] | ~wet
        area = _G["elem_area"]
    else:
        wet = _G["nlev_nod"] - 1 > layer
        mask = _G["dateline"] | ~wet[_G["tri"]].any(1)
        area = _G["nod_area"][layer]
    tri = mtri.Triangulation(_G["lon"], _G["lat"], _G["tri"], mask=mask)
    _G["cache"][field] = (tri, wet, area, on_elem)
    return _G["cache"][field]


def read(field, fstep, kind):
    streams, chans, *_ = FIELDS[field]
    out = []
    for st, ch in zip(streams, chans):
        arr = _G["z"][f"0/{st}/{fstep}/{kind}/data"]
        x = np.asarray(arr.oindex[:, channels(st).index(ch)], dtype=np.float64)
        out.append(x[:, 0] if x.ndim == 2 else x)
    return np.hypot(*out) if len(out) == 2 else out[0]


def valid_date(field, fstep):
    st = FIELDS[field][0][0]
    return str(np.asarray(_G["z"][f"0/{st}/{fstep}/target/times"][0]))[:10]


def polar_geometry(hemi):
    """(triangulation in cap coordinates, nodes in the cap) for 'nh' or 'sh'."""
    key = f"polar_{hemi}"
    if key in _G["cache"]:
        return _G["cache"][key]
    lon, lat = np.deg2rad(_G["lon"]), _G["lat"]
    r = 90.0 - lat if hemi == "nh" else 90.0 + lat
    x = r * np.sin(lon)
    y = -r * np.cos(lon) if hemi == "nh" else r * np.cos(lon)
    cap = r <= CAP_DEG + 2.0
    tri = mtri.Triangulation(x, y, _G["tri"], mask=~cap[_G["tri"]].all(1))
    _G["cache"][key] = (tri, cap)
    return _G["cache"][key]


def draw_polar(args):
    """Sea-ice frame: rows NH / SH, columns truth / emulator / difference, 15 % ice edges."""
    field, idx, fstep, lims, frame_dir = args
    _, _, title, unit, cmap = FIELDS[field]
    truth = np.clip(read(field, fstep, "target"), 0.0, 1.0)
    pred = np.clip(read(field, fstep, "prediction"), 0.0, 1.0)
    area = _G["nod_area"][0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4), constrained_layout=True)
    for row, hemi in zip(axes, ("nh", "sh")):
        tri, cap = polar_geometry(hemi)
        side = _G["lat"] > 0 if hemi == "nh" else _G["lat"] < 0
        ext_t, ext_p = (float((v * area)[side].sum()) * 1e-12 for v in (truth, pred))
        rmse = float(np.sqrt(np.average((pred - truth)[side & cap] ** 2, weights=area[side & cap])))
        panels = ((truth, f"AWI-CM3, ice area {ext_t:.2f}·10⁶ km²", cmap, 0.0, 1.0),
                  (pred, f"Emulator, ice area {ext_p:.2f}·10⁶ km²", cmap, 0.0, 1.0),
                  (pred - truth, f"Emulator − truth (RMSE poleward of {90 - CAP_DEG:.0f}° "
                                 f"{rmse:.3f})", "RdBu_r", -lims[2], lims[2]))
        for col, (ax, (val, name, cm, vmin, vmax)) in enumerate(zip(row, panels)):
            ax.add_patch(plt.Circle((0, 0), CAP_DEG, color="#c9c7c1", zorder=0))  # land
            pc = ax.tripcolor(tri, val, cmap=cm, vmin=vmin, vmax=vmax)
            circle = plt.Circle((0, 0), CAP_DEG, transform=ax.transData)
            pc.set_clip_path(circle)
            for edge, colour in ((truth, "k"), (pred, "#d6336c")):
                if col == 2 or (col == 1 and edge is truth):
                    cs = ax.tricontour(tri, edge, levels=[ICE_EDGE], colors=colour,
                                       linewidths=0.8)
                    cs.set_clip_path(circle)
            ax.add_patch(plt.Circle((0, 0), CAP_DEG, fill=False, lw=0.8, color="#3d3c38"))
            ax.set_xlim(-CAP_DEG, CAP_DEG)
            ax.set_ylim(-CAP_DEG, CAP_DEG)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{'Arctic' if hemi == 'nh' else 'Antarctic'} — {name}", loc="left",
                         fontsize=10)
            if hemi == "sh":
                fig.colorbar(pc, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02,
                             label=unit)
    fig.suptitle(f"{title} — {valid_date(field, fstep)}   (lead {fstep} d, year "
                 f"{fstep / 365.25:.1f} from 2200-01-01; black: AWI-CM3 {ICE_EDGE:.0%} edge, "
                 f"pink: emulator)", x=0.01, ha="left", fontsize=12)
    fig.savefig(frame_dir / f"frame_{idx:05d}.png", dpi=100)
    plt.close(fig)
    return rmse


def limits(field, fsteps):
    """Fixed colour limits: truth 1st/99th percentile at three leads, |diff| 99th at the last."""
    if field in POLAR:
        return 0.0, 1.0, 0.6
    _, wet, *_ = geometry(field)
    sample = [fsteps[0], fsteps[len(fsteps) // 2], fsteps[-1]]
    t = np.concatenate([read(field, f, "target")[wet] for f in sample])
    lo, hi = np.nanpercentile(t, [1, 99])
    if field == "speed_2":
        lo = 0.0
    d = read(field, fsteps[-1], "prediction")[wet] - read(field, fsteps[-1], "target")[wet]
    return float(lo), float(hi), float(np.nanpercentile(np.abs(d), 99))


def draw(args):
    field, idx, fstep, lims, frame_dir = args
    _, _, title, unit, cmap = FIELDS[field]
    tri, wet, area, on_elem = geometry(field)
    truth, pred = read(field, fstep, "target"), read(field, fstep, "prediction")
    diff = pred - truth
    w = area[wet] / area[wet].sum()
    rmse = float(np.sqrt(np.sum(w * diff[wet] ** 2)))
    lo, hi, dlim = lims
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), constrained_layout=True)
    panels = ((truth, "AWI-CM3 (truth)", cmap, lo, hi),
              (pred, "Emulator", cmap, lo, hi),
              (diff, f"Emulator − truth   (global RMSE {rmse:.3g} {unit})", "RdBu_r", -dlim, dlim))
    for ax, (val, name, cm, vmin, vmax) in zip(axes, panels):
        val = np.where(wet, val, 0.0)
        if on_elem:
            pc = ax.tripcolor(tri, facecolors=val, cmap=cm, vmin=vmin, vmax=vmax)
        else:
            pc = ax.tripcolor(tri, val, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_facecolor("#c9c7c1")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-80, 90)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, loc="left", fontsize=11)
        fig.colorbar(pc, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02, label=unit)
    fig.suptitle(f"{title} — {valid_date(field, fstep)}   (lead {fstep} d, year "
                 f"{fstep / 365.25:.1f} of the free-running forecast from 2200-01-01)",
                 x=0.01, ha="left", fontsize=13)
    fig.savefig(frame_dir / f"frame_{idx:05d}.png", dpi=100)
    plt.close(fig)
    return rmse


def encode(frame_dir, out, fps):
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", str(frame_dir / "frame_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fields", nargs="+", default=list(FIELDS))
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"paper_eval/figures/{a.run}/animations")
    out.mkdir(parents=True, exist_ok=True)
    init(a.run)
    last = max(int(k) for k in _G["z"]["0/TEMP"].group_keys())
    fsteps = [1] + list(range(a.stride, last + 1, a.stride))
    with Pool(a.workers, initializer=init, initargs=(a.run,)) as pool:
        for field in a.fields:
            lims = limits(field, fsteps)
            frame_dir = out / f"frames_{field}"
            frame_dir.mkdir(exist_ok=True)
            tasks = [(field, i, f, lims, frame_dir) for i, f in enumerate(fsteps)]
            rmse = pool.map(draw_polar if field in POLAR else draw, tasks, chunksize=4)
            encode(frame_dir, out / f"{field}.mp4", a.fps)
            shutil.copy(frame_dir / f"frame_{len(fsteps) - 1:05d}.png", out / f"{field}_last.png")
            shutil.copy(frame_dir / f"frame_{len(fsteps) // 2:05d}.png", out / f"{field}_mid.png")
            shutil.rmtree(frame_dir)
            print(f"{field}: {len(fsteps)} frames, limits {lims}, RMSE first/last "
                  f"{rmse[0]:.4g}/{rmse[-1]:.4g} -> {out / f'{field}.mp4'}", flush=True)


if __name__ == "__main__":
    main()
