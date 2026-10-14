"""
Animations of a one-year forced forecast: truth | forecast | forecast - truth, one frame per day.

    .venv/bin/python animate_year.py RUN [--fields sst sss ...] [--fps 15] [--workers 16]

Reads results/<RUN>/validation_chkpt00000_rank*.zip (written by sbatch_fesom_forced_rollout.sh)
and draws the native FESOM mesh with paper_eval's map helpers. Colour limits are fixed over the
whole year (1st-99th percentile of the truth, 5th-95th for salinity; difference limits from the
99th percentile of the absolute error), so changes over time are visible. Writes
paper_eval/figures/anim_<RUN>/<field>.mp4.
"""

import argparse
import os
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))  # paper_eval is a plain folder at the repo root

from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402
from paper_eval.meshtools import load_mesh, masks_and_weights  # noqa: E402
from paper_eval.plotting import style  # noqa: E402
from paper_eval.plotting.fesom_maps import draw_elem_field, draw_node_field  # noqa: E402

warnings.filterwarnings("ignore")
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"

# field -> (stream, channel(s), label, unit, sequential colormap)
# salinity: 5th-95th percentile limits, else marginal seas (Mediterranean ~39-40 psu, Baltic)
# take the whole colour range and the open ocean shows no contrast
PCT = {"sss": (5, 95), "salt_450": (5, 95)}
FIELDS = {
    "sst": ("FESOM_NODE", ("sst",), "SST", "°C", "viridis"),
    "sss": ("FESOM_NODE", ("sss",), "SSS", "psu", "viridis"),
    "ssh": ("FESOM_NODE", ("ssh",), "SSH", "m", "viridis"),
    "speed_2": ("FESOM_ELEM", ("u_2", "v_2"), "Current speed, 2 m", "m/s", "viridis"),
    "temp_107": ("FESOM_NODE", ("temp_107",), "Temperature, 107 m", "°C", "viridis"),
    "salt_450": ("FESOM_NODE", ("salt_450",), "Salinity, 450 m", "psu", "viridis"),
}

_ctx = {}


def _field(root, sample, stream, channels, lead, kind):
    group = root[f"{sample}/{stream}/{lead}/{kind}"]
    names = list(group.attrs["channels"])
    data = np.asarray(group["data"])
    if kind == "prediction":
        data = data[..., 0]
    cols = [data[:, names.index(c)].astype(np.float64) for c in channels]
    return np.sqrt(cols[0] ** 2 + cols[1] ** 2) if len(cols) == 2 else cols[0]


def _init(run, field):
    stream, channels, *_ = FIELDS[field]
    root = open_merged_ranks(REPO / "results" / run, 0)
    sample = root.store.samples[0]
    names = list(root[f"{sample}/{stream}/1/target"].attrs["channels"])
    mask, _ = masks_and_weights(load_mesh(MESH), stream, names)
    _ctx.update(root=root, sample=sample, init=root.store.init_times[sample],
                dry=~mask[names.index(channels[0])])


def _limits(run, field, leads):
    """Fixed colour limits over the year, from a subsample of leads."""
    _init(run, field)
    stream, channels, *_ = FIELDS[field]
    root, sample, dry = _ctx["root"], _ctx["sample"], _ctx["dry"]
    truth, err = [], []
    for lead in leads[:: max(1, len(leads) // 24)]:
        t = _field(root, sample, stream, channels, lead, "target")
        p = _field(root, sample, stream, channels, lead, "prediction")
        truth.append(t[~dry])
        err.append(np.abs(p - t)[~dry])
    lo, hi = np.nanpercentile(np.concatenate(truth), PCT.get(field, (1, 99)))
    d = float(np.nanpercentile(np.concatenate(err), 99))
    return float(lo), float(hi), d


def _frame(args):
    run, field, lead, limits, frame_dir = args
    if not _ctx or _ctx.get("field") != field:
        _init(run, field)
        _ctx["field"] = field
    stream, channels, label, unit, cmap = FIELDS[field]
    root, sample, dry = _ctx["root"], _ctx["sample"], _ctx["dry"]
    truth = _field(root, sample, stream, channels, lead, "target")
    pred = _field(root, sample, stream, channels, lead, "prediction")
    lo, hi, d = limits
    draw = draw_node_field if stream == "FESOM_NODE" else draw_elem_field

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.55), constrained_layout=True)
    panels = (
        ("Truth (FESOM)", truth, cmap, lo, hi, f"{label} [{unit}]"),
        (f"Forecast ({run})", pred, cmap, lo, hi, f"{label} [{unit}]"),
        ("Forecast − truth", pred - truth, style.DIVERGING_CMAP, -d, d, f"[{unit}]"),
    )
    for ax, (title, values, cm, vmin, vmax, cbar) in zip(axes, panels, strict=True):
        draw(ax, MESH, values, dry=dry, cmap=cm, vmin=vmin, vmax=vmax, cbar_label=cbar,
             shrink=0.7)
        ax.set_title(title, loc="left", fontsize=10)
    date = np.datetime64(_ctx["init"], "D") + np.timedelta64(lead, "D")
    fig.suptitle(f"{label}   {date}   (day {lead} of the forecast from "
                 f"{np.datetime64(_ctx['init'], 'D')})", x=0.01, ha="left", fontsize=12)
    path = frame_dir / f"{lead:03d}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("run")
    parser.add_argument("--fields", nargs="+", default=list(FIELDS), choices=list(FIELDS))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--leads", type=int, default=0, help="only the first N leads (test)")
    args = parser.parse_args()

    out = REPO / "paper_eval" / "figures" / f"anim_{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    root = open_merged_ranks(REPO / "results" / args.run, 0)
    sample = root.store.samples[0]
    leads = sorted(int(k) for k in root[f"{sample}/FESOM_NODE"].group_keys() if int(k) > 0)
    if args.leads:
        leads = leads[: args.leads]

    for field in args.fields:
        limits = _limits(args.run, field, leads)
        frame_dir = out / f"frames_{field}"
        frame_dir.mkdir(exist_ok=True)
        jobs = [(args.run, field, lead, limits, frame_dir) for lead in leads]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            frames = list(pool.map(_frame, jobs, chunksize=4))
        video = out / f"{field}.mp4"
        with imageio.get_writer(video, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8) as writer:
            for f in frames:
                writer.append_data(imageio.imread(f))
        shutil.copy(frames[min(len(frames) - 1, 179)], out / f"{field}_still.png")
        shutil.rmtree(frame_dir)
        print(f"{field}: {len(frames)} frames, limits {limits} -> {video}", flush=True)


if __name__ == "__main__":
    main()
