"""
Check a climatological forcing archive built by build_clim_forcing.py against the original.

1. zarr metadata of each year file: `.zarray` of `data` and `dates` identical to the original
   year file, `data` attrs identical apart from the added `n_points` (equal to the original
   archive's first file), `dates` identical for every time step.
2. Both archives opened with DataReaderFesom as the IFS_ATMO forcing stream (stream config of
   config/streams/fesom_vars) for some ocean days: same channels, normalisation, coordinates,
   stamps; the climatological values equal an independent mean over the reference years (rows
   located by searching each year's `dates` for the stamp with the same month, day and hour,
   not by index arithmetic).
3. For information: difference between the real and the climatological forcing field the model
   sees on those days (normalised, averaged over the window and onto the HEALPix cells).

    OMP_NUM_THREADS=4 .venv/bin/python scripts/clim_forcing/verify_clim_forcing.py \
        --orig /work/ab0995/a270088/Kacper/weathergenertor/AWICM3/atmos_all \
        --clim /work/ab0995/a270225/clim_forcing/atmos_all --year 2209 \
        --days 2209-01-01 2209-03-01 2209-05-30
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

from weathergen.common.config import load_streams
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.multi_stream_data_sampler import forcing_time_window_handler
from weathergen.model.forcing import forcing_field_from_window
from weathergen.readers_extra.registry import get_extra_reader

REPO = Path(__file__).resolve().parents[2]
HEALPIX_LEVEL = 5
NUM_CELLS = 12 * 4**HEALPIX_LEVEL


def rms(x: NDArray) -> NDArray:
    return np.sqrt((x**2).mean(0))


def check_metadata(orig: Path, clim: Path, year: int) -> dict:
    o = zarr.open_group(orig / f"{year}.zarr", mode="r")
    c = zarr.open_group(clim / f"{year}.zarr", mode="r")
    out = {}
    for name in ("data", "dates"):
        zo = json.loads((orig / f"{year}.zarr" / name / ".zarray").read_text())
        zc = json.loads((clim / f"{year}.zarr" / name / ".zarray").read_text())
        out[f"{name}_zarray_identical"] = zo == zc
    ao, ac = dict(o["data"].attrs), dict(c["data"].attrs)
    first = sorted(glob.glob(f"{orig}/*"))[0]
    n_points = zarr.open_group(first, mode="r")["data"].attrs["n_points"]
    without_n_points = {k: v for k, v in ac.items() if k != "n_points"}
    out["data_attrs_identical_except_n_points"] = without_n_points == ao
    out["n_points"] = ac.get("n_points")
    out["n_points_equals_first_file"] = ac.get("n_points") == n_points
    out["dates_attrs_identical"] = dict(o["dates"].attrs) == dict(c["dates"].attrs)
    # dates: every row, in blocks of 20 time steps
    n = int(n_points)
    steps = o["dates"].shape[0] // n
    same = True
    for s in range(0, steps, 20):
        e = min(s + 20, steps)
        same &= bool(np.array_equal(o["dates"][s * n : e * n], c["dates"][s * n : e * n]))
    out["dates_identical_all_rows"] = same
    out["time_steps"] = steps
    out["first_last_stamp"] = [str(c["dates"][0, 0]), str(c["dates"][(steps - 1) * n, 0])]
    return out


def independent_mean(
    orig: Path, ref_years: list[int], stamps: NDArray, channels: list[str], n: int
) -> NDArray:
    """Mean over `ref_years` of the rows stamped with each of `stamps`' month, day and hour."""
    stamp_md = [(str(s)[5:13]) for s in stamps.astype("datetime64[h]")]  # 'MM-DDTHH'
    total = np.zeros((len(stamps) * n, len(channels)), dtype=np.float64)
    for y in ref_years:
        g = zarr.open_group(orig / f"{y}.zarr", mode="r")
        per_step = g["dates"][::n, 0].astype("datetime64[h]")
        keys = np.array([str(s)[5:13] for s in per_step])
        cols = list(g["data"].attrs["colnames"])
        cidx = [cols.index(ch) for ch in channels]
        for k, md in enumerate(stamp_md):
            hit = np.flatnonzero(keys == md)
            assert len(hit) == 1, (y, md, hit)
            row = int(hit[0])
            assert g["dates"][row * n, 0] == g["dates"][(row + 1) * n - 1, 0]
            block = g["data"][row * n : (row + 1) * n][:, cidx]
            total[k * n : (k + 1) * n] += block
    return total / len(ref_years)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--orig", type=Path, required=True)
    p.add_argument("--clim", type=Path, required=True)
    p.add_argument("--year", type=int, default=2209)
    p.add_argument("--ref-start", type=int, default=2170)
    p.add_argument("--ref-end", type=int, default=2199)
    p.add_argument("--days", nargs="+", default=["2209-01-01", "2209-03-01", "2209-05-30"])
    p.add_argument("--streams", type=Path, default=REPO / "config/streams/fesom_vars")
    p.add_argument("--out", type=Path, default=None, help="write the summary as JSON")
    args = p.parse_args()

    summary = {"metadata": check_metadata(args.orig, args.clim, args.year)}

    info = load_streams(args.streams)["IFS_ATMO"]
    info.setdefault("name", "IFS_ATMO")
    days = TimeWindowHandler(
        np.datetime64(f"{args.year}-01-01T00:00"),
        np.datetime64(f"{args.year + 1}-01-02T00:00"),
        np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    windows = forcing_time_window_handler(days, info)
    cls = get_extra_reader(info["reader"])
    readers = {
        k: cls(tw_handler=windows, filename=path, stream_info=dict(info), stage="test")
        for k, path in (("orig", args.orig), ("clim", args.clim))
    }
    ro, rc = readers["orig"], readers["clim"]
    ro._lazy_init()
    rc._lazy_init()
    summary["reader"] = {
        "source_channels": list(rc.source_channels),
        "channels_identical": list(ro.source_channels) == list(rc.source_channels),
        "mean_identical": bool(np.array_equal(ro.src_mean_vec, rc.src_mean_vec)),
        "std_identical": bool(np.array_equal(ro.src_stdev_vec, rc.src_stdev_vec)),
        "coords_identical": bool(np.array_equal(ro.cached_source_coords, rc.cached_source_coords)),
        "mesh_size": [int(ro.source_mesh_size), int(rc.source_mesh_size)],
        "period": [str(ro.source_period), str(rc.source_period)],
        "stream_id": [ro.properties.get("stream_id"), rc.properties.get("stream_id")],
        "start_end_source": [
            [str(ro.start_source), str(ro.end_source)],
            [str(rc.start_source), str(rc.end_source)],
        ],
    }
    ref_years = list(range(args.ref_start, args.ref_end + 1))
    n = int(rc.source_mesh_size)
    summary["days"] = {}
    for day in args.days:
        jan1 = np.datetime64(f"{args.year}-01-01")
        idx = int((np.datetime64(day) - jan1) // np.timedelta64(1, "D"))
        do, dc = ro.get_source(idx), rc.get_source(idx)
        stamps = np.unique(dc.datetimes)
        ref = independent_mean(args.orig, ref_years, stamps, list(rc.source_channels), n)
        diff = np.abs(dc.data.astype(np.float64) - ref)
        scale = rc.src_stdev_vec.astype(np.float64)
        fo, fc = (
            forcing_field_from_window(
                np.asarray(r.normalize_source_channels(d.data.copy())),
                d.datetimes,
                d.coords,
                None,
                HEALPIX_LEVEL,
                NUM_CELLS,
                "mean",
            )[0]
            for r, d in ((ro, do), (rc, dc))
        )
        nv = len(rc.source_channels)
        fo, fc = np.asarray(fo)[:, :nv], np.asarray(fc)[:, :nv]
        summary["days"][day] = {
            "stamps": [str(s) for s in np.unique(dc.datetimes)],
            "stamps_identical": bool(np.array_equal(do.datetimes, dc.datetimes)),
            "coords_identical": bool(np.array_equal(do.coords, dc.coords)),
            "shape": list(dc.data.shape),
            "max_abs_diff_vs_independent_mean_in_std": dict(
                zip(rc.source_channels, (diff.max(0) / scale).round(9).tolist(), strict=True)
            ),
            "max_rel_diff_vs_independent_mean": float(
                (diff / np.maximum(np.abs(ref), 1e-30))[np.abs(ref) > 1e-6 * scale].max()
            ),
            "clim_differs_from_real": bool(not np.array_equal(do.data, dc.data)),
            "rms_real_minus_clim_field_normalised": dict(
                zip(rc.source_channels, rms(fo - fc).round(4).tolist(), strict=True)
            ),
            "spatial_std_real_field_normalised": dict(
                zip(rc.source_channels, fo.std(0).round(4).tolist(), strict=True)
            ),
        }
    text = json.dumps(summary, indent=1)
    sys.stdout.write(text + "\n")
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
