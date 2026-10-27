"""
Calendar-day climatology of a FESOM-format forcing archive (e.g. the AWI-CM3 IFS `atmos_all`).

For every 6-hourly stamp of each target year the output holds the mean, over the reference
years, of the stamp with the same month, day and hour. The result is written in the source
archive's own format (one `<year>.zarr` per target year: a `data` array of shape
(steps * n_points, n_columns) with one chunk per time step and a `dates` array of the same
row count, zarr v2, same dtype / chunks / compressor / fill value), so DataReaderFesom reads
it exactly like the original archive.

    OMP_NUM_THREADS=4 .venv/bin/python scripts/clim_forcing/build_clim_forcing.py \
        --src /work/ab0995/a270088/Kacper/weathergenertor/AWICM3/atmos_all \
        --dst /work/ab0995/a270225/clim_forcing/atmos_all \
        --ref-start 2170 --ref-end 2199 --years 2209 --workers 32

Choices:
- Calendar matching, not day-of-year matching: in a leap reference year March 1 is matched
  to March 1, so the seasonal cycle (and the solar forcing) stays on the calendar.
- Feb 29 of the leap reference years is never used. Leap target years are refused (their
  Feb 29 stamps would have only 1/4 of the reference years).
- Mean over the years in float64, stored as float32; NaNs are skipped (mean over the years
  with a value, NaN where every year is NaN). lon / lat are copied, after checking that every
  reference year has the same grid.
- `data` attrs are copied from the source file of the same year, or from the last reference
  year, and `n_points` is added: DataReaderFesom takes the mesh size from the first file's
  attrs, and in the source archive only the first file (2000.zarr) carries it. colnames,
  means, std and obs_id stay those of the source archive, so the normalisation is unchanged.
- Group attrs record the provenance (the source archive's groups have no attrs).

Nothing is written to `--src`. `--dst` must not contain anything but the year files: the
reader globs every entry of the directory.
"""

import argparse
import calendar
import datetime as dt
import glob
import json
import logging
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numcodecs
import numpy as np
import zarr

_logger = logging.getLogger("build_clim_forcing")

STEP = dt.timedelta(hours=6)

# set in each worker by _init_worker
_W: dict = {}


def year_stamps(year: int) -> list[dt.datetime]:
    """The 6-hourly stamps one year file holds: Jan 1 00 UTC to Dec 31 18 UTC."""
    start = dt.datetime(year, 1, 1)
    n = (dt.datetime(year + 1, 1, 1) - start) // STEP
    return [start + i * STEP for i in range(n)]


def ref_row(year: int, stamp: dt.datetime) -> int:
    """Time-step index, in `year`'s file, of the stamp with `stamp`'s month, day and hour."""
    match = dt.datetime(year, stamp.month, stamp.day, stamp.hour)
    return (match - dt.datetime(year, 1, 1)) // STEP


def _init_worker(src: str, dst: str, ref_years: list[int], n_points: int, perms: dict) -> None:
    numcodecs.blosc.set_nthreads(1)
    _W["src"] = {y: zarr.open_group(f"{src}/{y}.zarr", mode="r") for y in ref_years}
    _W["dst_root"] = dst
    _W["dst"] = {}
    _W["ref_years"] = ref_years
    _W["n"] = n_points
    _W["perms"] = perms


def _process_slot(job: tuple[int, int, int, int]) -> dict:
    """Mean of one target stamp over the reference years, written as one chunk."""
    year, slot, lon_col, lat_col = job
    n = _W["n"]
    stamp = dt.datetime(year, 1, 1) + slot * STEP
    if year not in _W["dst"]:
        _W["dst"][year] = zarr.open_group(f"{_W['dst_root']}/{year}.zarr", mode="r+")
    out = _W["dst"][year]

    total = None
    count = None
    grid = None
    grid_mismatch = 0.0
    for y in _W["ref_years"]:
        row = ref_row(y, stamp)
        g = _W["src"][y]
        first, last = g["dates"][row * n, 0], g["dates"][(row + 1) * n - 1, 0]
        want = np.datetime64(stamp.replace(year=y), "ns")
        if first != want or last != want:
            raise RuntimeError(f"{y}.zarr row {row}: dates {first}..{last}, expected {want}")
        block = g["data"][row * n : (row + 1) * n]
        perm = _W["perms"][y]
        if perm is not None:
            block = block[:, perm]
        block = block.astype(np.float64)
        if grid is None:
            grid = block[:, [lon_col, lat_col]].copy()
            total = np.zeros_like(block)
            count = np.zeros(block.shape, dtype=np.int32)
        else:
            grid_mismatch = max(
                grid_mismatch, float(np.abs(block[:, [lon_col, lat_col]] - grid).max())
            )
        valid = ~np.isnan(block)
        total += np.where(valid, block, 0.0)
        count += valid

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    mean[:, [lon_col, lat_col]] = grid
    out["data"][slot * n : (slot + 1) * n] = mean.astype(np.float32)
    out["dates"][slot * n : (slot + 1) * n] = np.full((n, 1), np.datetime64(stamp, "ns"))
    return {
        "year": year,
        "slot": slot,
        "grid_mismatch": grid_mismatch,
        "partial": int(((count > 0) & (count < len(_W["ref_years"]))).sum()),
        "all_nan": int((count == 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--ref-start", type=int, default=2170)
    p.add_argument("--ref-end", type=int, default=2199)
    p.add_argument("--years", type=int, nargs="+", required=True, help="target years")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--slots", type=int, default=None, help="only the first N stamps of each year (testing)"
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst == src or src in dst.parents:
        raise ValueError(f"--dst {dst} must lie outside --src {src}")
    ref_years = list(range(args.ref_start, args.ref_end + 1))
    for y in args.years:
        if calendar.isleap(y):
            raise NotImplementedError(f"target year {y} is a leap year (no Feb 29 climatology)")

    # the reader globs `<dir>/*` (hidden files such as the top-level .zgroup excluded) and
    # takes the mesh size from the first file
    first_file = sorted(glob.glob(f"{src}/*"))[0]
    n_points = int(zarr.open_group(first_file, mode="r")["data"].attrs["n_points"])
    ref0 = zarr.open_group(src / f"{ref_years[-1]}.zarr", mode="r")
    colnames = list(ref0["data"].attrs["colnames"])
    perms = {}
    for y in ref_years:
        cols = list(zarr.open_group(src / f"{y}.zarr", mode="r")["data"].attrs["colnames"])
        perms[y] = None if cols == colnames else [cols.index(c) for c in colnames]
    lon_col, lat_col = colnames.index("lon"), colnames.index("lat")
    leap_refs = [y for y in ref_years if calendar.isleap(y)]
    _logger.info(
        f"{len(ref_years)} reference years {ref_years[0]}-{ref_years[-1]} "
        f"(leap years, Feb 29 skipped: {leap_refs}); n_points {n_points}; "
        f"{sum(p is not None for p in perms.values())} years with reordered columns"
    )

    dst.mkdir(parents=True, exist_ok=True)
    if (src / ".zgroup").exists():
        shutil.copyfile(src / ".zgroup", dst / ".zgroup")
    jobs = []
    for year in args.years:
        tmpl_path = src / f"{year}.zarr"
        tmpl = zarr.open_group(
            tmpl_path if tmpl_path.exists() else src / f"{ref_years[-1]}.zarr", mode="r"
        )
        tmpl_data, tmpl_dates = tmpl["data"], tmpl["dates"]
        if list(tmpl_data.attrs["colnames"]) != colnames:
            raise ValueError(f"{tmpl_path}: column order differs from {ref_years[-1]}.zarr")
        n_steps = len(year_stamps(year))
        attrs = dict(tmpl_data.attrs)
        attrs["n_points"] = n_points
        g = zarr.open_group(dst / f"{year}.zarr", mode="w", zarr_format=2)
        g.attrs["climatology"] = {
            "source": str(src),
            "reference_years": [ref_years[0], ref_years[-1]],
            "method": "mean over the reference years of the stamp with the same month, day "
            "and hour; Feb 29 of leap reference years unused; NaN-skipping; lon/lat copied",
            "attrs_from": str(tmpl_path if tmpl_path.exists() else src / f"{ref_years[-1]}.zarr"),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
        }
        common = {"chunk_key_encoding": {"name": "v2", "separator": "."}, "order": "C"}
        data = g.create_array(
            "data",
            shape=(n_steps * n_points, len(colnames)),
            chunks=tmpl_data.chunks,
            dtype=tmpl_data.dtype,
            compressors=tmpl_data.compressors,
            filters=tmpl_data.filters or None,
            fill_value=tmpl_data.fill_value,
            **common,
        )
        data.attrs.update(attrs)
        g.create_array(
            "dates",
            shape=(n_steps * n_points, 1),
            chunks=tmpl_dates.chunks,
            dtype=tmpl_dates.dtype,
            compressors=tmpl_dates.compressors,
            filters=tmpl_dates.filters or None,
            fill_value=tmpl_dates.fill_value,
            **common,
        )
        if tmpl_data.chunks != (n_points, len(colnames)):
            raise ValueError(f"expected one chunk per time step, got {tmpl_data.chunks}")
        n_slots = n_steps if args.slots is None else min(args.slots, n_steps)
        jobs += [(year, s, lon_col, lat_col) for s in range(n_slots)]

    t0 = time.time()
    stats = []
    ctx = mp.get_context("fork")
    with ctx.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(src), str(dst), ref_years, n_points, perms),
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_slot, jobs, chunksize=1)):
            stats.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                eta = el / (i + 1) * (len(jobs) - i - 1)
                _logger.info(f"{i + 1}/{len(jobs)} stamps, {el:.0f} s, eta {eta:.0f} s")

    summary = {
        "stamps": len(stats),
        "max_grid_mismatch": max(s["grid_mismatch"] for s in stats),
        "partial_nan_values": sum(s["partial"] for s in stats),
        "all_nan_values": sum(s["all_nan"] for s in stats),
        "seconds": round(time.time() - t0),
    }
    _logger.info(f"done: {json.dumps(summary)}")
    if summary["max_grid_mismatch"] != 0.0:
        raise RuntimeError("lon/lat differ between reference years")


if __name__ == "__main__":
    main()
