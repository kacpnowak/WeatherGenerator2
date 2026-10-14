"""CLI: build a day-of-year climatology for one EERIE gridded-ocean stream (OCEAN_2D or
OCEAN_3D) from its kerchunk-referenced truth archive.

    .venv/bin/python -m paper_eval.build_clim_eerie --config paper_eval/configs/dbuzcpgf.yaml \\
        --stream OCEAN_2D [--ref-start 2012 --ref-end 2013] [--force]

Counterpart of `build_clim.py` (FESOM's native-mesh climatology builder) for EERIE's
fixed 0.25 deg global lat/lon grid (`eerie_grid.EerieGrid`): the truth archive here is a
global gridded field `(time, lev|surface, lat, lon)`, not FESOM's flat per-node/element
zarr, and land (or sub-bathymetry) cells decode to NaN (CF `missing_value`) rather than
being excluded by an externally supplied mesh mask.

Grid convention: the truth store's own longitude axis is phased 0..359.75, but
`eerie_grid.EerieGrid`'s (and the results archive's own point coordinates') convention
is -180..179.75 -- `_read_batch` rolls each day's `(lat, lon)` array by `n_lon // 2`
along its longitude axis before flattening row-major, so the flat cell index of the
result matches `EerieGrid.cell_index`'s `row * n_lon + col` exactly (verified against a
real target point's value: matches the results archive's own physical value bit-for-bit
after the roll, see task-25 report). `_grid_transform` derives (and validates) this once
per store, from the axes themselves, rather than assuming a fixed convention -- OCEAN_2D
and OCEAN_3D turn out to differ (see `_grid_transform`'s docstring).

NaN handling: a cell's land/sub-bathymetry status is overwhelmingly a fixed property of
the ocean model's grid, not a function of time -- but NOT perfectly so: a handful of
open-ocean cells (verified: e.g. OCEAN_2D's `avg_zos`, ~13/366 days in 2012, always a
different 1-2 cells out of 1,038,240 each time) go transiently NaN on an isolated day,
evidently a rare upstream data glitch rather than bathymetry. A cell's WETNESS is
therefore tracked as "finite on at least one processed day" -- a single cheap `(n_cells,
nchan)` running union (`finite_count > 0`), not a per-day comparison against one
reference day (which would be fragile: if that one reference day happened to be a glitch
day for some cell, a perfectly good cell would be wrongly forced to NaN for the whole
output). Every non-finite value, whether permanent land or a transient glitch, is
zero-filled before accumulating (avoiding f64 `NaN + x -> NaN` from ever poisoning a
sum), and the per-doy divisor stays the single FESOM-style scalar `count[doy]` --
"acceptable simplification given ocean cells are [overwhelmingly] finite every day": a
glitch cell's climatology is averaged over a couple fewer real days than the shared
divisor implies, a bias quantified and logged once at the end (`finite_count` also
counts, per cell/channel, how many of the processed days were actually finite) rather
than built into a full per-cell-per-doy divisor -- see the task-25 report's "Design
decisions" section for the measured impact (negligible: at most a handful of cells out
of >1e6, never observed for OCEAN_3D's depth-varying bathymetry channels, which ARE
perfectly time-invariant).

Output layout is IDENTICAL to `climatology.py`'s `build_climatology`/`ClimLookup` (
`clim` (366, n_cells, nchan) f32, `count` (366,) int32, same `channels`/`ref_start`/
`ref_end` attrs) -- `climatology.ClimLookup` reads an EERIE clim zarr with zero code
changes, which is exactly how `scores.py`'s EERIE branch consumes it.
"""

import argparse
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr
import zarr

from paper_eval import eerie_grid
from paper_eval.climatology import doy_index
from paper_eval.config import clim_dir, load_config
from paper_eval.merged_zarr import open_merged_ranks

logger = logging.getLogger(__name__)

#: Consecutive days read from the truth store per batch. Balances xarray/dask
#: per-call overhead (a single-day, single-channel read measured ~7x slower per day
#: than a 30-day batched read, see task-25 report's benchmark) against transient
#: memory (~1.8 GB at this batch size for OCEAN_3D's 14 channels -- negligible next to
#: the (366, n_cells, nchan) f64 accumulator itself).
_DAY_BATCH = 30


def _open_truth(path: str | Path) -> xr.Dataset:
    """Open a kerchunk-parquet-referenced truth store lazily: `fsspec`'s `reference`
    filesystem + `xr.open_zarr(consolidated=False)`, per the brief.

    CF decoding (xarray's default `mask_and_scale`) turns the store's `missing_value`
    fill (land, or -- for OCEAN_3D -- sub-bathymetry at a given depth level) into NaN
    automatically; verified bit-for-bit identical to manually comparing raw values
    against the `missing_value` attr directly (see task-25 report), despite xarray's
    "multiple fill values" warning (the zarr array's own metadata `fill_value: 0.0` is
    a zarr-level default for an unwritten chunk, never actually hit here -- every
    day's chunk is referenced -- so it never spuriously masks a genuine 0.0 physical
    value; also verified directly). Nothing is read yet (dask-backed); only
    `.isel(...).values` in `_read_batch` actually decompresses data.
    """
    fs = fsspec.filesystem("reference", fo=str(path))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multiple fill values.*")
        return xr.open_zarr(fs.get_mapper(""), consolidated=False)


def _channel_lookup(ds: xr.Dataset, channels: list[str]) -> list[tuple[str, str, int]]:
    """`channels` -> `[(channel, var_name, sel_index), ...]` via the store's
    `weathergen_col_map` global attrs (`{var, sel}`).

    `sel`'s dict KEY (e.g. `'lev'`) is not always the variable's own second-dimension
    name -- `avg_von_surf`/`avg_uoe_surf` (OCEAN_3D) are mapped with `sel={'lev': 0}`
    even though their actual array dimension is named `lev_avg_von_surf`/
    `lev_avg_uoe_surf` (verified). Only `sel`'s single INDEX VALUE is trusted; the
    dimension it indexes is always the variable's own `.dims[1]` (the axis right
    after `time`), read from the store directly rather than from `sel`'s key.
    """
    col_map = ds.attrs["weathergen_col_map"]
    out = []
    for ch in channels:
        if ch not in col_map:
            raise ValueError(f"channel {ch!r} missing from the truth store's weathergen_col_map")
        entry = col_map[ch]
        var_name = entry["var"]
        if var_name not in ds.data_vars:
            raise ValueError(f"channel {ch!r}: var {var_name!r} not found in the truth store")
        sel_idx = next(iter(entry["sel"].values()))
        out.append((ch, var_name, int(sel_idx)))
    return out


def _grid_transform(ds: xr.Dataset, grid: eerie_grid.EerieGrid) -> tuple[bool, int]:
    """Derive (and validate) how to transform this truth store's raw `(lat, lon)`
    row-major data axes into `eerie_grid.EerieGrid`'s canonical ascending-lat,
    -180..179.75-lon convention, from the store's own `lat`/`lon` node coordinate
    arrays. `_read_batch` applies the result to every batch it reads before
    flattening, so the resulting flat cell index matches `EerieGrid.cell_index`'s
    `row * n_lon + col` exactly.

    OCEAN_2D and OCEAN_3D are the same physical 0.25 deg earth grid, but their raw
    storage differs: verified empirically (a real target point, see task-25 report)
    that OCEAN_2D's data rows are lat-ASCENDING (-90..90, matching `EerieGrid.lat`
    directly, `lat_flip=False`) while OCEAN_3D's are lat-DESCENDING (90..-90,
    `lat_flip=True`) -- an earlier version of this builder assumed OCEAN_2D's
    orientation unconditionally and got a wrong-hemisphere NaN for every OCEAN_3D
    cell, caught by this function raising before any data was actually processed.
    Longitude is a simple cyclic roll (raw 0..359.75 -> -180..179.75) for both, but
    the roll amount is likewise derived here, not assumed to be `n_lon // 2`.

    Returns `(lat_flip, lon_roll)`: `_read_batch` applies `lat_flip` (row `r` ->
    `n_lat - 1 - r`) BEFORE `np.roll`-ing by `lon_roll` along the longitude axis.
    """
    n_lat, n_lon = grid.lat.size, grid.lon.size
    lat_nodes = np.asarray(ds["lat"].values, dtype=np.float64).reshape(n_lat, n_lon)
    lon_nodes = np.asarray(ds["lon"].values, dtype=np.float64).reshape(n_lat, n_lon)
    if not np.allclose(lat_nodes, lat_nodes[:, :1]) or not np.allclose(lon_nodes, lon_nodes[:1, :]):
        raise ValueError(
            "truth store lat/lon node arrays are not a regular (lat, lon) grid "
            "(lat must be constant per row, lon constant per column)"
        )

    lat_col0 = lat_nodes[:, 0]
    lat_flip = bool(lat_col0[0] > lat_col0[-1])
    lat_check = lat_col0[::-1] if lat_flip else lat_col0
    if not np.allclose(lat_check, grid.lat):
        raise ValueError(
            "truth store lat axis (tried both ascending and descending) does not "
            "match eerie_grid.EerieGrid.lat -- row-major flattening would land on "
            "the wrong cell index"
        )

    lon_row0 = lon_nodes[0, :]
    centered = np.where(lon_row0 >= 180.0, lon_row0 - 360.0, lon_row0)
    lon_roll = (n_lon - int(np.argmin(centered))) % n_lon
    lon_check = np.roll(centered, lon_roll)
    if not np.allclose(lon_check, grid.lon, atol=1e-6):
        raise ValueError(
            "truth store lon axis (rolled into -180..179.75) does not match "
            "eerie_grid.EerieGrid.lon -- row-major flattening would land on the "
            "wrong cell index"
        )
    return lat_flip, lon_roll


def _read_batch(
    ds: xr.Dataset,
    lookup: list[tuple[str, str, int]],
    day0: int,
    day1: int,
    n_lat: int,
    n_lon: int,
    lat_flip: bool,
    lon_roll: int,
) -> np.ndarray:
    """Read truth-store days `[day0, day1)` for every `(channel, var, sel_idx)` in
    `lookup`, returning `(day1 - day0, n_lat * n_lon, nchan)` float32, row-major
    `(lat, lon)`-flattened after applying `_grid_transform`'s `(lat_flip, lon_roll)`
    so the flat cell order matches `EerieGrid.cell_index`'s convention.

    Channels sharing one underlying `var` (e.g. OCEAN_3D's `avg_uoe` at three depth
    levels) are read together in one `.isel(...)` call -- one dask/xarray call per
    distinct `var`, not per channel, since chunks are per `(day, level)` regardless
    (reading levels together saves per-call overhead, not I/O).
    """
    nb = day1 - day0
    nchan = len(lookup)
    n_cells = n_lat * n_lon
    block = np.empty((nb, n_cells, nchan), dtype=np.float32)

    by_var: dict[str, list[tuple[int, int]]] = {}
    for pos, (_ch, var_name, sel_idx) in enumerate(lookup):
        by_var.setdefault(var_name, []).append((pos, sel_idx))

    for var_name, entries in by_var.items():
        dim1 = ds[var_name].dims[1]
        sel_idxs = [s for _, s in entries]
        raw = ds[var_name].isel(time=slice(day0, day1), **{dim1: sel_idxs}).values
        # raw: (nb, len(entries), n_lat, n_lon) -- axis 1 order follows sel_idxs, i.e.
        # entries' order, so raw[:, k] <-> entries[k].
        for k, (pos, _sel_idx) in enumerate(entries):
            field = raw[:, k]  # (nb, n_lat, n_lon)
            if lat_flip:
                field = field[:, ::-1, :]
            field = np.roll(field, lon_roll, axis=2)  # lon 0..359.75 -> -180..179.75
            block[:, :, pos] = field.reshape(nb, n_cells)

    return block


def build_climatology_eerie(
    truth_path: str | Path,
    channels: list[str],
    grid: eerie_grid.EerieGrid,
    ref_start: int,
    ref_end: int,
    out_path: str | Path,
) -> None:
    """Build a day-of-year climatology over years `[ref_start, ref_end]` on the fixed
    EERIE grid and write it to `out_path` as a zarr group -- `climatology.py`'s
    `build_climatology`/`ClimLookup` layout (`clim` (366, n_cells, nchan) f32,
    `count` (366,) int32), see this module's docstring.

    The float64 accumulator spans the full 366-day climatology at grid resolution for
    the whole reference period (~9 GB for OCEAN_2D's 3 channels, ~43 GB for OCEAN_3D's
    14 -- expected and fine within the sbatch job's memory budget; do not shrink it
    with float32 accumulation, per `climatology.build_climatology`'s same rule).
    """
    truth_path = Path(truth_path)
    out_path = Path(out_path)
    channels = list(channels)
    nchan = len(channels)
    n_cells = grid.n_cells
    n_lat, n_lon = grid.lat.size, grid.lon.size

    ds = _open_truth(truth_path)
    lat_flip, lon_roll = _grid_transform(ds, grid)
    lookup = _channel_lookup(ds, channels)

    dates = np.asarray(ds["time"].values).astype("datetime64[D]")

    accum_sum = np.zeros((366, n_cells, nchan), dtype=np.float64)
    count = np.zeros(366, dtype=np.int64)
    #: Total days (across the whole reference period) each (cell, channel) was finite
    #: -- a cell/channel is "wet" iff this is ever > 0 (see module docstring). Cheap:
    #: one (n_cells, nchan) int64 array, NOT the (366, n_cells) per-channel array the
    #: brief flags as a memory concern.
    finite_count = np.zeros((n_cells, nchan), dtype=np.int64)
    n_days_total = 0

    for year in range(ref_start, ref_end + 1):
        t_year0 = time.perf_counter()
        y0 = np.datetime64(f"{year}-01-01")
        y1 = np.datetime64(f"{year + 1}-01-01")
        idx = np.nonzero((dates >= y0) & (dates < y1))[0]
        if idx.size == 0:
            raise ValueError(f"{truth_path}: no data for year {year} in the truth store")
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + idx.size)):
            raise ValueError(
                f"{truth_path}: year {year}'s days are not a contiguous run in the "
                "store's time axis -- ragged/gapped year?"
            )
        day0, n_days = int(idx[0]), int(idx.size)
        doys = doy_index(dates[day0 : day0 + n_days])

        for b0 in range(0, n_days, _DAY_BATCH):
            b1 = min(b0 + _DAY_BATCH, n_days)
            block = _read_batch(
                ds, lookup, day0 + b0, day0 + b1, n_lat, n_lon, lat_flip, lon_roll
            )

            finite = np.isfinite(block)  # (nb, n_cells, nchan)
            finite_count += finite.sum(axis=0)
            # Zero-fill EVERY non-finite value (permanent land/sub-bathymetry AND any
            # transient glitch alike) before accumulating -- f64 `NaN + x -> NaN`
            # would otherwise permanently poison that (doy, cell, channel) sum from a
            # single bad day for the rest of the whole reference period.
            block[~finite] = 0.0
            batch_doys = doys[b0:b1] - 1
            if np.unique(batch_doys).size != batch_doys.size:
                raise ValueError(f"year {year} batch [{b0}:{b1}]: duplicate day-of-year index")
            accum_sum[batch_doys] += block.astype(np.float64)
            count[batch_doys] += 1

        n_days_total += n_days
        elapsed = time.perf_counter() - t_year0
        logger.info("year %d: %d days in %.1fs", year, n_days, elapsed)

    if n_days_total == 0:
        raise ValueError(f"ref_start={ref_start} > ref_end={ref_end}: no years to process")

    wet_mask = finite_count > 0  # (n_cells, nchan) -- "ever finite" = physically wet
    partial = wet_mask & (finite_count < n_days_total)
    n_partial = int(partial.sum())
    if n_partial:
        min_frac = float(finite_count[partial].min()) / n_days_total
        logger.warning(
            "%d (cell, channel) pair(s) out of %d wet were finite on only SOME (not "
            "all %d) processed days (min finite fraction %.6f) -- an 'acceptable "
            "simplification' per the brief: the per-doy divisor (`count[doy]`) is "
            "shared across every cell/channel, so these cells' climatology is "
            "averaged over a few fewer real days than the divisor implies; see "
            "task-25 report's documented concern for the measured impact",
            n_partial, int(wet_mask.sum()), n_days_total, min_frac,
        )

    # Divide in place and cast down before dropping the float64 accumulator -- same
    # reasoning as climatology.build_climatology: avoids materializing a second
    # full-size float64 buffer via `accum_sum / divisor`.
    divisor = np.where(count == 0, 1, count).astype(np.float64)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        accum_sum /= divisor
    clim = accum_sum.astype(np.float32)
    del accum_sum, divisor

    # Force NaN at every (cell, channel) never finite across the whole reference
    # period (`wet_mask`, permanent land/sub-bathymetry) -- the actual "land cells
    # hold NaN" contract; without this, a land cell (contributing 0 above) would
    # average to 0.0, not NaN.
    clim = np.where(wet_mask[None, :, :], clim, np.nan).astype(np.float32)

    # Doys with count==0 (possible only for Feb 29 -- array index 59 -- if the
    # reference period had no leap year) are filled from the neighbouring doy, same
    # as climatology.build_climatology.
    for idx in np.nonzero(count == 0)[0]:
        neighbor = idx - 1 if idx > 0 else 365
        clim[idx] = clim[neighbor]
        logger.info(
            "doy %d had zero samples in [%d, %d]; filled from doy %d",
            idx + 1, ref_start, ref_end, neighbor + 1,
        )

    if out_path.exists():
        shutil.rmtree(out_path)
    out = zarr.open_group(str(out_path), mode="w")
    out.create_array("clim", data=clim, chunks=(1, n_cells, nchan))
    out.create_array("count", data=count.astype(np.int32), chunks=(366,))
    out.attrs.update(
        {
            "channels": channels,
            "ref_start": int(ref_start),
            "ref_end": int(ref_end),
            "n_lat": int(n_lat),
            "n_lon": int(n_lon),
            "grid_res": eerie_grid.GRID_RES,
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument("--stream", required=True, choices=["OCEAN_2D", "OCEAN_3D"])
    parser.add_argument(
        "--ref-start", type=int, default=None, help="Overrides cfg['climatology']['ref_start']."
    )
    parser.add_argument(
        "--ref-end", type=int, default=None, help="Overrides cfg['climatology']['ref_end']."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing climatology output."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    ref_start = args.ref_start if args.ref_start is not None else cfg["climatology"]["ref_start"]
    ref_end = args.ref_end if args.ref_end is not None else cfg["climatology"]["ref_end"]

    out_path = clim_dir(cfg) / f"clim_{args.stream}_{ref_start}-{ref_end}.zarr"
    if out_path.exists() and not args.force:
        print(f"Refusing to overwrite existing climatology: {out_path} (pass --force to overwrite)")
        sys.exit(1)

    logger.info(
        "reading channel list for %s from results store %s", args.stream, cfg["results_dir"]
    )
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    s0 = root.store.samples[0]
    channels = list(root[f"{s0}/{args.stream}/1/target"].attrs["channels"])

    grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])
    truth_path = cfg["truth"][args.stream]
    logger.info(
        "building EERIE climatology: stream=%s truth=%s ref=%d-%d out=%s n_cells=%d nchan=%d",
        args.stream, truth_path, ref_start, ref_end, out_path, grid.n_cells, len(channels),
    )
    build_climatology_eerie(
        truth_path=truth_path,
        channels=channels,
        grid=grid,
        ref_start=ref_start,
        ref_end=ref_end,
        out_path=out_path,
    )
    logger.info("done: %s", out_path)


if __name__ == "__main__":
    main()
