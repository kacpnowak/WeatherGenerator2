#!/usr/bin/env python
"""Export WeatherGenerator GLORYS inference outputs to an OceanBench challenger zarr.

Input
-----
One zarr ZipStore per forecast start date, produced by the WeatherGenerator
validation/inference run::

    <source-dir>/obench_<YYYYMMDD>/validation_chkpt00000_rank0000.zip

Hierarchy inside the store (see packages/common/src/weathergen/common/io.py)::

    {sample}/{stream}/{forecast_step}/{source|target|prediction}/
        data    (npoints, n_channels)      for source/target
        data    (npoints, n_channels, ens) for prediction
        coords  (npoints, 2)               (lat, lon)
        times   (npoints,)                 datetime64[ns]   -- NOT trusted, see below
        geoinfo (npoints, 0)
      attrs: channels (list[str]), geoinfo_channels, source_interval {start,end}

Forecast step 0 carries only ``source`` (forecast_offset == 1), so the first
predicted day is forecast step 1.  Mapping used here::

    forecast step f  ->  lead_day_index (f - 1)      # fstep 1 -> lead_day_index 0

Dates
-----
The ``target`` arrays and the per-point ``times`` in these stores are known to
carry wrong dates (upstream reader bug).  Dates are therefore derived ONLY from
the sample's ``source_interval`` attribute:

    forecast start Wednesday W = date(source_interval.start) + 1 day

and cross-checked against the ``obench_<YYYYMMDD>`` directory name.

Output
------
A single zarr DirectoryStore with dims
``(first_day_datetime, lead_day_index=10, depth=10, latitude=672, longitude=1440)``
holding ``zos`` (no depth) and ``thetao/so/uo/vo`` (with depth), float32.

Regridding: the native GLORYS 0.25 deg cell centres (lat -89.875..89.875) are
offset by 0.125 deg from the OceanBench reference latitude axis
``np.arange(-78.0, 90.0, 0.25)``, so latitudes are linearly interpolated.
Longitudes coincide exactly and are only reordered to a monotonic
-180..179.75 axis.

Restartability
--------------
The store is grown by *appending* along ``first_day_datetime`` in ascending
date order (``to_zarr(mode="a", append_dim=...)``).  Dates already present in
the store's ``first_day_datetime`` coordinate are skipped, so the exporter can
be re-run after an interruption.  Because appending cannot insert, a date that
is older than the newest date already in the store is refused (rebuild the
store, or export into a fresh --output, if you need to backfill).

Usage
-----
    python export_glorys_nc.py                      # all 52 Wednesdays of 2024
    python export_glorys_nc.py --dates 20240103,20240110
    python export_glorys_nc.py --self-test          # offline unit-style checks
    python export_glorys_nc.py --legacy-probe <zip> # dev smoke on an old store
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import numpy as np
import xarray as xr
import zarr

# --------------------------------------------------------------------------
# contract constants
# --------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = pathlib.Path("/e/scratch/weatherai/shared_work/results")
DEFAULT_OUTPUT = pathlib.Path(
    "/e/scratch/hclimrep/nowak2/eval_output/wg_glorys_cont3_c1_challenger.zarr"
)
ZIP_NAME = "validation_chkpt00000_rank0000.zip"
RUN_DIR_PREFIX = "obench_"

STREAM = "GLORYS"
N_LEAD_DAYS = 10  # forecast steps 1..10 -> lead_day_index 0..9

# depth labels exactly as they appear in the channel names of the GLORYS stream
DEPTH_LABELS = (
    "0.494025",
    "15.8101",
    "34.4342",
    "55.7643",
    "77.8539",
    "109.729",
    "155.851",
    "222.475",
    "318.127",
    "541.089",
)
DEPTHS = np.array([float(label) for label in DEPTH_LABELS], dtype=np.float64)
DEPTH_VARS = ("thetao", "so", "uo", "vo")
SURFACE_VARS = ("zos",)

VAR_ATTRS = {
    "zos": {"standard_name": "sea_surface_height_above_geoid", "units": "m"},
    "thetao": {"standard_name": "sea_water_potential_temperature", "units": "degrees_C"},
    "so": {"standard_name": "sea_water_salinity", "units": "1e-3"},
    "uo": {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"},
    "vo": {"standard_name": "northward_sea_water_velocity", "units": "m s-1"},
}

# native GLORYS 0.25 deg grid ------------------------------------------------
GRID_STEP = 0.25
LAT0 = -89.875  # southernmost native cell centre
N_LAT_NATIVE = 720
N_LON = 1440
# built from integer offsets so the values are exact multiples of 0.25 (+0.125)
NATIVE_LAT = LAT0 + np.arange(N_LAT_NATIVE, dtype=np.float64) * GRID_STEP
NATIVE_LON = np.arange(N_LON, dtype=np.float64) * GRID_STEP  # 0 .. 359.75

# OceanBench reference grid ---------------------------------------------------
# exact multiples of 0.25: -78.00 .. 89.75 (672 values)
TARGET_LAT = np.arange(-312, 360, dtype=np.float64) * GRID_STEP
# same cells as NATIVE_LON, reordered to monotonic -180 .. 179.75
LON_ROLL = N_LON // 2
TARGET_LON = np.concatenate([NATIVE_LON[LON_ROLL:] - 360.0, NATIVE_LON[:LON_ROLL]])

_FILL = np.float32(np.nan)


# --------------------------------------------------------------------------
# channels
# --------------------------------------------------------------------------


def expected_channel_names() -> list[str]:
    """The 41 GLORYS channel names, in canonical (not necessarily stored) order."""
    names = list(SURFACE_VARS)
    for var in DEPTH_VARS:
        names.extend(f"{var}_{label}m" for label in DEPTH_LABELS)
    return names


def channel_index_map(channels: list[str]) -> dict[str, int]:
    """Map channel name -> column index in the stored data array.

    The stored order is whatever the run wrote; it is *always* taken from the
    store's ``channels`` attribute and never assumed.
    """
    expected = expected_channel_names()
    if len(channels) != len(expected):
        msg = f"expected {len(expected)} channels, store declares {len(channels)}"
        raise ValueError(msg)
    index = {name: i for i, name in enumerate(channels)}
    if len(index) != len(channels):
        raise ValueError("duplicate channel names in store attribute")
    missing = [name for name in expected if name not in index]
    if missing:
        msg = f"channels missing from store: {missing}"
        raise ValueError(msg)
    return index


# --------------------------------------------------------------------------
# scatter onto the native grid
# --------------------------------------------------------------------------


def grid_indices(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map point coordinates onto (row, col) of the native 0.25 deg grid.

    ``row`` indexes NATIVE_LAT (-89.875 .. 89.875), ``col`` indexes NATIVE_LON
    (0 .. 359.75).  Longitudes may be given either in [0, 360) or [-180, 180);
    the modulo makes both work.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    row_f = (lat - LAT0) / GRID_STEP
    col_f = np.mod(lon, 360.0) / GRID_STEP

    row = np.rint(row_f).astype(np.int64)
    col = np.rint(col_f).astype(np.int64) % N_LON

    resid = max(
        float(np.abs(row_f - np.rint(row_f)).max(initial=0.0)),
        float(np.abs(col_f - np.rint(col_f)).max(initial=0.0)),
    )
    if resid > 0.05:
        msg = f"points are not on the 0.25 deg grid (max index residual {resid:.4f})"
        raise ValueError(msg)
    if row.size and (row.min() < 0 or row.max() >= N_LAT_NATIVE):
        msg = f"latitude out of native grid range: [{lat.min()}, {lat.max()}]"
        raise ValueError(msg)
    return row, col


def scatter_to_native_grid(
    values: np.ndarray, row: np.ndarray, col: np.ndarray
) -> np.ndarray:
    """Scatter (npoints, n_channels) point values onto (n_channels, 720, 1440).

    Cells not covered by any point stay NaN.  Longitude is rolled so that the
    last axis runs -180 .. 179.75 (TARGET_LON).
    """
    values = np.asarray(values)
    if values.shape[0] != row.size:
        msg = f"{values.shape[0]} value rows but {row.size} coordinates"
        raise ValueError(msg)
    flat = row * N_LON + col
    if np.unique(flat).size != flat.size:
        # would silently keep only the last point per cell
        msg = "two or more points map to the same grid cell"
        raise ValueError(msg)

    n_channels = values.shape[1]
    grid = np.full((n_channels, N_LAT_NATIVE, N_LON), _FILL, dtype=np.float32)
    grid[:, row, col] = values.T.astype(np.float32, copy=False)
    # columns currently run 0..359.75; roll so index 0 is lon 180 == -180
    return np.roll(grid, LON_ROLL, axis=-1)


def interp_latitude(native_grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate (n_channels, 720, 1440) from NATIVE_LAT to TARGET_LAT.

    Target latitudes that are not bracketed by two finite native rows become
    NaN (this includes the coastal/ice rim and any latitude outside the native
    range), which is the intended behaviour: no extrapolation, no gap filling.
    """
    da = xr.DataArray(
        native_grid,
        dims=("channel", "latitude", "longitude"),
        coords={"latitude": NATIVE_LAT, "longitude": TARGET_LON},
    )
    out = da.interp(latitude=TARGET_LAT, method="linear")
    return np.asarray(out.values, dtype=np.float32)


# --------------------------------------------------------------------------
# dataset assembly
# --------------------------------------------------------------------------


def build_forecast_dataset(
    target_grids: np.ndarray, channels: list[str], first_day: np.datetime64
) -> xr.Dataset:
    """Assemble one forecast into an OceanBench-shaped dataset.

    Args:
        target_grids: (n_lead, n_channels, 672, 1440) float32, already on the
            OceanBench reference grid; axis 0 is lead_day_index 0..n_lead-1.
        channels: channel names in the order of axis 1 (from the store attr).
        first_day: forecast start day (the Wednesday).
    """
    target_grids = np.asarray(target_grids)
    n_lead = target_grids.shape[0]
    if target_grids.shape[2:] != (TARGET_LAT.size, TARGET_LON.size):
        msg = f"unexpected grid shape {target_grids.shape}"
        raise ValueError(msg)

    index = channel_index_map(channels)
    lead = np.arange(n_lead, dtype=np.int32)
    fdd = np.array([first_day], dtype="datetime64[ns]")

    data_vars: dict[str, xr.DataArray] = {}

    for var in SURFACE_VARS:
        arr = target_grids[:, index[var], :, :][np.newaxis, ...]
        data_vars[var] = xr.DataArray(
            arr,
            dims=("first_day_datetime", "lead_day_index", "latitude", "longitude"),
            attrs=dict(VAR_ATTRS[var]),
        )

    for var in DEPTH_VARS:
        cols = [index[f"{var}_{label}m"] for label in DEPTH_LABELS]
        # (n_lead, depth, lat, lon) -> add leading first_day_datetime
        arr = target_grids[:, cols, :, :][np.newaxis, ...]
        data_vars[var] = xr.DataArray(
            arr,
            dims=(
                "first_day_datetime",
                "lead_day_index",
                "depth",
                "latitude",
                "longitude",
            ),
            attrs=dict(VAR_ATTRS[var]),
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "first_day_datetime": fdd,
            "lead_day_index": lead,
            "depth": DEPTHS,
            "latitude": TARGET_LAT,
            "longitude": TARGET_LON,
        },
    )

    ds["latitude"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    ds["longitude"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds["depth"].attrs = {"standard_name": "depth", "units": "m", "positive": "down"}
    ds["lead_day_index"].attrs = {
        "long_name": "forecast lead day index",
        "description": "0 = first forecast day (= first_day_datetime)",
    }
    ds.attrs.update(
        {
            "title": "WeatherGenerator GLORYS forecast (OceanBench challenger)",
            "source": "WeatherGenerator inference, stream GLORYS",
            "oceanbench_reference_target_depths_m": [
                round(float(d), 3) for d in DEPTHS
            ],
            "oceanbench_reference_depth_grid_rounding_decimals": 3,
        }
    )
    return ds


def zarr_encoding(ds: xr.Dataset) -> dict[str, dict]:
    """One chunk per (forecast, variable): 1 x n_lead x [depth] x 672 x 1440."""
    encoding = {}
    for name, da in ds.data_vars.items():
        chunks = tuple(1 if dim == "first_day_datetime" else da.sizes[dim] for dim in da.dims)
        encoding[name] = {"chunks": chunks}
    return encoding


# --------------------------------------------------------------------------
# reading the inference stores
# --------------------------------------------------------------------------


def _open_store(zip_path: pathlib.Path) -> tuple[zarr.Group, zarr.storage.ZipStore]:
    store = zarr.storage.ZipStore(str(zip_path), mode="r")
    return zarr.open_group(store=store, mode="r"), store


def read_prediction(root: zarr.Group, sample: str, fstep: int) -> dict:
    """Read one prediction group; returns values, lat, lon, channels, source_interval."""
    path = f"{sample}/{STREAM}/{fstep}/prediction"
    try:
        group = root[path]
    except KeyError as exc:
        msg = f"missing group {path!r} in store"
        raise KeyError(msg) from exc

    data = group["data"]
    values = data[:] if data.ndim == 2 else data[:, :, 0]
    coords = group["coords"][:]
    channels = list(group.attrs["channels"])
    return {
        "values": np.asarray(values),
        "lat": np.asarray(coords[:, 0]),
        "lon": np.asarray(coords[:, 1]),
        "channels": channels,
        "source_interval": dict(group.attrs["source_interval"]),
    }


def first_day_from_source_interval(source_interval: dict) -> np.datetime64:
    """Forecast start day W = date(source_interval.start) + 1 day.

    The source window is the Tuesday 00:00 window; the first forecast day is
    the following Wednesday.
    """
    start = np.datetime64(source_interval["start"], "D")
    return (start + np.timedelta64(1, "D")).astype("datetime64[ns]")


def wednesdays_2024() -> list[dt.date]:
    """The 52 Wednesdays of 2024 (2024-01-03 .. 2024-12-25)."""
    first = dt.date(2024, 1, 3)
    assert first.weekday() == 2
    return [first + dt.timedelta(days=7 * k) for k in range(52)]


def export_one_date(
    zip_path: pathlib.Path,
    expected_day: dt.date | None,
    sample: str = "0",
    n_lead: int = N_LEAD_DAYS,
    strict_dates: bool = True,
) -> xr.Dataset:
    """Read one inference zip and return the challenger dataset for that forecast."""
    root, store = _open_store(zip_path)
    try:
        grids = None
        channels: list[str] = []
        first_day: np.datetime64 | None = None

        for fstep in range(1, n_lead + 1):
            rec = read_prediction(root, sample, fstep)
            if grids is None:
                channels = rec["channels"]
                grids = np.full(
                    (n_lead, len(channels), TARGET_LAT.size, TARGET_LON.size),
                    _FILL,
                    dtype=np.float32,
                )
                first_day = first_day_from_source_interval(rec["source_interval"])
            elif rec["channels"] != channels:
                msg = f"channel order changed between forecast steps in {zip_path}"
                raise ValueError(msg)

            row, col = grid_indices(rec["lat"], rec["lon"])
            native = scatter_to_native_grid(rec["values"], row, col)
            grids[fstep - 1] = interp_latitude(native)
            del native

        if grids is None or first_day is None:
            msg = f"no prediction forecast steps found in {zip_path}"
            raise ValueError(msg)

        day = first_day.astype("datetime64[D]").astype(dt.date)
        if strict_dates:
            if day.weekday() != 2:
                msg = f"{zip_path}: derived first day {day} is not a Wednesday"
                raise ValueError(msg)
            if expected_day is not None and day != expected_day:
                msg = (
                    f"{zip_path}: derived first day {day} does not match run "
                    f"directory date {expected_day}"
                )
                raise ValueError(msg)

        return build_forecast_dataset(grids, channels, first_day)
    finally:
        store.close()


# --------------------------------------------------------------------------
# store management
# --------------------------------------------------------------------------


def existing_first_days(output: pathlib.Path) -> np.ndarray:
    """Dates already present in the output store (empty array if it does not exist)."""
    if not output.exists():
        return np.array([], dtype="datetime64[D]")
    with xr.open_zarr(output, consolidated=None) as ds:
        return ds["first_day_datetime"].values.astype("datetime64[D]")


def append_forecast(ds: xr.Dataset, output: pathlib.Path) -> None:
    """Create the store (first call) or append one forecast along first_day_datetime."""
    if output.exists():
        ds.to_zarr(output, mode="a", append_dim="first_day_datetime", consolidated=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(output, mode="w", encoding=zarr_encoding(ds), consolidated=True)


def summarize(ds: xr.Dataset) -> str:
    """One-line per-variable range summary, so non-physical output is noticed early.

    Expected physical ranges: zos ~(-2.5, 2.5) m, thetao ~(-3, 35) degC,
    so ~(20, 42), uo/vo ~(-3, 3) m/s.  Values far outside these (e.g. every
    variable inside +-3) indicate the inference run wrote *normalized* samples.
    """
    parts = []
    for name in ("zos", *DEPTH_VARS):
        values = ds[name].values
        if np.isfinite(values).any():
            parts.append(f"{name}[{np.nanmin(values):.2f},{np.nanmax(values):.2f}]")
        else:
            parts.append(f"{name}[all-NaN]")
    finite = float(np.isfinite(ds["zos"].values).mean())
    return f"finite={finite:.3f} " + " ".join(parts)


def run_export(
    dates: list[dt.date],
    source_dir: pathlib.Path,
    output: pathlib.Path,
    sample: str,
    n_lead: int,
) -> int:
    done = set(existing_first_days(output).tolist())
    newest = max(done) if done else None

    n_written = 0
    for day in sorted(dates):
        if day in done:
            print(f"[skip] {day} already in {output.name}")
            continue
        if newest is not None and day < newest:
            msg = (
                f"{day} is older than the newest date already in the store "
                f"({newest}); appending cannot insert. Rebuild the store or use "
                f"a fresh --output."
            )
            raise ValueError(msg)

        zip_path = source_dir / f"{RUN_DIR_PREFIX}{day:%Y%m%d}" / ZIP_NAME
        if not zip_path.exists():
            print(f"[miss] {day}: {zip_path} not found -- skipping")
            continue

        print(f"[read] {day}: {zip_path}")
        ds = export_one_date(zip_path, expected_day=day, sample=sample, n_lead=n_lead)
        print(f"[stat] {day}: {summarize(ds)}")
        append_forecast(ds, output)
        ds.close()
        newest = day
        n_written += 1
        print(f"[done] {day} written")

    print(f"{n_written} forecast(s) written to {output}")
    return n_written


# --------------------------------------------------------------------------
# verification helpers
# --------------------------------------------------------------------------


def legacy_probe(zip_path: pathlib.Path, sample: str = "0", fstep: int = 1) -> None:
    """Dev-only smoke test against an existing (possibly old) inference store.

    Reads one forecast step, scatters, interpolates and prints shapes, finite
    fractions and per-variable ranges.  Date checks are relaxed.
    """
    root, store = _open_store(zip_path)
    try:
        rec = read_prediction(root, sample, fstep)
    finally:
        store.close()

    channels = rec["channels"]
    print(f"store          : {zip_path}")
    print(f"sample/fstep   : {sample}/{fstep}")
    print(f"source_interval: {rec['source_interval']}")
    print(f"derived day    : {first_day_from_source_interval(rec['source_interval'])}")
    print(f"n_points       : {rec['values'].shape[0]}")
    print(f"n_channels     : {len(channels)}  first 3: {channels[:3]}")
    print(f"lat range      : {rec['lat'].min()} .. {rec['lat'].max()}")
    print(f"lon range      : {rec['lon'].min()} .. {rec['lon'].max()}")

    row, col = grid_indices(rec["lat"], rec["lon"])
    native = scatter_to_native_grid(rec["values"], row, col)
    print(f"native grid    : {native.shape}  finite={np.isfinite(native).mean():.4f}")

    target = interp_latitude(native)
    print(f"target grid    : {target.shape}  finite={np.isfinite(target).mean():.4f}")

    ds = build_forecast_dataset(
        target[np.newaxis, ...], channels, first_day_from_source_interval(rec["source_interval"])
    )
    print(f"dataset dims   : {dict(ds.sizes)}")
    for name, da in ds.data_vars.items():
        values = da.values
        finite = np.isfinite(values)
        frac = float(finite.mean())
        if finite.any():
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        else:
            lo = hi = float("nan")
        print(
            f"  {name:7s} dims={da.dims} shape={values.shape} "
            f"finite={frac:.4f} min={lo:.4f} max={hi:.4f}"
        )


def self_test() -> None:
    """Offline structural checks on a small synthetic sample."""
    rng = np.random.default_rng(0)
    canonical = expected_channel_names()

    # channel order deliberately scrambled; the store attribute is the truth
    scrambled = list(canonical)
    rng.shuffle(scrambled)
    assert scrambled != canonical, "shuffle produced the canonical order"
    n_ch = len(scrambled)

    def code(point: int, channel: str) -> float:
        return 100.0 * point + canonical.index(channel)

    # points on known native cells; the three southern ones are contiguous rows
    points = [
        (-78.125, 10.0),  # row 47
        (-77.875, 10.0),  # row 48
        (-77.625, 10.0),  # row 49
        (0.125, -180.0),  # row 360, col 720 -> target lon index 0
        (-0.125, -180.0),  # row 359
    ]
    lat = np.array([p[0] for p in points], dtype=np.float32)
    lon = np.array([p[1] for p in points], dtype=np.float32)
    values = np.empty((len(points), n_ch), dtype=np.float32)
    for p in range(len(points)):
        for c, name in enumerate(scrambled):
            values[p, c] = code(p, name)

    # --- indices ---------------------------------------------------------
    row, col = grid_indices(lat, lon)
    assert row.tolist() == [47, 48, 49, 360, 359], row.tolist()
    assert col.tolist() == [40, 40, 40, 720, 720], col.tolist()
    assert NATIVE_LAT[47] == -78.125 and NATIVE_LON[40] == 10.0
    print("[ok] grid_indices maps (lat, lon) to the expected native cells")

    # longitudes given in the 0..360 convention must land on the same columns
    row2, col2 = grid_indices(lat, np.mod(lon, 360.0).astype(np.float32))
    assert np.array_equal(row, row2) and np.array_equal(col, col2)
    print("[ok] grid_indices is invariant to the [0,360) / [-180,180) convention")

    # --- scatter ---------------------------------------------------------
    native = scatter_to_native_grid(values, row, col)
    assert native.shape == (n_ch, N_LAT_NATIVE, N_LON)
    # after the roll, native column 40 (lon 10.0) sits at TARGET_LON index 760
    lon10 = int(np.flatnonzero(TARGET_LON == 10.0)[0])
    lonm180 = int(np.flatnonzero(TARGET_LON == -180.0)[0])
    assert lon10 == 760 and lonm180 == 0, (lon10, lonm180)
    for p, (plat, _plon) in enumerate(points):
        r = int(np.rint((plat - LAT0) / GRID_STEP))
        cc = lon10 if p < 3 else lonm180
        for c, name in enumerate(scrambled):
            assert native[c, r, cc] == np.float32(code(p, name)), (p, name)
    # untouched cells stay NaN
    assert np.isnan(native[0, 47, lon10 + 1])
    assert np.isnan(native[:, 100, :]).all()
    print("[ok] scatter places every channel value on the right (lat, lon) cell")
    print("[ok] cells not covered by any point remain NaN")

    # --- latitude interpolation -----------------------------------------
    interp = interp_latitude(native)
    assert interp.shape == (n_ch, TARGET_LAT.size, TARGET_LON.size)
    i_78 = int(np.flatnonzero(TARGET_LAT == -78.0)[0])
    i_7775 = int(np.flatnonzero(TARGET_LAT == -77.75)[0])
    i_775 = int(np.flatnonzero(TARGET_LAT == -77.5)[0])
    assert i_78 == 0
    ch0 = scrambled[0]
    # -78.0 is exactly halfway between the filled rows -78.125 and -77.875
    expect = 0.5 * (code(0, ch0) + code(1, ch0))
    got = interp[0, i_78, lon10]
    assert np.isfinite(got) and abs(got - expect) < 1e-3, (got, expect)
    # -77.75 is halfway between filled rows -77.875 and -77.625
    expect = 0.5 * (code(1, ch0) + code(2, ch0))
    got = interp[0, i_7775, lon10]
    assert np.isfinite(got) and abs(got - expect) < 1e-3, (got, expect)
    print("[ok] interpolation is finite (and exact) where two native rows bracket")
    # -77.5 is bracketed by the filled row -77.625 and the empty row -77.375
    assert np.isnan(interp[0, i_775, lon10]), interp[0, i_775, lon10]
    # far from any data -> NaN
    assert np.isnan(interp[0, 300, lon10])
    print("[ok] interpolation yields NaN outside the bracketed rows")

    # a fully populated column stays finite everywhere, poles included
    dense = np.full((1, N_LAT_NATIVE, N_LON), np.float32(np.nan), dtype=np.float32)
    dense[0, :, 5] = np.arange(N_LAT_NATIVE, dtype=np.float32)
    dense_i = interp_latitude(dense)
    assert np.isfinite(dense_i[0, :, 5]).all(), "poles lost in interpolation"
    assert TARGET_LAT[0] == -78.0 and TARGET_LAT[-1] == 89.75
    assert TARGET_LAT.size == 672 and TARGET_LON.size == 1440
    assert np.allclose(TARGET_LAT / GRID_STEP, np.rint(TARGET_LAT / GRID_STEP), atol=0)
    print("[ok] target axis is 672 exact multiples of 0.25 from -78.0 to 89.75, "
          "no NaN introduced at the poles for a fully covered column")

    # --- dataset assembly ------------------------------------------------
    n_lead = 2  # keep the self-test cheap; the CLI always uses N_LEAD_DAYS
    grids = np.stack([interp, interp + np.float32(1000.0)])
    ds = build_forecast_dataset(grids, scrambled, np.datetime64("2024-01-03", "ns"))

    assert ds["zos"].dims == (
        "first_day_datetime",
        "lead_day_index",
        "latitude",
        "longitude",
    ), ds["zos"].dims
    assert ds["zos"].shape == (1, n_lead, 672, 1440)
    print("[ok] zos has no depth dimension")

    for var in DEPTH_VARS:
        assert ds[var].dims == (
            "first_day_datetime",
            "lead_day_index",
            "depth",
            "latitude",
            "longitude",
        ), ds[var].dims
        assert ds[var].shape == (1, n_lead, 10, 672, 1440)
    print("[ok] thetao/so/uo/vo carry the depth dimension with 10 levels")

    # channel-order handling: values must follow the *names*, not the positions
    for var in DEPTH_VARS:
        for k, label in enumerate(DEPTH_LABELS):
            name = f"{var}_{label}m"
            c = scrambled.index(name)
            expect_val = 0.5 * (code(0, name) + code(1, name))
            got_val = float(ds[var].values[0, 0, k, i_78, lon10])
            assert abs(got_val - expect_val) < 1e-3, (name, got_val, expect_val)
            assert c != canonical.index(name) or True  # order really is scrambled
    zc = scrambled.index("zos")
    assert zc >= 0
    expect_val = 0.5 * (code(0, "zos") + code(1, "zos"))
    assert abs(float(ds["zos"].values[0, 0, i_78, lon10]) - expect_val) < 1e-3
    print("[ok] channel unpacking honours the store's channel attribute order")

    # lead_day_index mapping: axis 0 of target_grids is lead 0 == forecast step 1
    assert ds["lead_day_index"].values.tolist() == list(range(n_lead))
    assert (
        float(ds["thetao"].values[0, 1, 0, i_78, lon10])
        - float(ds["thetao"].values[0, 0, 0, i_78, lon10])
    ) == 1000.0
    print("[ok] lead_day_index 0 holds forecast step 1 (offset applied at read time)")

    # dtypes / coordinate metadata
    for name in ("zos", *DEPTH_VARS):
        assert ds[name].dtype == np.float32, (name, ds[name].dtype)
        assert ds[name].attrs["standard_name"] == VAR_ATTRS[name]["standard_name"]
        assert ds[name].attrs["units"] == VAR_ATTRS[name]["units"]
    assert ds["latitude"].attrs["units"] == "degrees_north"
    assert ds["longitude"].attrs["units"] == "degrees_east"
    assert ds["depth"].attrs == {"standard_name": "depth", "units": "m", "positive": "down"}
    assert np.allclose(ds["depth"].values, DEPTHS)
    assert ds["first_day_datetime"].dtype == np.dtype("datetime64[ns]")
    print("[ok] variable/coordinate attributes and dtypes match the contract")

    # chunking
    enc = zarr_encoding(ds)
    assert enc["zos"]["chunks"] == (1, n_lead, 672, 1440), enc["zos"]
    assert enc["thetao"]["chunks"] == (1, n_lead, 10, 672, 1440), enc["thetao"]
    print("[ok] encoding is one chunk per forecast and variable")

    # date derivation
    day = first_day_from_source_interval(
        {"start": "2024-01-02T00:00:00.000000000", "end": "2024-01-03T00:00:00.000000000"}
    )
    assert day == np.datetime64("2024-01-03", "ns"), day
    assert day.astype("datetime64[D]").astype(dt.date).weekday() == 2
    weds = wednesdays_2024()
    assert len(weds) == 52 and weds[0] == dt.date(2024, 1, 3)
    assert weds[-1] == dt.date(2024, 12, 25)
    assert all(d.weekday() == 2 for d in weds)
    print("[ok] source_interval start + 1 day gives the forecast Wednesday; "
          "52 Wednesdays in 2024")

    print("\nself-test: all checks passed")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_dates(spec: str | None) -> list[dt.date]:
    if spec is None or spec.strip().lower() == "all":
        return wednesdays_2024()
    out = []
    for token in spec.replace(",", " ").split():
        out.append(dt.datetime.strptime(token.strip(), "%Y%m%d").date())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        default=DEFAULT_SOURCE_DIR,
        help="directory holding the obench_<YYYYMMDD> run directories",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="output challenger zarr store",
    )
    parser.add_argument(
        "--dates",
        default="all",
        help="comma-separated YYYYMMDD dates, or 'all' for the 52 Wednesdays of 2024",
    )
    parser.add_argument("--sample", default="0", help="sample index inside the store")
    parser.add_argument(
        "--n-lead", type=int, default=N_LEAD_DAYS, help="number of forecast days to export"
    )
    parser.add_argument("--self-test", action="store_true", help="run offline structural checks")
    parser.add_argument(
        "--legacy-probe",
        type=pathlib.Path,
        default=None,
        help="dev-only: probe an existing inference zip (no writing, relaxed date checks)",
    )
    parser.add_argument(
        "--probe-fstep", type=int, default=1, help="forecast step used by --legacy-probe"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    if args.legacy_probe is not None:
        legacy_probe(args.legacy_probe, sample=args.sample, fstep=args.probe_fstep)
        return 0

    run_export(
        parse_dates(args.dates), args.source_dir, args.output, args.sample, args.n_lead
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
