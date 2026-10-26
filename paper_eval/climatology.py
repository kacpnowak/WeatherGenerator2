"""Day-of-year climatology of the FESOM ground truth, built once per stream from the
AWICM3 truth archive.

Consumed later for ACC anomalies and as the climatology baseline forecast. Building it
is the long-pole batch job of the pipeline (hours per stream over a 30-year reference
period), so `build_clim.py` submits it via `sbatch/build_clim.sbatch` rather than
running it inline with the rest of the compute stage.
"""

import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger(__name__)

#: Cumulative days before each month (index 1..12) in a FIXED leap calendar (Feb = 29
#: days, always). Index 0 is unused.
_CUM_DAYS_BEFORE_MONTH = np.array([0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])

#: (month, day) -> 1..366 lookup, keyed by month*32+day. Built once at import time.
#: Feb 29 always maps to 60 and Mar 1 always maps to 61, regardless of whether the
#: calendar year of a given date is itself a leap year -- unlike np.datetime64's raw
#: dayofyear, which shifts Mar 1 between 60 and 61 depending on the year.
_MONTH_DAY_TO_DOY = np.zeros(13 * 32, dtype=np.int32)
for _m in range(1, 13):
    for _d in range(1, 32):
        _MONTH_DAY_TO_DOY[_m * 32 + _d] = _CUM_DAYS_BEFORE_MONTH[_m] + _d


def doy_index(dates: np.ndarray) -> np.ndarray:
    """Fixed leap-calendar day-of-year index, 1..366, vectorized over `dates`.

    `dates` is any np.datetime64 array (or scalar); only the calendar date matters
    (time-of-day, if present, is truncated). Feb 29 is always 60 and Mar 1 is always
    61 -- a fixed (month, day) -> index map, not the raw (year-dependent) dayofyear.
    """
    days = np.asarray(dates).astype("datetime64[D]")
    months = days.astype("datetime64[M]")
    month = (months.astype(np.int64) % 12 + 1).astype(np.int64)
    day = (days - months).astype(np.int64) + 1
    return _MONTH_DAY_TO_DOY[month * 32 + day]


def _read_day(data_arr: zarr.Array, npts: int, d: int) -> np.ndarray:
    """Read one full day chunk (npts, n_cols)."""
    return np.asarray(data_arr[d * npts : (d + 1) * npts, :])


def build_climatology(
    truth_dir: str | Path,
    channels: list[str],
    ref_start: int,
    ref_end: int,
    out_path: str | Path,
) -> None:
    """Build a day-of-year climatology over years [ref_start, ref_end] and write it
    to `out_path` as a zarr group.

    `truth_dir` contains one zarr v2 store per year, named `<prefix>_{YYYY}.zarr`
    where `<prefix>` is `truth_dir.name` (e.g. `.../AWICM3/ocean_node` holds
    `ocean_node_2000.zarr`, `ocean_node_2001.zarr`, ...). Each store's `data` array
    is read one full day-chunk at a time (a 1-worker ThreadPoolExecutor prefetches
    day d+1 while day d accumulates) to bound memory to one year's worth of chunks
    in flight while still overlapping I/O with the (attrs-driven) column slicing.

    The float64 accumulator spans the full 366-day climatology at truth resolution
    for the whole reference period (~16 GB node / ~28 GB elem) -- expected and fine
    on a 96 GB+ node; do not shrink it with float32 accumulation.
    """
    truth_dir = Path(truth_dir)
    out_path = Path(out_path)
    prefix = truth_dir.name
    channels = list(channels)
    nchan = len(channels)

    accum_sum = None  # (366, npts, nchan) float64, allocated once npts is known
    count = np.zeros(366, dtype=np.int64)
    npts = None

    for year in range(ref_start, ref_end + 1):
        t_year0 = time.perf_counter()
        store_path = truth_dir / f"{prefix}_{year}.zarr"
        g = zarr.open_group(str(store_path), mode="r")
        data_arr = g["data"]
        attrs = data_arr.attrs.asdict()

        year_npts = attrs.get("n_points")
        if year_npts is None:
            year_npts = attrs.get("nod2")
        if year_npts is None:
            raise ValueError(f"{store_path}: data.attrs has neither 'n_points' nor 'nod2'")
        year_npts = int(year_npts)

        if accum_sum is None:
            npts = year_npts
            accum_sum = np.zeros((366, npts, nchan), dtype=np.float64)
        elif year_npts != npts:
            raise ValueError(f"{store_path}: npts {year_npts} != expected {npts} (prior years)")

        colnames = list(attrs["colnames"])
        cols = [colnames.index(c) for c in channels]

        dates_col = np.asarray(g["dates"][:, 0])
        day_dates = dates_col[::npts]
        n_days = data_arr.shape[0] // npts
        if n_days != len(day_dates):
            raise ValueError(
                f"{store_path}: n_days {n_days} (from data.shape) != "
                f"{len(day_dates)} (from dates stride) -- ragged year?"
            )

        doys = doy_index(day_dates)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_read_day, data_arr, npts, 0)
            for d in range(n_days):
                day_block = future.result()
                if d + 1 < n_days:
                    future = pool.submit(_read_day, data_arr, npts, d + 1)
                field = day_block[:, cols].astype(np.float64)
                doy = int(doys[d])
                accum_sum[doy - 1] += field
                count[doy - 1] += 1

        elapsed = time.perf_counter() - t_year0
        logger.info("year %d: %d days in %.1fs", year, n_days, elapsed)

    if accum_sum is None:
        raise ValueError(f"ref_start={ref_start} > ref_end={ref_end}: no years to process")

    # Divide in place and cast down before dropping the float64 accumulator: `accum_sum
    # / divisor` would materialize a second full-size (366, npts, nchan) float64 buffer
    # (another ~16-28 GB) before the final astype, doubling peak memory for no reason.
    divisor = np.where(count == 0, 1, count).astype(np.float64)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        accum_sum /= divisor
    clim = accum_sum.astype(np.float32)
    del accum_sum, divisor

    # Doys with count==0 (possible only for Feb 29 -- array index 59 -- if the
    # reference period had no leap year) are filled from the neighbouring doy
    # (the previous day in the fixed calendar; Feb 28 for Feb 29).
    for idx in np.nonzero(count == 0)[0]:
        neighbor = idx - 1 if idx > 0 else 365
        clim[idx] = clim[neighbor]
        logger.info("doy %d had zero samples in [%d, %d]; filled from doy %d", idx + 1, ref_start, ref_end, neighbor + 1)

    if out_path.exists():
        shutil.rmtree(out_path)
    out = zarr.open_group(str(out_path), mode="w")
    out.create_array("clim", data=clim, chunks=(1, npts, nchan))
    out.create_array("count", data=count.astype(np.int32), chunks=(366,))
    out.attrs.update({"channels": channels, "ref_start": int(ref_start), "ref_end": int(ref_end)})


class ClimLookup:
    """Day-of-year climatology reader, LRU-cached by doy."""

    def __init__(self, path: str | Path, channels: list[str], cache_size: int = 64):
        self._root = zarr.open_group(str(path), mode="r")
        stored = list(self._root.attrs["channels"])
        channels = list(channels)
        if stored != channels:
            raise ValueError(
                f"climatology channel order mismatch: stored {stored} != requested {channels}"
            )
        self._clim = self._root["clim"]
        self._load_doy = lru_cache(maxsize=cache_size)(self._load_doy)

    def _load_doy(self, doy: int) -> np.ndarray:
        arr = np.asarray(self._clim[doy - 1])
        arr.setflags(write=False)  # cached and shared -- see get()'s docstring
        return arr

    def get(self, date: np.datetime64) -> np.ndarray:
        """(npts, nchan) float32 climatology field for the day-of-year of `date`.

        The returned array is the shared, LRU-cached buffer for that day-of-year --
        it is NOT a fresh copy, and it is marked read-only (`writeable=False`) so any
        in-place mutation attempt (e.g. `result[:] = ...`, `result += 1`) raises
        `ValueError` immediately instead of silently corrupting the cached value for
        every later caller. Pure binary ops that allocate a new array (`pred - clim`,
        `np.where(..., clim, ...)`, etc.) are fine and the normal way to use this.
        """
        doy = int(doy_index(np.datetime64(date)))
        return self._load_doy(doy)


class ChannelClimLookup:
    """Day-of-year climatology assembled by channel NAME from one or more climatology
    zarrs -- the per-variable streams' counterpart of `ClimLookup` (same `get`).

    Each requested channel is read from the first of `paths` whose stored channel list
    holds it, so a stream can reuse a shared climatology (e.g. FESOM_NODE's, for
    temp_2..temp_450) and take the rest from a supplementary one (FESOM_NODE_extra, for
    temp_535.. and sea ice). Every source must have the same point count; a channel
    found in none of them raises ValueError.
    """

    def __init__(self, paths: list[str | Path], channels: list[str], cache_size: int = 64):
        channels = list(channels)
        remaining = list(range(len(channels)))
        self._sources: list[tuple[zarr.Array, np.ndarray, np.ndarray]] = []
        npts = None
        for path in paths:
            if not remaining:
                break
            root = zarr.open_group(str(path), mode="r")
            stored = list(root.attrs["channels"])
            take = [i for i in remaining if channels[i] in stored]
            if not take:
                continue
            clim = root["clim"]
            if npts is None:
                npts = clim.shape[1]
            elif clim.shape[1] != npts:
                raise ValueError(f"{path}: {clim.shape[1]} points != {npts} of the other sources")
            src_idx = np.array([stored.index(channels[i]) for i in take])
            self._sources.append((clim, src_idx, np.array(take)))
            remaining = [i for i in remaining if i not in take]
        if remaining:
            raise ValueError(
                f"climatology: channel(s) {[channels[i] for i in remaining]} in none of {list(map(str, paths))}"
            )
        self._npts = npts
        self._nchan = len(channels)
        self._load_doy = lru_cache(maxsize=cache_size)(self._load_doy)

    def _load_doy(self, doy: int) -> np.ndarray:
        arr = np.empty((self._npts, self._nchan), dtype=np.float32)
        for clim, src_idx, dst_idx in self._sources:
            arr[:, dst_idx] = np.asarray(clim[doy - 1])[:, src_idx]
        arr.setflags(write=False)  # cached and shared, as in ClimLookup.get
        return arr

    def get(self, date: np.datetime64) -> np.ndarray:
        """(npts, nchan) float32, read-only, shared (see `ClimLookup.get`)."""
        doy = int(doy_index(np.datetime64(date)))
        return self._load_doy(doy)
