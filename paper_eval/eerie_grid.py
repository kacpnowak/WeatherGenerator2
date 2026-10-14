"""EERIE gridded-stream geometry: the fixed 0.25 degree global lat/lon grid used by
run dbuzcpgf's OCEAN_2D/OCEAN_3D streams.

Unlike FESOM's native, irregular CORE2 mesh (``meshtools.py``: a fixed per-stream
node/element index shared by every sample and forecast step), EERIE's
target/prediction points are GLOBAL_SPARSE points RE-DRAWN every (sample, forecast
step) -- there is no fixed point index to build a mask/weight array against ahead of
time. What IS fixed is the grid those sparse points are drawn from: every point (sparse
target/prediction, or the full-grid source) sits exactly on a cell of a global 0.25
degree lat/lon grid, so geometry here means mapping (lat, lon) -> a flat cell index
rather than indexing into a fixed mesh (see ``scores.py``'s ``geometry: eerie``
branch).

Only grid geometry and per-channel normalization stats are read here -- never the
truth archives' actual channel data (not needed by Task 24: the EERIE climatology
baseline is Task 25's job).
"""

from pathlib import Path

import fsspec
import numpy as np
import zarr

#: Grid spacing, degrees. The grid spans lat -90..90 (721 points) and lon -180..179.75
#: (1440 points) -- the same convention the results archive's own point coordinates
#: use (verified against run dbuzcpgf: target/source coords land exactly on this grid,
#: residual 0.0), which is NOT the raw truth-store node arrays' own longitude phase
#: (0..359.75) -- see `load_grid`.
GRID_RES = 0.25

#: Data variable dimension signature identifying a 2D (surface) gridded field in the
#: truth kerchunk store, used to read the grid's (n_lat, n_lon) shape without reading
#: any channel data.
_GRID_DIMS = ("time", "surface", "lat", "lon")


class EerieGrid:
    """Fixed 0.25-degree global grid: (lat, lon) -> flat cell index `row * n_lon + col`.

    `lat`/`lon` are the grid's own ascending axis values (`lat`: -90..90 inclusive,
    `lon`: -180..179.75), matching the longitude convention the results archive's
    `target`/`prediction`/`source` point coordinates already use -- NOT necessarily
    the raw truth-store `lat`/`lon` node arrays' own convention (see `load_grid`).
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat = lat
        self.lon = lon
        self.n_cells = int(lat.size * lon.size)

    def cell_index(self, coords: np.ndarray) -> np.ndarray:
        """(npts, 2) `[lat, lon]` -> flat int64 index `row * n_lon + col`.

        Rounds each coordinate to the nearest grid cell via exact rounding
        (`(lat + 90) / GRID_RES`, `(lon + 180) / GRID_RES`) and raises if any point's
        residual from its nearest cell is not negligible -- per the verified data
        facts, every point (sparse target/prediction or full-grid source) IS a grid
        cell; a large residual means a coordinate convention mismatch, not a point
        needing nearest-neighbour snapping.
        """
        coords = np.asarray(coords)
        lat = coords[:, 0].astype(np.float64)
        lon = coords[:, 1].astype(np.float64)

        row_f = (lat + 90.0) / GRID_RES
        col_f = (lon + 180.0) / GRID_RES
        row = np.rint(row_f).astype(np.int64)
        col = np.rint(col_f).astype(np.int64)

        resid = np.maximum(np.abs(row_f - row), np.abs(col_f - col))
        max_resid = float(resid.max()) if resid.size else 0.0
        if max_resid >= 1e-3:
            raise ValueError(
                f"EerieGrid.cell_index: max abs residual {max_resid!r} >= 1e-3 -- a "
                "coordinate does not land on the 0.25 deg grid cell it was assumed to"
            )
        if (row.min(initial=0) < 0 or row.max(initial=0) >= self.lat.size
                or col.min(initial=0) < 0 or col.max(initial=0) >= self.lon.size):
            raise ValueError("EerieGrid.cell_index: a coordinate falls outside the grid extent")

        return row * self.lon.size + col

    def cell_latitudes(self, idx: np.ndarray) -> np.ndarray:
        """Latitude of each flat cell index (`row = idx // n_lon`)."""
        idx = np.asarray(idx)
        return self.lat[idx // self.lon.size]


def _open_kerchunk(path: str | Path) -> zarr.Group:
    """Open a kerchunk-parquet-referenced zarr store read-only, via `fsspec`'s
    `reference` filesystem -- no `xr.open_zarr`: only attrs/shape metadata is needed
    here, never the (huge, CF-decoded) channel data itself."""
    fs = fsspec.filesystem("reference", fo=str(path))
    return zarr.open_group(fs.get_mapper(""), mode="r")


#: `load_grid` result cache, keyed by truth path -- module-level so every stream of a
#: worker process shares one `EerieGrid` instance (grid geometry is identical for
#: OCEAN_2D/OCEAN_3D: both are the same physical 0.25 deg earth grid) and repeat calls
#: (e.g. once per worker process, once again in the `compute.py` parent for
#: `reduce_blocks`) don't re-open the kerchunk store.
_grid_cache: dict[str, EerieGrid] = {}


def load_grid(truth_2d_path: str | Path) -> EerieGrid:
    """Load (and cache) the fixed lat/lon grid from a 2D EERIE truth kerchunk store.

    Reads only grid geometry: a `(time, surface, lat, lon)` data variable's shape
    (for `n_lat`/`n_lon`, rather than hardcoding 721/1440) and the store's own
    per-node `lat`/`lon` coordinate arrays (`nodes = n_lat * n_lon`, one global 0.25
    deg grid including land), used only to validate resolution/count -- NOT their
    values: the store's raw longitude phase (0..359.75) differs from the results
    archive's point-coordinate convention (-180..179.75, verified), so the returned
    `lon` axis is built analytically in that latter convention rather than copied
    from the store.
    """
    key = str(truth_2d_path)
    if key in _grid_cache:
        return _grid_cache[key]

    g = _open_kerchunk(truth_2d_path)
    data_names = [
        name
        for name in g.array_keys()
        if tuple(g[name].attrs.get("_ARRAY_DIMENSIONS", ())) == _GRID_DIMS
    ]
    if not data_names:
        raise ValueError(f"{truth_2d_path}: no {_GRID_DIMS} data variable found")
    n_lat, n_lon = g[data_names[0]].shape[-2:]

    lat_nodes = np.asarray(g["lat"], dtype=np.float64)
    lon_nodes = np.asarray(g["lon"], dtype=np.float64)
    if lat_nodes.size != n_lat * n_lon or lon_nodes.size != n_lat * n_lon:
        raise ValueError(
            f"{truth_2d_path}: lat/lon node count {lat_nodes.size} != "
            f"n_lat*n_lon ({n_lat}*{n_lon})"
        )
    lat2 = lat_nodes.reshape(n_lat, n_lon)
    lon2 = lon_nodes.reshape(n_lat, n_lon)
    if not np.allclose(lat2, lat2[:, :1]) or not np.allclose(lon2, lon2[:1, :]):
        raise ValueError(
            f"{truth_2d_path}: lat/lon node arrays are not a regular (lat, lon) grid "
            "(lat must be constant per row, lon constant per column)"
        )

    lat_spacing = np.diff(lat2[:, 0])
    lon_spacing = np.diff(lon2[0, :])
    if not np.allclose(lat_spacing, GRID_RES) or not np.allclose(lon_spacing, GRID_RES):
        raise ValueError(f"{truth_2d_path}: grid spacing != {GRID_RES} deg")

    lat = -90.0 + GRID_RES * np.arange(n_lat, dtype=np.float64)
    lon = -180.0 + GRID_RES * np.arange(n_lon, dtype=np.float64)
    if not np.isclose(lat[-1], 90.0):
        raise ValueError(f"{truth_2d_path}: lat axis does not reach 90 deg (got {lat[-1]})")

    grid = EerieGrid(lat, lon)
    _grid_cache[key] = grid
    return grid


def load_norm_stats(truth_path: str | Path, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(mean_vec, std_vec) float32, aligned to `channels` order -- the per-channel
    normalization constants the training reader applied to build `source`, read from
    a kerchunk truth store's global attrs (`weathergen_means`: dict ch -> mean,
    `weathergen_vars`: dict ch -> VARIANCE, `std = sqrt(variance)`).

    Raises `ValueError` on any channel missing from the attrs (never silently reuses
    a mismatched constant), mirroring `scores._load_norm_stats`'s FESOM counterpart.
    """
    g = _open_kerchunk(truth_path)
    means = g.attrs["weathergen_means"]
    variances = g.attrs["weathergen_vars"]

    mean_vec = np.empty(len(channels), dtype=np.float32)
    std_vec = np.empty(len(channels), dtype=np.float32)
    for i, name in enumerate(channels):
        if name not in means or name not in variances:
            raise ValueError(
                f"{truth_path}: channel {name!r} missing from weathergen_means/"
                "weathergen_vars"
            )
        mean_vec[i] = means[name]
        std_vec[i] = np.sqrt(variances[name])
    return mean_vec, std_vec
