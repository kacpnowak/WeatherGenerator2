"""Nearest-neighbour regridding from the native FESOM mesh to a regular lat/lon grid.

Used only by the spectra pipeline (plan decision D6): a field defined on mesh nodes
(``FESOM_NODE``) or element centroids (``FESOM_ELEM``) is looked up at its nearest
source point for every regular-grid cell centre, with grid cells farther than
``max_dist_km`` from any source point marked invalid (NaN) rather than extrapolated.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from paper_eval.meshtools import Mesh

#: Mean Earth radius (km) used to convert unit-sphere chord distance to great-circle km.
_EARTH_RADIUS_KM = 6371.0


def _lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit-sphere xyz for `lon`/`lat` in degrees (any matching shape); xyz on last axis."""
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    coslat = np.cos(lat_r)
    return np.stack([coslat * np.cos(lon_r), coslat * np.sin(lon_r), np.sin(lat_r)], axis=-1)


def _elem_centroids_xyz(mesh: Mesh) -> np.ndarray:
    """(nelem, 3) unit-sphere centroid of each triangle.

    Mean of the three vertex xyz vectors, re-normalised to unit length. Averaging raw
    longitudes would be wrong across the dateline; this 3-D-mean-then-renormalise
    approach sidesteps that entirely.
    """
    node_xyz = _lonlat_to_xyz(mesh.lon, mesh.lat)  # (nod2, 3)
    tri_xyz = node_xyz[mesh.tri_elem]  # (nelem, 3, 3)
    c = tri_xyz.mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return c


def _n_source(mesh: Mesh, stream: str) -> int:
    """Number of source points `stream` has on `mesh` (nodes or elements)."""
    if stream == "FESOM_NODE":
        return mesh.lon.shape[0]
    if stream == "FESOM_ELEM":
        return mesh.tri_elem.shape[0]
    raise ValueError(f"Unknown stream: {stream!r}")


def _grid_centres(res_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes/longitudes of a regular `res_deg` grid."""
    nlat = round(180 / res_deg)
    nlon = round(360 / res_deg)
    glat = -90 + res_deg / 2 + res_deg * np.arange(nlat)
    glon = -180 + res_deg / 2 + res_deg * np.arange(nlon)
    return glat, glon


def _cache_path(cache_dir: str | Path, stream: str, res_deg: float) -> Path:
    return Path(cache_dir) / f"regrid_{stream}_{res_deg}.npz"


class Regridder:
    """Nearest-neighbour lookup from mesh source points to a regular lat/lon grid.

    Attributes:
        idx: (nlat, nlon) int32, index into the source array of the nearest point.
        valid: (nlat, nlon) bool, True where that nearest point is within `max_dist_km`.
        glat: (nlat,) cell-centre latitudes.
        glon: (nlon,) cell-centre longitudes.
        n_source: number of source points the regridder was built against (used by
            `__call__`'s shape assert and by `load_or_build`'s cache sanity check).
    """

    def __init__(
        self, idx: np.ndarray, valid: np.ndarray, glat: np.ndarray, glon: np.ndarray, n_source: int
    ):
        self.idx = idx
        self.valid = valid
        self.glat = glat
        self.glon = glon
        self.n_source = n_source

    @classmethod
    def build(
        cls, mesh: Mesh, stream: str, res_deg: float = 0.25, max_dist_km: float = 150.0
    ) -> "Regridder":
        """Build a regridder for `stream`'s source points onto a `res_deg` regular grid.

        `stream == "FESOM_NODE"` uses mesh nodes (`mesh.lon`, `mesh.lat`);
        `stream == "FESOM_ELEM"` uses element centroids (`_elem_centroids_xyz`). Nearest
        neighbour is found with a `cKDTree` over source xyz on the unit sphere, queried
        at every grid-cell-centre xyz; chord distance is converted to great-circle km
        (`angle = 2*arcsin(min(d,2)/2)`, `km = 6371.0*angle`) and compared to
        `max_dist_km` for validity.
        """
        if stream == "FESOM_NODE":
            src_xyz = _lonlat_to_xyz(mesh.lon, mesh.lat)
        elif stream == "FESOM_ELEM":
            src_xyz = _elem_centroids_xyz(mesh)
        else:
            raise ValueError(f"Unknown stream: {stream!r}")

        glat, glon = _grid_centres(res_deg)
        glon_g, glat_g = np.meshgrid(glon, glat)  # each (nlat, nlon)
        grid_xyz = _lonlat_to_xyz(glon_g, glat_g)  # (nlat, nlon, 3)
        nlat, nlon = glat.shape[0], glon.shape[0]

        tree = cKDTree(src_xyz)
        d, idx = tree.query(grid_xyz.reshape(-1, 3))
        angle = 2 * np.arcsin(np.minimum(d, 2.0) / 2.0)
        km = _EARTH_RADIUS_KM * angle
        valid = km <= max_dist_km

        return cls(
            idx=idx.reshape(nlat, nlon).astype(np.int32),
            valid=valid.reshape(nlat, nlon),
            glat=glat,
            glon=glon,
            n_source=src_xyz.shape[0],
        )

    @classmethod
    def load_or_build(
        cls, cache_dir: str | Path, mesh: Mesh, stream: str, res_deg: float, max_dist_km: float
    ) -> "Regridder":
        """Load `<cache_dir>/regrid_{stream}_{res_deg}.npz` if present, else build+save it."""
        path = _cache_path(cache_dir, stream, res_deg)
        if path.exists():
            r = cls.load(path)
            expected = _n_source(mesh, stream)
            if r.n_source != expected:
                raise ValueError(
                    f"{path}: cached n_source {r.n_source} != mesh {stream} count {expected}"
                )
            return r
        r = cls.build(mesh, stream, res_deg=res_deg, max_dist_km=max_dist_km)
        r.save(path)
        return r

    def __call__(self, field_1d: np.ndarray) -> np.ndarray:
        """Regrid `field_1d` (n_source,) onto the (nlat, nlon) grid, NaN where invalid."""
        assert field_1d.shape[0] == self.n_source, (
            f"field has {field_1d.shape[0]} source points, regridder built for {self.n_source}"
        )
        out = field_1d[self.idx].astype(np.float32)
        out[~self.valid] = np.nan
        return out

    def save(self, path: str | Path) -> None:
        """Write the npz cache (`idx, valid, glat, glon, n_source`) to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            idx=self.idx,
            valid=self.valid,
            glat=self.glat,
            glon=self.glon,
            n_source=self.n_source,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Regridder":
        """Load a regridder previously written by `save`."""
        with np.load(path) as z:
            return cls(
                idx=z["idx"],
                valid=z["valid"],
                glat=z["glat"],
                glon=z["glon"],
                n_source=int(z["n_source"]),
            )
