"""FESOM CORE2 mesh utilities: triangulation, area weights, wet masks, channel parsing.

The mesh (``fesom.mesh.diag.nc``) is the single source of truth for which points are
"wet" at a given vertical level. Masks derived here from ``nlevels``/``nlevels_nod2D``
must be reused identically for prediction, target and climatology data (design decision
D3) -- never derive wetness from data values (e.g. ``== 0.0``), since a real value can
legitimately be zero.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.tri import Triangulation

#: 2-D (single-level) channel names on mesh nodes: FESOM_NODE's surface fields and the
#: per-variable SURFACE stream's sea-ice fields (wet at every node, surface-area weight).
_CHANNELS_2D = {"sst", "sss", "ssh", "a_ice", "m_ice"}

#: Mesh entity of each legacy stream name; `masks_and_weights` also takes the entity
#: ("node"/"elem") directly, which is what per-variable streams (TEMP, SALT, SURFACE,
#: U, V) resolve to through `paper_eval.config.stream_entity`.
_STREAM_ENTITY = {"FESOM_NODE": "node", "FESOM_ELEM": "elem"}


@dataclass
class Mesh:
    """CORE2 mesh geometry, areas and wet-layer counts."""

    lon: np.ndarray  # (nod2,) float64, degrees
    lat: np.ndarray  # (nod2,) float64, degrees
    tri_elem: np.ndarray  # (nelem, 3) int, 0-based node indices
    nod_area: np.ndarray  # (48, nod2) float64, m^2
    elem_area: np.ndarray  # (nelem,) float64, m^2
    n_wet_layers_nod: np.ndarray  # (nod2,) int
    n_wet_layers_elem: np.ndarray  # (nelem,) int
    nz1: np.ndarray  # (47,) float64, mid-layer depths, m


def load_mesh(path: str | Path) -> Mesh:
    """Load the CORE2 mesh diagnostics file into a `Mesh`.

    ``face_nodes`` is stored 1-based as (3, elem); it is transposed and shifted to
    0-based (nelem, 3) to match `tri_elem`.
    """
    with xr.open_dataset(path) as ds:
        lon = ds["lon"].values
        lat = ds["lat"].values
        tri_elem = ds["face_nodes"].values.T - 1
        nod_area = ds["nod_area"].values
        elem_area = ds["elem_area"].values
        n_wet_layers_nod = ds["nlevels_nod2D"].values - 1
        n_wet_layers_elem = ds["nlevels"].values - 1
        nz1 = ds["nz1"].values

    return Mesh(
        lon=lon,
        lat=lat,
        tri_elem=tri_elem,
        nod_area=nod_area,
        elem_area=elem_area,
        n_wet_layers_nod=n_wet_layers_nod,
        n_wet_layers_elem=n_wet_layers_elem,
        nz1=nz1,
    )


def triangulation(mesh: Mesh) -> Triangulation:
    """Matplotlib triangulation of the mesh, with dateline-spanning triangles masked.

    A triangle whose vertices sit at both +180 and -180 would smear a band across
    the whole map, so triangles whose longitude span exceeds 180 degrees are masked.
    """
    tri = Triangulation(mesh.lon, mesh.lat, mesh.tri_elem)
    lon_e = mesh.lon[mesh.tri_elem]
    mask = lon_e.max(axis=1) - lon_e.min(axis=1) > 180
    tri.set_mask(mask)
    return tri


def parse_channel(name: str, nz1: np.ndarray) -> tuple[str, float | None, int | None]:
    """Split a channel name into (prefix, depth, level_idx).

    2-D channels (sst, sss, ssh, a_ice, m_ice) return (name, None, None). 3-D channels are named
    ``<prefix>_<suffix>`` where ``suffix`` is the integer part of a depth in `nz1`;
    the matching depth and its index into `nz1` are returned, e.g.
    ``parse_channel("temp_107", nz1) == ("temp", 107.5, 11)``.

    Raises:
        ValueError: if ``name`` has no depth suffix matching any entry in `nz1`.
    """
    if name in _CHANNELS_2D:
        return name, None, None

    prefix, _, suffix = name.rpartition("_")
    if not prefix:
        raise ValueError(f"Unknown channel name (no depth suffix): {name!r}")

    suffix_int = int(suffix)
    matches = np.nonzero(nz1.astype(int) == suffix_int)[0]
    if matches.size == 0:
        raise ValueError(f"Channel {name!r}: no nz1 depth matches suffix {suffix_int}")

    level_idx = int(matches[0])
    depth = float(nz1[level_idx])
    return prefix, depth, level_idx


def masks_and_weights(
    mesh: Mesh, stream: str, channels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel wet mask and area weight, for every point of `stream`.

    Args:
        stream: mesh entity, "node" (npts = nod2) or "elem" (npts = nelem), or the
            legacy stream name standing for it ("FESOM_NODE"/"FESOM_ELEM").
        channels: channel names, parsed with `parse_channel`.

    Returns:
        mask: (nchan, npts) bool. The sole wet/dry authority downstream -- never
            derive wetness from data values.
        w: (nchan, npts) float32 area weight (m^2).
    """
    entity = _STREAM_ENTITY.get(stream, stream)
    if entity not in ("node", "elem"):
        raise ValueError(f"Unknown stream/mesh entity: {stream!r}")

    nchan = len(channels)

    if entity == "node":
        npts = mesh.lon.shape[0]
        mask = np.empty((nchan, npts), dtype=bool)
        w = np.empty((nchan, npts), dtype=np.float32)
        for i, name in enumerate(channels):
            _, _, level_idx = parse_channel(name, mesh.nz1)
            if level_idx is None:
                mask[i] = True
                w[i] = mesh.nod_area[0]
            else:
                mask[i] = mesh.n_wet_layers_nod > level_idx
                w[i] = mesh.nod_area[level_idx]
        return mask, w

    # elem: all channels are 3-D.
    npts = mesh.elem_area.shape[0]
    mask = np.empty((nchan, npts), dtype=bool)
    w = np.empty((nchan, npts), dtype=np.float32)
    for i, name in enumerate(channels):
        _, _, level_idx = parse_channel(name, mesh.nz1)
        if level_idx is None:
            raise ValueError(f"elem channel {name!r} must be 3-D (got a 2-D name)")
        mask[i] = mesh.n_wet_layers_elem > level_idx
        w[i] = mesh.elem_area
    return mask, w


def _unit_xyz(coords: np.ndarray) -> np.ndarray:
    """(n, 2) [lat, lon] degrees -> (n, 3) unit vectors."""
    lat, lon = np.deg2rad(coords[:, 0].astype(np.float64)), np.deg2rad(coords[:, 1].astype(np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


class PointIndex:
    """A stream's full point set (mesh nodes or elements, in mesh order) and the lookup of
    a subsampled target's points in it.

    Streams trained with `max_num_targets` write only a random subset of the points per
    forecast step (re-drawn every step) in `target`/`prediction`; `source` (fstep 0)
    always holds the full set in mesh order. `densify` scatters such a subset back onto
    the full set (NaN where not sampled), so the fixed per-point masks, weights and
    climatology apply unchanged. Full-size data (every FESOM_NODE/FESOM_ELEM run) is
    returned as is: it is stored in mesh order.
    """

    #: max chord distance (unit sphere) between a target point and its match, ~60 m
    _TOL = 1e-5

    def __init__(self, coords_full: np.ndarray):
        self.coords = np.asarray(coords_full)
        self.npts = self.coords.shape[0]
        self._tree = None

    def index(self, coords: np.ndarray) -> np.ndarray:
        """Position of each of `coords` ((n, 2) [lat, lon]) in the full point set."""
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(_unit_xyz(self.coords))
        dist, idx = self._tree.query(_unit_xyz(np.asarray(coords)))
        if dist.size and dist.max() > self._TOL:
            raise ValueError(f"target point(s) off the stream's point set (max distance {dist.max():.3g})")
        if np.unique(idx).size != idx.size:
            raise ValueError("subsampled target has duplicate points")
        return idx

    def densify(self, data: np.ndarray, coords) -> tuple[np.ndarray, np.ndarray | None]:
        """(full-size data, sampled-point mask or None). `coords` is only read when
        `data` is a subset (pass the zarr array, not its values, to keep that lazy)."""
        if data.shape[0] == self.npts:
            return data, None
        idx = self.index(np.asarray(coords))
        out = np.full((self.npts,) + data.shape[1:], np.nan, dtype=data.dtype)
        out[idx] = data
        sampled = np.zeros(self.npts, dtype=bool)
        sampled[idx] = True
        return out, sampled
