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

#: 2-D (single-level) channel names for FESOM_NODE.
_CHANNELS_2D = {"sst", "sss", "ssh"}


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

    2-D channels (sst, sss, ssh) return (name, None, None). 3-D channels are named
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
        stream: "FESOM_NODE" (npts = nod2) or "FESOM_ELEM" (npts = nelem).
        channels: channel names, parsed with `parse_channel`.

    Returns:
        mask: (nchan, npts) bool. The sole wet/dry authority downstream -- never
            derive wetness from data values.
        w: (nchan, npts) float32 area weight (m^2).
    """
    if stream not in ("FESOM_NODE", "FESOM_ELEM"):
        raise ValueError(f"Unknown stream: {stream!r}")

    nchan = len(channels)

    if stream == "FESOM_NODE":
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

    # FESOM_ELEM: all channels are 3-D.
    npts = mesh.elem_area.shape[0]
    mask = np.empty((nchan, npts), dtype=bool)
    w = np.empty((nchan, npts), dtype=np.float32)
    for i, name in enumerate(channels):
        _, _, level_idx = parse_channel(name, mesh.nz1)
        if level_idx is None:
            raise ValueError(f"FESOM_ELEM channel {name!r} must be 3-D (got a 2-D name)")
        mask[i] = mesh.n_wet_layers_elem > level_idx
        w[i] = mesh.elem_area
    return mask, w
