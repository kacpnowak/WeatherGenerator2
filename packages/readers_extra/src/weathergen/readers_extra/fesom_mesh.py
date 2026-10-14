"""Static FESOM mesh information for the FESOM reader: wet layers, bottom depth, layer depths."""

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class MeshEntity:
    """Nodes or elements of a FESOM mesh."""

    n_wet_layers: NDArray  # (npts,) number of wet layers at each point
    bottom_depth: NDArray  # (npts,) depth of the sea floor, m (positive)
    nz1: NDArray  # (nlev,) layer mid-depths, m


def read_mesh_entity(mesh_path: str | Path, n_points: int) -> MeshEntity:
    """The mesh entity (nodes or elements) with `n_points` points."""
    with xr.open_dataset(mesh_path) as ds:
        nz1 = ds["nz1"].values.astype(np.float64)
        if n_points == ds.sizes["nod2"]:
            n_wet = ds["nlevels_nod2D"].values - 1
            bottom = -ds["zbar_n_bottom"].values
        elif n_points == ds.sizes["elem"]:
            n_wet = ds["nlevels"].values - 1
            bottom = -ds["zbar_e_bottom"].values
        else:
            raise ValueError(
                f"{n_points} points are neither the mesh's nodes ({ds.sizes['nod2']}) nor its "
                f"elements ({ds.sizes['elem']}): {mesh_path}"
            )
    return MeshEntity(n_wet.astype(np.int64), bottom.astype(np.float64), nz1)


def channel_level(name: str, nz1: NDArray) -> int | None:
    """Level index of a `<var>_<depth>` channel (nearest layer mid-depth), None for 2-D ones."""
    head, _, tail = name.rpartition("_")
    if not head or not tail.isdigit():
        return None
    return int(np.argmin(np.abs(nz1 - float(tail))))


def dry_mask(channels: list[str], mesh: MeshEntity, n_steps: int) -> NDArray:
    """(n_steps * npts, nchannels): True where a channel's level lies below the sea floor."""
    day = np.zeros((mesh.n_wet_layers.shape[0], len(channels)), dtype=bool)
    for j, name in enumerate(channels):
        level = channel_level(name, mesh.nz1)
        if level is not None:
            day[:, j] = mesh.n_wet_layers <= level
    return np.tile(day, (n_steps, 1))


def bathymetry(mesh: MeshEntity) -> NDArray:
    """log(1 + bottom depth in m) per point."""
    return np.log1p(mesh.bottom_depth)
