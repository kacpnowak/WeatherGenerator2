# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in inference treatment of the latent land cells (``forecast.land_cells``).

The forecast-engine latent has one token per HEALPix cell. Cells that hold no node of the
ocean mesh ("land cells") receive no data and are never supervised; the per-step forcing is
still added to them. This option changes what happens to their tokens in a validation /
inference rollout. Configured under the mode config's ``forecast`` block, e.g.::

    test_config:
      forecast:
        land_cells: {mode: freeze, expected_num_land_cells: 2863}

Unset, null or ``mode: none`` disables it, and the rollout is exactly the default one. It is
read in validation and inference only; training never applies it.

Modes:

``freeze``
    After every forecast step (forecast engine, then the latent filter if one is set) the
    land-cell tokens are reset to their values in the initial encoded latent, i.e. the
    latent before the first forecast step of the rollout. The forcing is still injected into
    the land cells before each forecast-engine step. The initial land tokens are carried
    across rollout chunks (``ModelOutput.land_cells_init``) and saved with the resumable
    rollout state.

``no_forcing``
    The forcing injection of every step leaves the land-cell tokens unchanged (their tokens
    after the injection are those before it), so the forcing reaches the latent only through
    the ocean cells. Needs a token-nudging forcing mode (``additive`` or ``cross_attn``).

Land cells are the cells (nested HEALPix ordering, the latent's level) that contain no node
of the mesh files (``mesh_path``) of the non-forcing streams; the node coordinates are the
``lat`` / ``lon`` variables of the mesh file, binned with the tokenizer's convention
(``coords_to_hpyidxs``). ``expected_num_land_cells`` (optional) makes the rollout fail when
the computed count differs.
"""

from __future__ import annotations

import dataclasses
import zlib
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.datasets.utils import coords_to_hpyidxs

LAND_CELL_MODES = ("freeze", "no_forcing")
#: forcing injection modes that act on the cell tokens (needed by ``no_forcing``)
TOKEN_FORCING_MODES = ("additive", "cross_attn")


def mesh_node_latlon(path: str | Path) -> tuple[NDArray, NDArray]:
    """Node latitudes and longitudes (degrees) of a FESOM mesh diagnostics file."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return np.asarray(ds["lat"].values, np.float64), np.asarray(ds["lon"].values, np.float64)


def cells_with_points(healpix_level: int, lats: NDArray, lons: NDArray) -> NDArray:
    """Boolean (12 * 4**level,): True for the cells holding at least one of the points."""
    has = np.zeros(12 * 4**healpix_level, dtype=bool)
    has[np.asarray(coords_to_hpyidxs(healpix_level, np.asarray(lats), np.asarray(lons)))] = True
    return has


def land_cell_mask(streams_cfg, healpix_level: int) -> NDArray:
    """
    Boolean (12 * 4**level,): True for the latent cells without a node of the mesh of any
    non-forcing stream that has a ``mesh_path``.
    """
    paths = sorted(
        {
            str(s.get("mesh_path"))
            for s in streams_cfg.values()
            if s.get("mesh_path") and s.get("type") != "forcing"
        }
    )
    if not paths:
        raise ValueError("forecast.land_cells needs a stream with a mesh_path (the ocean mesh)")
    ocean = np.zeros(12 * 4**healpix_level, dtype=bool)
    for p in paths:
        ocean |= cells_with_points(healpix_level, *mesh_node_latlon(p))
    return ~ocean


@dataclasses.dataclass(frozen=True, eq=False)
class LandCells:
    """One of the `LAND_CELL_MODES` and the boolean land mask over the latent cells."""

    mode: str
    land_mask: torch.Tensor

    def __post_init__(self):
        if self.mode not in LAND_CELL_MODES:
            raise ValueError(
                f"forecast.land_cells.mode must be one of {LAND_CELL_MODES} or none, "
                f"got {self.mode!r}"
            )
        mask = torch.as_tensor(self.land_mask, dtype=torch.bool).flatten().cpu()
        object.__setattr__(self, "land_mask", mask)
        object.__setattr__(self, "_index", {})

    @classmethod
    def from_config(cls, cfg, streams_cfg=None, healpix_level=None, land_mask=None):
        """
        The setting of a `forecast.land_cells` config entry; None when it is disabled. The
        mask is computed from the streams' meshes unless `land_mask` is given.
        """
        if cfg is None:
            return None
        mode = cfg.get("mode")
        if mode is None or str(mode).lower() == "none":
            return None
        unknown = set(cfg.keys()) - {"mode", "expected_num_land_cells"}
        if unknown:
            raise ValueError(f"unknown forecast.land_cells keys: {sorted(unknown)}")
        if str(mode) not in LAND_CELL_MODES:
            raise ValueError(
                f"forecast.land_cells.mode must be one of {LAND_CELL_MODES} or none, got {mode!r}"
            )
        if land_mask is None:
            land_mask = land_cell_mask(streams_cfg, healpix_level)
        lc = cls(mode=str(mode), land_mask=torch.as_tensor(np.asarray(land_mask)))
        expected = cfg.get("expected_num_land_cells")
        if expected is not None and lc.num_land_cells != int(expected):
            raise ValueError(
                f"forecast.land_cells: {lc.num_land_cells} land cells, expected {int(expected)}"
            )
        return lc

    @property
    def num_land_cells(self) -> int:
        return int(self.land_mask.sum())

    @property
    def needs_init(self) -> bool:
        """Whether the initial land tokens are carried through the rollout."""
        return self.mode == "freeze"

    def as_dict(self) -> dict:
        """Mode, land-cell count and a checksum of the mask (saved with the rollout state)."""
        mask_bytes = self.land_mask.numpy().astype(np.uint8).tobytes()
        return {
            "mode": self.mode,
            "num_land_cells": self.num_land_cells,
            "num_cells": int(self.land_mask.numel()),
            "mask_crc32": zlib.crc32(mask_bytes),
        }

    def token_index(self, tokens: torch.Tensor, num_aux: int) -> torch.Tensor:
        """Indices of the land-cell tokens along dim 1 of `tokens` (num_aux aux tokens first)."""
        num_cells = tokens.shape[1] - num_aux
        if num_cells != self.land_mask.numel():
            raise ValueError(
                f"forecast.land_cells: latent has {num_cells} cell tokens, the land mask "
                f"{self.land_mask.numel()} cells"
            )
        key = (tokens.device, num_aux)
        if key not in self._index:
            idx = torch.nonzero(self.land_mask).flatten() + num_aux
            self._index[key] = idx.to(tokens.device)
        return self._index[key]

    def initial_tokens(self, tokens: torch.Tensor, num_aux: int) -> torch.Tensor | None:
        """Land tokens of the initial latent (B, num_land, dim) for `freeze`, else None."""
        if not self.needs_init:
            return None
        return tokens.index_select(1, self.token_index(tokens, num_aux)).detach().clone()

    def after_forcing(
        self, z_before: torch.Tensor, z_forced: torch.Tensor, num_aux: int
    ) -> torch.Tensor:
        """`no_forcing`: the forced tokens with the land tokens of `z_before`."""
        if self.mode != "no_forcing":
            return z_forced
        idx = self.token_index(z_forced, num_aux)
        return z_forced.index_copy(1, idx, z_before.index_select(1, idx).to(z_forced.dtype))

    def after_step(
        self, tokens: torch.Tensor, init: torch.Tensor | None, num_aux: int
    ) -> torch.Tensor:
        """`freeze`: the stepped tokens with the land tokens reset to `init`."""
        if self.mode != "freeze":
            return tokens
        if init is None:
            raise ValueError("forecast.land_cells freeze: no initial land tokens to reset to")
        idx = self.token_index(tokens, num_aux)
        return tokens.index_copy(1, idx, init.to(tokens.dtype))
