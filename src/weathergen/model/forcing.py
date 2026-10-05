# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Generic per-step spatial forcing for the forecasting engine.

A *forcing* is a prescribed field (SST, sea ice, aerosols, a CO2 field, solar forcing, ...)
that is re-read at every forecast step's valid time (see the ``type: forcing`` stream) and
injected into the forecasting engine so the rolling latent state is continually reminded of
an external boundary condition. Nothing here is specific to any particular field -- SST is
simply the first instance.

To inject a forcing it is first placed on the FE latent grid: the (fixed) forcing point cloud
is binned (scatter-mean) onto the HEALPix cells the FE tokens live on, then embedded to the
model width by a small MLP. The result -- a per-cell forcing embedding -- is shared by every
injection mode (``additive`` / ``cross_attn`` / ``global`` / ``adaln_local``).

The scatter index depends only on the (fixed) forcing grid and the HEALPix level, so it is
precomputed once via :func:`build_forcing_cell_index`. Everything here is cheap: one scatter
and one matmul per forecast step, independent of the assimilation engine.
"""

import numpy as np
import torch
import torch.nn as nn

from weathergen.datasets.utils import coords_to_hpyidxs


def build_forcing_cell_index(
    latitudes: np.ndarray, longitudes: np.ndarray, healpix_level: int
) -> torch.Tensor:
    """
    Map each forcing grid point to its HEALPix latent cell (nested convention).

    Uses the same convention (``ang2pix(..., nest=True)``) as the tokenizer, so the binned
    forcing cells align one-to-one with the FE latent token cells.

    Parameters
    ----------
    latitudes, longitudes :
        Forcing grid point coordinates, degrees, shape ``(num_points,)``.
    healpix_level :
        HEALPix level of the FE latent grid (num_cells = 12 * 4**healpix_level).

    Returns
    -------
    LongTensor of shape ``(num_points,)`` with the cell index in ``[0, num_cells)`` for
    every forcing point.
    """
    cell_idx = coords_to_hpyidxs(healpix_level, np.asarray(latitudes), np.asarray(longitudes))
    return torch.as_tensor(np.asarray(cell_idx), dtype=torch.long)


def scatter_to_cells(
    values: torch.Tensor, cell_idx: torch.Tensor, num_cells: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter-mean per-point forcing values onto HEALPix cells, ignoring NaNs.

    Parameters
    ----------
    values :
        Forcing point values, shape ``(..., num_points, num_vars)``. NaNs (e.g. SST over
        land) are treated as missing and excluded from the per-cell mean.
    cell_idx :
        Cell index per point, shape ``(num_points,)`` (from ``build_forcing_cell_index``).
    num_cells :
        Number of HEALPix cells.

    Returns
    -------
    (cell_values, cell_valid)
        cell_values : ``(..., num_cells, num_vars)`` per-cell mean (0 where no valid point
            contributed to a cell/var).
        cell_valid  : ``(..., num_cells, num_vars)`` fraction of valid points that
            contributed (0 = no data, e.g. all-land cell), usable as an input feature/mask.
    """
    *batch_shape, num_points, num_vars = values.shape
    flat = values.reshape(-1, num_points, num_vars)  # (B, P, V)
    B = flat.shape[0]
    device = flat.device

    valid = torch.isfinite(flat)  # (B, P, V)
    filled = torch.where(valid, flat, torch.zeros_like(flat))

    idx = cell_idx.to(device).view(1, num_points, 1).expand(B, num_points, num_vars)

    sums = torch.zeros(B, num_cells, num_vars, device=device, dtype=flat.dtype)
    sums.scatter_add_(1, idx, filled)
    counts = torch.zeros(B, num_cells, num_vars, device=device, dtype=flat.dtype)
    counts.scatter_add_(1, idx, valid.to(flat.dtype))

    cell_values = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    total = torch.zeros(B, num_cells, num_vars, device=device, dtype=flat.dtype)
    total.scatter_add_(1, idx, torch.ones_like(filled))
    cell_valid = torch.where(total > 0, counts / total.clamp_min(1.0), torch.zeros_like(counts))

    cell_values = cell_values.reshape(*batch_shape, num_cells, num_vars)
    cell_valid = cell_valid.reshape(*batch_shape, num_cells, num_vars)
    return cell_values, cell_valid


class ForcingEmbed(nn.Module):
    """
    Per-cell forcing embedding shared by all injection modes.

    Normalises the per-cell forcing field with the dataset statistics (or identity if the
    field is already normalised upstream), appends a validity feature (valid fraction per
    cell/var) so the model can tell present from absent/no-data, and maps to the model width
    with a small MLP.

    Parameters
    ----------
    num_vars :
        Number of forcing source channels (e.g. sea_surface_temperature, sea_ice_cover,
        land_sea_mask -> 3).
    dim_embed :
        Output embedding width (typically the FE model dim for additive injection).
    mean, stdev :
        Per-channel normalisation statistics, shape ``(num_vars,)``. If None, the input is
        assumed already normalised.
    hidden_factor :
        Width multiplier for the MLP hidden layer.
    """

    def __init__(
        self,
        num_vars: int,
        dim_embed: int,
        mean: np.ndarray | None = None,
        stdev: np.ndarray | None = None,
        hidden_factor: int = 2,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.dim_embed = dim_embed

        if mean is None:
            mean = np.zeros(num_vars, dtype=np.float32)
        if stdev is None:
            stdev = np.ones(num_vars, dtype=np.float32)
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("stdev", torch.as_tensor(np.asarray(stdev), dtype=torch.float32))

        # input = normalised values (num_vars) concatenated with validity (num_vars)
        in_dim = 2 * num_vars
        hidden = hidden_factor * dim_embed
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_embed),
        )

    def forward(self, cell_values: torch.Tensor, cell_valid: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cell_values : ``(..., num_cells, num_vars)`` per-cell forcing values (from scatter).
        cell_valid  : ``(..., num_cells, num_vars)`` per-cell validity fraction.

        Returns
        -------
        ``(..., num_cells, dim_embed)`` per-cell forcing embedding.
        """
        mean = self.mean.to(cell_values.dtype)
        stdev = self.stdev.to(cell_values.dtype).clamp_min(1e-6)
        normed = (cell_values - mean) / stdev
        # zero-out normalised value where there is no data so it reads as "absent"
        normed = torch.where(cell_valid > 0, normed, torch.zeros_like(normed))
        x = torch.cat([normed, cell_valid.to(normed.dtype)], dim=-1)
        return self.mlp(x)


class ForcingInjection(nn.Module):
    """
    Inject a per-step spatial forcing into the forecasting engine, one of four modes.

    Applied once per forecast step, in the model rollout loop, just before the FE advances the
    ``(B, num_tokens, dim)`` latent state. The auxiliary tokens (register/class) are left
    untouched; only the per-cell patch tokens / the FE conditioning are affected. The forcing
    field for a step is global across the batch (same forecast time), so a per-cell tensor
    ``(num_cells, dim)`` broadcasts across the batch dimension.

    Modes (all a **zero-initialised gated residual** -> exact no-op at init, so a model
    fine-tuned from a pretrained checkpoint starts identically and learns to use the forcing):

      * ``additive``    : add the per-cell forcing embedding to the cell tokens.
      * ``cross_attn``  : cell tokens cross-attend to the per-cell forcing tokens (non-local
                          coupling / teleconnections).
      * ``global``      : pool the forcing over cells and add the projected vector to the
                          (global) FE conditioning -- the cheap "index" flavour; the existing
                          FE AdaLN is untouched.
      * ``adaln_local`` : make the FE conditioning per-cell (daytime condition + projected
                          per-cell forcing); every FE block's AdaLN then modulates each cell
                          token by its local forcing. Requires a condition stream (dim_aux > 0).

    ``forward`` returns ``(tokens, condition)``: token-nudge modes modify ``tokens``,
    conditioning modes modify ``condition``.
    """

    _MODES = ("none", "additive", "cross_attn", "global", "adaln_local")

    def __init__(
        self,
        num_vars: int,
        dim_embed: int,
        mode: str,
        dim_aux: int = 0,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        assert mode in self._MODES, f"unknown forcing mode {mode!r}, expected {self._MODES}"
        self.mode = mode
        self.num_vars = num_vars
        self.dim_embed = dim_embed
        self.dim_aux = dim_aux

        if mode == "none":
            return

        if mode in ("global", "adaln_local") and dim_aux <= 0:
            raise ValueError(
                f"forcing mode {mode!r} routes the forcing through the FE conditioning (AdaLN) "
                "and needs a condition stream (dim_aux > 0); use additive/cross_attn otherwise."
            )

        # Forcing field is already normalised by the sampler (reader stats) -> identity here.
        self.embed = ForcingEmbed(num_vars, dim_embed)
        # zero-init gate -> injection starts as an exact no-op
        self.gate = nn.Parameter(torch.zeros(1))

        if mode == "cross_attn":
            self.norm_q = nn.LayerNorm(dim_embed)
            self.cross_attn = nn.MultiheadAttention(dim_embed, num_heads, batch_first=True)
        elif mode in ("global", "adaln_local"):
            self.cond_proj = nn.Linear(dim_embed, dim_aux)

    @staticmethod
    def _is_empty(x) -> bool:
        return x is None or not torch.is_tensor(x) or x.numel() == 0

    def _cell_emb(self, forcing_field: torch.Tensor, dtype) -> torch.Tensor:
        """Per-cell embedding (num_cells, dim) from the scattered field (num_cells, 2*num_vars)."""
        if forcing_field.dim() == 3:  # tolerate a leading batch dim
            forcing_field = forcing_field[0]
        num_vars = forcing_field.shape[-1] // 2
        cell_values = forcing_field[..., :num_vars]
        cell_valid = forcing_field[..., num_vars:]
        return self.embed(cell_values, cell_valid).to(dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        forcing_field: torch.Tensor,
        num_aux: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        tokens :
            FE latent state ``(B, num_tokens, dim)`` (num_aux leading aux tokens then num_cells).
        condition :
            FE conditioning fed to the AdaLN, global ``(dim_aux,)`` vector (or empty).
        forcing_field :
            Scattered forcing for this step ``(num_cells, 2*num_vars)`` = [cell_values |
            cell_valid]; empty when no forcing is available (then this is a no-op).
        num_aux :
            Number of leading auxiliary tokens to leave untouched.

        Returns
        -------
        (tokens, condition) with the forcing injected according to the mode.
        """
        if self.mode == "none" or self._is_empty(forcing_field):
            return tokens, condition

        cell_emb = self._cell_emb(forcing_field, tokens.dtype)  # (num_cells, dim)

        if self.mode == "additive":
            patch = tokens[:, num_aux:] + self.gate * cell_emb
            tokens = torch.cat([tokens[:, :num_aux], patch], dim=1)
        elif self.mode == "cross_attn":
            q = self.norm_q(tokens[:, num_aux:])  # (B, num_cells, dim)
            kv = cell_emb.unsqueeze(0).expand(q.shape[0], -1, -1)
            delta, _ = self.cross_attn(q, kv, kv, need_weights=False)
            patch = tokens[:, num_aux:] + self.gate * delta
            tokens = torch.cat([tokens[:, :num_aux], patch], dim=1)
        elif self.mode == "global":
            g = self.cond_proj(cell_emb.mean(dim=0))  # (dim_aux,)
            condition = condition + self.gate * g
        elif self.mode == "adaln_local":
            # per-cell conditioning: broadcast the global daytime condition, add per-cell forcing
            base = condition.reshape(1, -1) if condition.numel() > 0 else 0.0
            condition = base + self.gate * self.cond_proj(cell_emb)  # (num_cells, dim_aux)

        return tokens, condition
