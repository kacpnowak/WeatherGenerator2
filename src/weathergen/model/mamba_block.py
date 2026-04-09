# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Mamba-2 SSM blocks for WeatherGenerator.

Provides drop-in replacements for MultiSelfAttentionHead / MultiSelfAttentionHeadLocal
in the ForecastingEngine and GlobalAssimilationEngine.

Based on:
- Mamba-2 (Dao & Gu, 2024): "Transformers are SSMs" — SSD duality framework
- Nemotron-H (NVIDIA, 2025): hybrid Mamba-2 + sparse attention architecture
- Vision Mamba (Zhu et al., 2024): bidirectional scanning for spatial data

Key design decisions:
1. Bidirectional scanning (forward + backward) to handle non-causal spatial data on HealPix
2. Pre-norm + residual pattern matching existing attention blocks
3. Same forward signature: (x, coords=None, ada_ln_aux=None) for checkpoint compatibility
4. AdaLayerNorm support for auxiliary conditioning (e.g. timestep embedding)
"""

from functools import partial

import torch
import torch.nn as nn

from weathergen.model.norms import AdaLayerNorm, RMSNorm

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


class BiMamba2Block(nn.Module):
    """Bidirectional Mamba-2 SSM block with pre-norm and residual connection.

    Runs Mamba-2 in both forward and backward token order over the sequence
    dimension, then merges the two outputs via a linear projection. This
    addresses the inherent causality of SSMs when applied to spatial
    (non-sequential) data such as HealPix weather grids.

    The forward signature matches MultiSelfAttentionHead / MultiSelfAttentionHeadLocal
    so it can be used as a drop-in replacement in ForecastingEngine.

    Args:
        dim_embed: Model embedding dimension (must be divisible by headdim, default 64).
        d_state: SSM state expansion factor. Typical values: 64 or 128.
        d_conv: Local convolution width in the Mamba block.
        expand: Block expansion factor. Parameters ≈ 3 * expand * dim_embed^2.
        headdim: Head dimension for Mamba-2's multi-head structure.
        dropout_rate: Dropout rate applied after the output projection.
        with_residual: Whether to add residual connection.
        norm_type: "LayerNorm" or "RMSNorm".
        norm_eps: Epsilon for normalization layers.
        dim_aux: If set, use AdaLayerNorm with this auxiliary dimension.
    """

    def __init__(
        self,
        dim_embed: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout_rate: float = 0.0,
        with_residual: bool = True,
        norm_type: str = "LayerNorm",
        norm_eps: float = 1e-5,
        dim_aux: int | None = None,
    ):
        super().__init__()

        if Mamba2 is None:
            raise ImportError(
                "mamba-ssm is not installed. Install with: "
                "MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation"
            )

        assert dim_embed % headdim == 0, (
            f"dim_embed ({dim_embed}) must be divisible by headdim ({headdim})"
        )

        self.with_residual = with_residual
        self.dim_embed = dim_embed

        # Pre-normalization (matching attention block pattern)
        norm_eps_float = float(norm_eps)
        if norm_type == "LayerNorm":
            norm_fn = partial(nn.LayerNorm, elementwise_affine=False, eps=norm_eps_float)
        else:
            norm_fn = partial(RMSNorm, eps=norm_eps_float)

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps_float)
        else:
            self.lnorm = norm_fn(dim_embed)

        # Forward-direction Mamba-2
        self.mamba_fwd = Mamba2(
            d_model=dim_embed,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # Backward-direction Mamba-2 (separate parameters)
        self.mamba_bwd = Mamba2(
            d_model=dim_embed,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        # Merge bidirectional outputs: 2*dim_embed -> dim_embed
        self.out_proj = nn.Linear(2 * dim_embed, dim_embed, bias=False)

        # Dropout (matching attention block pattern)
        self.dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        ada_ln_aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass matching MultiSelfAttentionHead interface.

        Args:
            x: Input tensor of shape (batch, seq_len, dim_embed) or (seq_len, dim_embed).
            coords: Ignored. Accepted for interface compatibility with attention blocks
                    that use RoPE. SSMs have implicit positional information via their
                    recurrent structure.
            ada_ln_aux: Optional auxiliary tensor for AdaLayerNorm conditioning.

        Returns:
            Output tensor of the same shape as input.
        """
        # Handle both 2D (seq, dim) and 3D (batch, seq, dim) inputs
        needs_batch_dim = x.dim() == 2
        if needs_batch_dim:
            x = x.unsqueeze(0)

        if self.with_residual:
            residual = x

        # Pre-norm
        if ada_ln_aux is not None:
            x_normed = self.lnorm(x, ada_ln_aux)
        else:
            x_normed = self.lnorm(x)

        # Forward scan (natural HealPix nested order)
        y_fwd = self.mamba_fwd(x_normed)

        # Backward scan (reversed sequence, then flip output back)
        y_bwd = self.mamba_bwd(x_normed.flip(dims=[1])).flip(dims=[1])

        # Merge bidirectional outputs
        y = self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        y = self.dropout(y)

        # Residual connection
        if self.with_residual:
            y = y + residual

        if needs_batch_dim:
            y = y.squeeze(0)

        return y


class Mamba2Block(nn.Module):
    """Unidirectional Mamba-2 SSM block with pre-norm and residual connection.

    Lighter-weight alternative to BiMamba2Block. Uses a single scan direction.
    Can be used with alternating ordering across layers (Mamba-ND style):
    even layers scan forward, odd layers scan backward.

    Same forward signature as BiMamba2Block and MultiSelfAttentionHead.
    """

    def __init__(
        self,
        dim_embed: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout_rate: float = 0.0,
        with_residual: bool = True,
        norm_type: str = "LayerNorm",
        norm_eps: float = 1e-5,
        dim_aux: int | None = None,
        reverse: bool = False,
    ):
        super().__init__()

        if Mamba2 is None:
            raise ImportError(
                "mamba-ssm is not installed. Install with: "
                "MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation"
            )

        assert dim_embed % headdim == 0, (
            f"dim_embed ({dim_embed}) must be divisible by headdim ({headdim})"
        )

        self.with_residual = with_residual
        self.reverse = reverse

        # Pre-normalization
        norm_eps_float = float(norm_eps)
        if norm_type == "LayerNorm":
            norm_fn = partial(nn.LayerNorm, elementwise_affine=False, eps=norm_eps_float)
        else:
            norm_fn = partial(RMSNorm, eps=norm_eps_float)

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps_float)
        else:
            self.lnorm = norm_fn(dim_embed)

        self.mamba = Mamba2(
            d_model=dim_embed,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

        self.dropout = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        ada_ln_aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass matching MultiSelfAttentionHead interface."""
        needs_batch_dim = x.dim() == 2
        if needs_batch_dim:
            x = x.unsqueeze(0)

        if self.with_residual:
            residual = x

        # Pre-norm
        if ada_ln_aux is not None:
            x_normed = self.lnorm(x, ada_ln_aux)
        else:
            x_normed = self.lnorm(x)

        # Optionally reverse for alternating-order (Mamba-ND style)
        if self.reverse:
            y = self.mamba(x_normed.flip(dims=[1])).flip(dims=[1])
        else:
            y = self.mamba(x_normed)

        y = self.dropout(y)

        if self.with_residual:
            y = y + residual

        if needs_batch_dim:
            y = y.squeeze(0)

        return y
