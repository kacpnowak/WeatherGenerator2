# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import torch

# healpy and scipy are imported lazily by the spherical RoPE precomputation only, so the model
# imports and runs without them when spherical RoPE is off.


def _import_healpy():
    import healpy as hp  # noqa: PLC0415

    # Suppress verbose healpy transform messages during spherical RoPE coefficient precomputation.
    logging.getLogger("healpy").setLevel(logging.WARNING)
    return hp


####################################################################################################
def positional_encoding_harmonic(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device
    dtype = x.dtype

    len_token_seq = x.shape[-2]
    pe = torch.zeros(len_token_seq, dim_embed, device=dev, dtype=dtype)
    position = torch.arange(0, len_token_seq, device=dev, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim_embed, 2, device=dev, dtype=dtype) * -(math.log(10000) / dim_embed)
    )

    pe[:, 0::2] = torch.sin(position * div[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_idx(x, s_idx):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    len_token_seq = x.shape[0]
    pe = torch.zeros(x.shape[-2:], device=dev)
    pos = (s_idx + 1) * torch.ones(len_token_seq, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed

    pe[:, 0::2] = torch.sin(torch.outer(pos, xs))
    pe[:, 1::2] = torch.cos(torch.outer(pos, xs))
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_global(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    pe = torch.zeros(x.shape[-3], x.shape[-2], dim_embed, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed
    pe[..., 0::2] = 0.5 * torch.sin(torch.outer(8 * torch.arange(x.shape[-2], device=dev), xs))
    pe[..., 0::2] += (
        torch.sin(torch.outer(torch.arange(x.shape[-3], device=dev), xs))
        .unsqueeze(1)
        .repeat((1, x.shape[-2], 1))
    )
    pe[..., 1::2] = 0.5 * torch.cos(torch.outer(8 * torch.arange(x.shape[-2], device=dev), xs))
    pe[..., 1::2] += (
        torch.cos(torch.outer(torch.arange(x.shape[-3], device=dev), xs))
        .unsqueeze(1)
        .repeat((1, x.shape[-2], 1))
    )
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_coord(x, lats, lons):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    pe = torch.zeros(x.shape[0], dim_embed, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed
    pe[..., 0::2] = 0.5 * torch.sin(torch.outer(lats, xs))
    pe[..., 1::2] = 0.5 * torch.cos(torch.outer(lons, xs))[..., : pe[..., 1::2].shape[-1]]
    x = x + pe

    return x


####################################################################################################
# The functions rotate_half() and apply_rotary_pos_emb() below are derived from LLaMA and Qwen3
# models, originally developed by Meta Platforms, Inc., The Qwen team, Alibaba Group and the
# HuggingFace Inc. team, licensed under the Apache License, Version 2.0.
# Source: https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py


def rotate_half(x):
    """Rotates half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine embedding tensor.
        sin: Sine embedding tensor.
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin for broadcasting.
    """

    cos = cos.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


####################################################################################################
def rotary_embedding_2d(coords, dim, base=10000.0):
    """Create 2D RoPE embeddings from latitude/longitude coordinates.

    Args:
        coords: Tensor of shape (..., 2) with coordinates in radians (lat, lon).
        dim: Head dimension to encode; must be divisible by 4.
        base: RoPE base frequency.

    Returns:
        Tuple of (cos, sin) tensors with shape (..., dim).
    """

    assert coords.shape[-1] == 2, (
        f"coords last dimension must be 2 (lat, lon); got {coords.shape[-1]}"
    )
    assert dim % 4 == 0, f"2D rotary embeddings require dim to be divisible by 4; got {dim}"

    # Split the rotary frequencies evenly between latitude and longitude to stay local to each cell.
    half_dim = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim, 2, device=coords.device, dtype=coords.dtype) / half_dim)
    )

    lat, lon = coords.unbind(dim=-1)
    freq_lat = lat.unsqueeze(-1) * inv_freq
    freq_lon = lon.unsqueeze(-1) * inv_freq

    freqs = torch.cat((freq_lat, freq_lon), dim=-1)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(emb)
    sin = torch.sin(emb)

    return cos, sin


####################################################################################################
def rotary_pos_emb_2d(q, k, coords, base=10000.0, unsqueeze_dim=1):
    """Convenience wrapper that builds 2D RoPE embeddings and applies them to q/k."""

    cos, sin = rotary_embedding_2d(coords, q.shape[-1], base=base)
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


# Spherical RoPE
def _max_supported_spherical_band(dim_embed: int, num_heads: int) -> int:
    head_dim = dim_embed // num_heads
    max_complex = (head_dim - (head_dim % 2)) // 2
    return max(0, (max_complex - 1) // 2)


ROPE_MODES = ("none", "2d", "spherical", "spherical_unitary")


# Attention modules that can apply RoPE: the encoder's query aggregation engine, the global
# assimilation engine and the forecast engine.
ROPE_MODULES = ("aggregation", "global", "forecast")


def get_rope_modules(cf) -> tuple[str, ...]:
    """Modules that apply RoPE (config key rope_modules: "all" or e.g. "forecast" or
    "global|forecast"). Restricting RoPE to some modules also restricts the head-dimension
    capacity checks to those modules."""

    spec = cf.get("rope_modules", "all") or "all"
    if spec == "all":
        return ROPE_MODULES
    names = spec.split("|") if isinstance(spec, str) else list(spec)
    names = [str(n).strip() for n in names if str(n).strip()]
    unknown = sorted(set(names) - set(ROPE_MODULES))
    if unknown or not names:
        raise ValueError(
            f"Unsupported rope_modules={spec!r}; use 'all' or '|'-separated names from "
            f"{', '.join(ROPE_MODULES)}."
        )
    return tuple(m for m in ROPE_MODULES if m in names)


def _rope_modules_restricted(cf) -> bool:
    return get_rope_modules(cf) != ROPE_MODULES


def get_rope_mode(cf, logger=None, module: str | None = None) -> str:
    """Resolve RoPE mode, including temporary backwards compatibility for rope_2D.

    With `module` (one of ROPE_MODULES), returns "none" when rope_modules excludes it.
    """

    rope_mode = cf.get("rope_mode", "none") or "none"
    rope_2d = cf.get("rope_2D", None)
    if rope_2d is not None:
        if logger is not None:
            logger.warning(
                "Config key 'rope_2D' is deprecated and will be removed. Use 'rope_mode' "
                "with one of: %s.",
                ", ".join(ROPE_MODES),
            )
        if rope_mode == "none":
            rope_mode = "2d" if rope_2d else "none"
    if rope_mode not in ROPE_MODES:
        raise ValueError(
            f"Unsupported rope_mode={rope_mode!r}; expected one of: {', '.join(ROPE_MODES)}."
        )
    if module is not None:
        if module not in ROPE_MODULES:
            raise ValueError(f"Unknown RoPE module {module!r}; expected one of {ROPE_MODULES}.")
        if module not in get_rope_modules(cf):
            return "none"
    return rope_mode


def _restricted_head_candidates(cf, max_band_fn) -> list[int]:
    """Band capacities of the attention modules selected by rope_modules (blocks > 0)."""
    mods = get_rope_modules(cf)
    dim = cf.ae_global_dim_embed
    candidates = []
    if "aggregation" in mods and cf.get("ae_aggregation_num_blocks", 0) > 0:
        candidates.append(max_band_fn(dim, cf.ae_aggregation_num_heads))
    if "global" in mods and cf.get("ae_global_num_blocks", 0) > 0:
        candidates.append(max_band_fn(dim, cf.ae_global_num_heads))
    if "forecast" in mods and cf.get("fe_num_blocks", 0) > 0:
        candidates.append(max_band_fn(dim, cf.fe_num_heads))
    if not candidates:
        raise ValueError(f"rope_modules={cf.get('rope_modules')!r} selects no attention blocks.")
    return candidates


def _spherical_band_capacity(cf) -> int:
    """Largest band supported by every attention module that actually applies spherical RoPE.

    Unlike the automatic band selection below, aggregation heads only constrain the
    capacity when aggregation blocks exist (with 0 blocks no rope is applied there).
    """
    if _rope_modules_restricted(cf):
        return min(_restricted_head_candidates(cf, _max_supported_spherical_band))
    candidates = [_max_supported_spherical_band(cf.ae_global_dim_embed, cf.ae_global_num_heads)]
    if cf.get("ae_aggregation_num_blocks", 0) > 0:
        candidates.append(
            _max_supported_spherical_band(cf.ae_global_dim_embed, cf.ae_aggregation_num_heads)
        )
    if cf.get("fe_num_blocks", 0) > 0:
        candidates.append(_max_supported_spherical_band(cf.ae_global_dim_embed, cf.fe_num_heads))
    return min(candidates)


def get_rope_spherical_band(cf) -> int:
    """Resolve spherical band index, supporting explicit config or automatic selection."""

    rope_spherical_band = cf.get("rope_spherical_band", None)
    if rope_spherical_band is not None:
        band = int(rope_spherical_band)
        if band < 1:
            raise ValueError(
                f"spherical RoPE requires band l >= 1 (got rope_spherical_band={band})."
            )
        supported = _spherical_band_capacity(cf)
        if band > supported:
            raise ValueError(
                f"rope_spherical_band={band} exceeds the largest band supported by the "
                f"smallest attention head dimension that applies spherical RoPE "
                f"(l={supported}): band l occupies 2*(2l+1) head dimensions."
            )
        return band

    # Automatic selection. Note the aggregation heads clamp the band even when
    # ae_aggregation_num_blocks == 0 -- kept as-is for parity with archived runs that
    # resolved their band under this rule.
    if _rope_modules_restricted(cf):
        candidates = _restricted_head_candidates(cf, _max_supported_spherical_band)
    else:
        candidates = [
            _max_supported_spherical_band(cf.ae_global_dim_embed, cf.ae_aggregation_num_heads),
            _max_supported_spherical_band(cf.ae_global_dim_embed, cf.ae_global_num_heads),
        ]
        if cf.get("fe_num_blocks", 0) > 0:
            candidates.append(
                _max_supported_spherical_band(cf.ae_global_dim_embed, cf.fe_num_heads)
            )
    band = min(candidates)
    if band < 1:
        raise ValueError(
            "spherical RoPE requires band l >= 1, but the smallest attention head dimension "
            f"only supports l={band}. Increase the head dimension or set rope_mode: none."
        )
    return band


def apply_rope(qs, ks, coords, rope_mode, unsqueeze_dim):
    rope_mode = rope_mode or "none"
    if rope_mode == "none":
        return qs, ks
    if coords is None:
        raise ValueError(f"coords must be provided when rope_mode={rope_mode}")
    if rope_mode == "2d":
        return rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=unsqueeze_dim)
    if rope_mode == "spherical":
        return rotary_pos_emb_spherical(qs, ks, coords, unsqueeze_dim=unsqueeze_dim)
    if rope_mode == "spherical_unitary":
        return rotary_pos_emb_spherical_unitary(qs, ks, coords, unsqueeze_dim=unsqueeze_dim)
    raise ValueError(f"Unsupported rope_mode={rope_mode}")


def rotary_pos_emb_spherical(
    q: torch.Tensor,
    k: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    unsqueeze_dim: int = 1,
):
    """Apply spherical-harmonic RoPE-style modulation to q/k using precomputed coefficients.

    Both q and k are multiplied by Y_lm(omega) at their respective positions. Under the real-pair
    representation of complex modes, the attention dot product is equivalent to
    Re[sum_m Y_lm(omega_r) Y_lm*(omega_s) q_m k_m*].
    """

    coeff_real, coeff_imag = coeffs
    return (
        _apply_complex_modulation(q, coeff_real, coeff_imag, unsqueeze_dim),
        _apply_complex_modulation(k, coeff_real, coeff_imag, unsqueeze_dim),
    )


def _apply_complex_modulation(
    x: torch.Tensor,
    coeff_real: torch.Tensor,
    coeff_imag: torch.Tensor,
    unsqueeze_dim: int,
) -> torch.Tensor:
    coeff_real = coeff_real.unsqueeze(unsqueeze_dim)
    coeff_imag = coeff_imag.unsqueeze(unsqueeze_dim)
    num_complex = coeff_real.shape[-1]
    max_complex = (x.shape[-1] - (x.shape[-1] % 2)) // 2
    if num_complex > max_complex:
        raise ValueError(
            f"Spherical RoPE requires {num_complex} complex modes but the head only supports "
            f"{max_complex}. Reduce rope_spherical_band or increase the head dimension."
        )
    num_rotary_dims = 2 * num_complex
    if num_rotary_dims == 0:
        return x

    # Cast the coefficients to x's dtype (deterministically, not the buffer's possibly
    # FSDP-cast dtype): the modulation runs under activation checkpointing, whose recompute
    # asserts identical tensor dtypes. The accuracy of the coefficients is set at build time
    # (sqrt(4*pi) normalization in float); the runtime modulation matches the bf16 q/k that
    # flash attention consumes anyway.
    coeff_real = coeff_real.to(dtype=x.dtype)
    coeff_imag = coeff_imag.to(dtype=x.dtype)
    x_rot = x[..., :num_rotary_dims].reshape(*x.shape[:-1], num_complex, 2)
    x_real = x_rot[..., 0]
    x_imag = x_rot[..., 1]
    out_real = (x_real * coeff_real) - (x_imag * coeff_imag)
    out_imag = (x_real * coeff_imag) + (x_imag * coeff_real)
    out = torch.stack((out_real, out_imag), dim=-1).flatten(-2, -1)
    if num_rotary_dims < x.shape[-1]:
        out = torch.cat((out, x[..., num_rotary_dims:]), dim=-1)
    return out


def build_spherical_rope_coeff_tensors(
    nside: int,
    band: int,
    num_local_queries: int,
    num_extra_tokens: int,
    amp_power: float = 1.0,
    rms_isometric: bool = True,
    device=None,
    dtype=torch.float32,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Build spherical-RoPE coefficient tensors for cell-level, extra tokens, and packed tokens.

    The orthonormal harmonics are rescaled by sqrt(4*pi) so that, at every point omega,
    mean_m |Y_lm(omega)|^2 = 1 (constant by Unsoeld's theorem). The modulation then preserves
    the RMS of isotropic q/k vectors and the attention logit scale matches rope_mode none/2d,
    making a post-modulation q/k norm unnecessary. The extra (register/class) tokens already
    use unit-magnitude coefficients and are consistent with this convention.

    amp_power (gamma) optionally compresses the per-mode amplitude profile |Y|^gamma (phase
    kept, per-pixel total energy renormalized to 2l+1). gamma=1 keeps the exact
    addition-theorem kernel; gamma->0 approaches phase-only (longitude-only) modulation.
    """

    if not rms_isometric and amp_power != 1.0:
        raise ValueError(
            "rope_spherical_rms_isometric=False is only supported with "
            "rope_spherical_amp_power=1.0: the amplitude compression renormalizes the "
            "per-pixel energy to 2l+1, which silently restores the 4*pi convention and "
            "overrides the orthonormal scale requested by rms_isometric=False."
        )

    real_maps, imag_maps = _healpy_band_maps(nside, band)
    # RMS-isometric normalization (4*pi convention); avoid mutating the lru_cached arrays.
    # rms_isometric=False keeps the orthonormal-harmonic scale (per-mode RMS 1/sqrt(4*pi)):
    # the modulated head-dim slice then enters attention ~sqrt(4*pi) weaker relative to the
    # unmodulated pass-through dims, faithfully reproducing the original post-norm scheme
    # (PR #2082). Note the constant is NOT absorbed by rope_spherical_post_norm, because it
    # scales only the rotary slice, not the full head vector the post-norm normalizes over.
    rms_scale = math.sqrt(4.0 * math.pi) if rms_isometric else 1.0
    real_maps = real_maps * rms_scale
    imag_maps = imag_maps * rms_scale
    if amp_power != 1.0:
        mag = np.hypot(real_maps, imag_maps)
        # guard mag==0: 0**0 == 1 would count vanished modes in the energy normalization
        new_mag = np.where(mag > 0.0, mag**amp_power, 0.0)
        scale = np.sqrt((2 * band + 1) / np.sum(new_mag**2, axis=-1, keepdims=True))
        ratio = np.where(mag > 0.0, new_mag * scale / np.where(mag > 0.0, mag, 1.0), 0.0)
        real_maps = real_maps * ratio
        imag_maps = imag_maps * ratio
    cell_real = torch.as_tensor(real_maps, device=device, dtype=dtype)
    cell_imag = torch.as_tensor(imag_maps, device=device, dtype=dtype)

    extra_real = torch.ones(
        num_extra_tokens, cell_real.shape[-1], device=cell_real.device, dtype=cell_real.dtype
    )
    extra_imag = torch.zeros_like(extra_real)
    packed_extra_real = (
        extra_real.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)
    )
    packed_extra_imag = (
        extra_imag.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)
    )

    packed_real = cell_real.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)
    packed_imag = cell_imag.unsqueeze(1).repeat(1, num_local_queries, 1).flatten(0, 1).unsqueeze(0)

    return (
        (cell_real, cell_imag),
        (extra_real, extra_imag),
        (packed_extra_real, packed_extra_imag),
        (packed_real, packed_imag),
    )


@lru_cache(maxsize=32)
def _healpy_band_maps(
    nside: int, band: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Precompute one spherical-harmonic band on the HEALPix grid using healpy.

    The returned columns store the complex coefficients Y_lm(omega) for fixed l=band and
    m=-l,...,+l. These are the position factors used in spherical RoPE:

        q_m^omega = Y_lm(omega) q_m,    k_m^omega = Y_lm(omega) k_m.

    The following attention dot product then implicitly forms
    Y_lm(omega_r) Y_lm*(omega_s), matching the spherical harmonics addition-theorem
    structure.
    """
    hp = _import_healpy()

    num_pixels = hp.nside2npix(nside)
    real_maps = np.zeros((num_pixels, 2 * band + 1), dtype=np.float64)
    imag_maps = np.zeros((num_pixels, 2 * band + 1), dtype=np.float64)
    alm_size = hp.sphtfunc.Alm.getsize(band, band)

    for m in range(0, band + 1):
        # healpy stores alm only for m >= 0 and alm2map reconstructs a real field. Setting
        # a_lm=1 gives 2 Re[Y_lm] for m>0, while a_lm=i gives -2 Im[Y_lm]. We combine these
        # two real maps below to recover the complex coefficient Y_lm itself.
        alm_real = np.zeros(alm_size, dtype=np.complex128)
        alm_real[hp.sphtfunc.Alm.getidx(band, band, m)] = 1.0
        real_map = hp.alm2map(alm_real, nside=nside, lmax=band, mmax=band, pol=False)
        real_map = hp.reorder(real_map, r2n=True)

        if m == 0:
            # Y_l0 is real, and healpy returns it directly because there is no -m counterpart
            # to merge into the real map.
            real_maps[:, band] = real_map
            continue

        alm_imag = np.zeros(alm_size, dtype=np.complex128)
        alm_imag[hp.sphtfunc.Alm.getidx(band, band, m)] = 1.0j
        imag_map = hp.alm2map(alm_imag, nside=nside, lmax=band, mmax=band, pol=False)
        imag_map = hp.reorder(imag_map, r2n=True)

        pos_idx = band + m
        neg_idx = band - m
        sign = -1.0 if m % 2 else 1.0

        # Columns are ordered as m=-l,...,+l, hence band+m for +m and band-m for -m.
        # The negative-order mode follows the standard convention
        # Y_l,-m = (-1)^m Y_lm*.
        real_maps[:, pos_idx] = real_map / 2.0
        imag_maps[:, pos_idx] = -imag_map / 2.0
        real_maps[:, neg_idx] = sign * real_map / 2.0
        imag_maps[:, neg_idx] = sign * imag_map / 2.0

    return real_maps, imag_maps


####################################################################################################
# Spherical unitary RoPE: multi-band real Wigner-D rotations.
#
# rope_mode "spherical" modulates q/k with one column of the degree-l rotation matrix (the
# spherical harmonics themselves), which is not norm preserving. Here q/k are instead rotated
# with the full real orthogonal representation matrices D^l(g_omega), g_omega = R_z(phi) R_y(theta),
# the exact spherical analogue of the 1D RoPE phases:
#   - exact isometry (orthogonal matrices), so no normalization is needed at all,
#   - exact relative-position kernel: q^T D^l(g_r)^T D^l(g_s) k = q^T D^l(g_r^{-1} g_s) k,
#   - multiple bands l=1..L give a multi-scale kernel, mirroring RoPE's frequency spectrum
#     (a single band's zonal kernel P_l oscillates; the band mix forms a decaying envelope).
# The "spherical" coefficients are proportional to the m'=0 column of these matrices
# (sqrt(2l+1) times it, with the RMS-isometric normalization), so this is a strict
# generalization. Choosing g_omega fixes a gauge (rotations about omega are a residual
# freedom); the meridian convention R_z(phi) R_y(theta) is used consistently everywhere.


def _real_sph_harm_band(band: int, theta, phi) -> npt.NDArray[np.float64]:
    """Real orthonormal spherical harmonics of degree l=band, columns ordered m=-l..l.

    Conventions only need to be internally consistent: the Wigner-D matrices are built and
    applied in this same basis, and all kernel properties are basis independent.
    """
    from scipy.special import lpmv  # noqa: PLC0415

    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    ct = np.cos(theta)
    out = np.zeros((theta.shape[0], 2 * band + 1), dtype=np.float64)
    for m in range(0, band + 1):
        # orthonormal normalization; lpmv includes the Condon-Shortley phase
        log_norm = 0.5 * (
            math.log((2 * band + 1) / (4.0 * math.pi))
            + math.lgamma(band - m + 1)
            - math.lgamma(band + m + 1)
        )
        p = lpmv(m, band, ct) * math.exp(log_norm)
        if m == 0:
            out[:, band] = p
        else:
            out[:, band + m] = math.sqrt(2.0) * p * np.cos(m * phi)
            out[:, band - m] = math.sqrt(2.0) * p * np.sin(m * phi)
    return out


def _wigner_z_rotation_blocks(band: int, phi) -> npt.NDArray[np.float64]:
    """Real-basis representation matrices of z-rotations, D^l(R_z(phi)).

    Acting on functions by (T(R) f)(x) = f(R^{-1} x), a z-rotation mixes the (m, -m) pair of
    real harmonics by a plane rotation with angle m*phi (the classical RoPE block structure).
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    num_modes = 2 * band + 1
    mats = np.zeros((phi.shape[0], num_modes, num_modes), dtype=np.float64)
    mats[:, band, band] = 1.0
    for m in range(1, band + 1):
        c, s = np.cos(m * phi), np.sin(m * phi)
        mats[:, band + m, band + m] = c
        mats[:, band - m, band + m] = s
        mats[:, band + m, band - m] = -s
        mats[:, band - m, band - m] = c
    return mats


@lru_cache(maxsize=32)
def _healpix_wigner_band_mats(nside: int, band: int) -> npt.NDArray[np.float64]:
    """Per-cell real Wigner-D matrices D^l(R_z(phi_c) R_y(theta_c)) on the HEALPix grid.

    The y-rotation factor d^l(theta) is obtained per iso-latitude ring (HEALPix has only
    4*nside-1 of them) by solving Y_j(R_y(-theta) u_k) = sum_i Y_i(u_k) d_ij on a fixed
    sample point set u_k; the degree-l space is rotation invariant, so the system is exact,
    and an SVD projection removes the residual least-squares noise to give an exactly
    orthogonal matrix. The z-rotation factor is analytic.
    """
    hp = _import_healpy()
    num_pixels = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(num_pixels), nest=True)

    # fixed sample directions for the fit, K >> 2l+1 for conditioning
    sample_nside = 8
    u = np.stack(hp.pix2vec(sample_nside, np.arange(hp.nside2npix(sample_nside))), axis=-1)
    theta_u = np.arccos(np.clip(u[:, 2], -1.0, 1.0))
    phi_u = np.arctan2(u[:, 1], u[:, 0])
    a = _real_sph_harm_band(band, theta_u, phi_u)
    pinv_a = np.linalg.pinv(a)

    ring_theta, ring_idx = np.unique(np.round(theta, 12), return_inverse=True)
    d_rings = np.zeros((ring_theta.shape[0], 2 * band + 1, 2 * band + 1), dtype=np.float64)
    for r, th in enumerate(ring_theta):
        ct, st = math.cos(th), math.sin(th)
        ry_inv = np.array([[ct, 0.0, -st], [0.0, 1.0, 0.0], [st, 0.0, ct]], dtype=np.float64)
        u_rot = u @ ry_inv.T
        theta_r = np.arccos(np.clip(u_rot[:, 2], -1.0, 1.0))
        phi_r = np.arctan2(u_rot[:, 1], u_rot[:, 0])
        d = pinv_a @ _real_sph_harm_band(band, theta_r, phi_r)
        w, _, vt = np.linalg.svd(d)
        d_rings[r] = w @ vt
    dz = _wigner_z_rotation_blocks(band, phi)
    return np.einsum("cij,cjk->cik", dz, d_rings[ring_idx])


def _max_supported_unitary_band(dim_embed: int, num_heads: int) -> int:
    head_dim = dim_embed // num_heads
    # bands l=1..L occupy sum_{l=1}^{L} (2l+1) = L(L+2) head dimensions
    return max(0, int(math.isqrt(head_dim + 1)) - 1)


def get_rope_unitary_bands(cf) -> list[int]:
    """Bands l=1..L used by spherical_unitary RoPE, fitting the smallest head dimension."""
    if _rope_modules_restricted(cf):
        candidates = _restricted_head_candidates(cf, _max_supported_unitary_band)
    else:
        candidates = [
            _max_supported_unitary_band(cf.ae_global_dim_embed, cf.ae_aggregation_num_heads),
            _max_supported_unitary_band(cf.ae_global_dim_embed, cf.ae_global_num_heads),
        ]
        if cf.get("fe_num_blocks", 0) > 0:
            candidates.append(_max_supported_unitary_band(cf.ae_global_dim_embed, cf.fe_num_heads))
    supported = min(candidates)

    max_band = cf.get("rope_unitary_max_band", None)
    if max_band is None:
        max_band = supported
    else:
        max_band = int(max_band)
        if max_band > supported:
            raise ValueError(
                f"rope_unitary_max_band={max_band} exceeds the largest band supported by the "
                f"smallest attention head dimension (L={supported}): bands l=1..L occupy "
                f"L*(L+2) head dimensions."
            )
    if max_band < 1:
        raise ValueError(
            f"spherical_unitary RoPE requires at least band l=1 (got L={max_band}); "
            f"the smallest attention head dimension supports L={supported}."
        )
    return list(range(1, max_band + 1))


def build_spherical_unitary_rope_tensors(
    nside: int,
    bands: list[int] | tuple[int, ...],
    num_local_queries: int,
    num_extra_tokens: int,
    device=None,
    dtype=torch.float32,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Build per-band Wigner-D tensors: (cell_mats, extra_mats, packed_mats).

    cell_mats[i] has shape (num_cells, 2l+1, 2l+1); extra_mats[i] are identities for the
    register/class tokens; packed_mats[i] interleave extras first and cells second, each
    repeated num_local_queries times, matching the packed order of rope_spherical_coeffs.
    """
    cell_mats, extra_mats, packed_mats = [], [], []
    for band in bands:
        mats = torch.as_tensor(_healpix_wigner_band_mats(nside, band), device=device, dtype=dtype)
        num_modes = mats.shape[-1]
        extra = (
            torch.eye(num_modes, device=mats.device, dtype=mats.dtype)
            .unsqueeze(0)
            .repeat(max(num_extra_tokens, 0), 1, 1)
        )
        packed = torch.cat(
            (
                extra.repeat_interleave(num_local_queries, dim=0),
                mats.repeat_interleave(num_local_queries, dim=0),
            ),
            dim=0,
        )
        cell_mats.append(mats)
        extra_mats.append(extra)
        packed_mats.append(packed)
    return cell_mats, extra_mats, packed_mats


def rotary_pos_emb_spherical_unitary(q, k, band_mats, unsqueeze_dim: int = 1):
    """Rotate q/k with per-token block-diagonal real Wigner-D matrices (exact isometry)."""
    return (
        _apply_band_rotations(q, band_mats, unsqueeze_dim),
        _apply_band_rotations(k, band_mats, unsqueeze_dim),
    )


def _token_dim_from_layout(x: torch.Tensor, unsqueeze_dim: int) -> int:
    # The diagonal (spherical) modulation locates the token dimension purely via
    # broadcasting; these are the three q/k layouts used by the attention classes.
    if x.dim() == 3 and unsqueeze_dim == 1:
        return 0  # varlen: (tokens, heads, head_dim)
    if x.dim() == 4 and unsqueeze_dim == 1:
        return 2  # local: (batch, heads, tokens, head_dim)
    if x.dim() == 4 and unsqueeze_dim == 2:
        return 1  # global/forecast: (batch, tokens, heads, head_dim)
    raise ValueError(
        f"Unsupported q/k layout for spherical_unitary RoPE: "
        f"dim={x.dim()}, unsqueeze_dim={unsqueeze_dim}"
    )


def _apply_band_rotations(x, band_mats, unsqueeze_dim):
    token_dim = _token_dim_from_layout(x, unsqueeze_dim)
    num_rotary_dims = sum(mats.shape[-1] for mats in band_mats)
    if num_rotary_dims > x.shape[-1]:
        raise ValueError(
            f"spherical_unitary RoPE requires {num_rotary_dims} head dimensions but only "
            f"{x.shape[-1]} are available. Reduce rope_unitary_max_band."
        )
    offset = 0
    pieces = []
    for mats in band_mats:
        while mats.dim() > 3:
            mats = mats.squeeze(0)
        if mats.shape[0] != x.shape[token_dim]:
            raise ValueError(
                f"spherical_unitary RoPE token mismatch: {mats.shape[0]} matrices vs "
                f"{x.shape[token_dim]} tokens (dim {token_dim})."
            )
        num_modes = mats.shape[-1]
        xs = x[..., offset : offset + num_modes]
        # compute in the (float32) matrix dtype, cast back afterwards
        # NOTE: do not disable autocast around the einsum. Under the training stack the
        # rope buffers reach the checkpointed forward as FSDP bf16 casts but the backward
        # recompute reads them as fp32; the ambient autocast normalizes the einsum operands
        # to bf16 in both passes, which is what keeps the activation-checkpoint recompute
        # metadata check happy (dtype divergence here raises CheckpointError in training).
        compute_dtype = torch.promote_types(x.dtype, mats.dtype)
        mats_c = mats.to(compute_dtype)
        xs_c = xs.to(compute_dtype)
        if token_dim == x.dim() - 2:
            out = torch.einsum("tij,...tj->...ti", mats_c, xs_c)
        else:
            out = torch.einsum("tij,...thj->...thi", mats_c, xs_c)
        pieces.append(out.to(x.dtype))
        offset += num_modes
    if offset < x.shape[-1]:
        pieces.append(x[..., offset:])
    return torch.cat(pieces, dim=-1)
