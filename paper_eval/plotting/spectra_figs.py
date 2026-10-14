"""f09: wavelength spectra (target vs. prediction) from `spectra.nc`.

Plan D5's spectral-density conversion, from cumulative E over ascending wavelength
lambda: `k = 2*pi/lambda`; `dE = E[i+1] - E[i]`; `dk = k[i+1] - k[i]` (negative since
lambda is ascending, so k is descending); `S = -dE/dk` (>= 0); plotted vs.
`lambda_mid = sqrt(lambda_i * lambda_{i+1})`, log-log.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

from paper_eval.config import fig_dir
from paper_eval.meshtools import parse_channel
from paper_eval.plotting import style
from paper_eval.plotting.fesom_maps import get_mesh

#: Vertical reference line: roughly 2x the CORE2 mesh's characteristic resolution,
#: below which filtered spectra become unreliable (see `paper_eval/spectra.py`).
_MESH_SCALE_KM = 200.0

#: Raw scale samples dropped from the *large-wavelength* (coarse) end before
#: differencing into a spectral density. `spectra.py`'s own `validate_cumulative`
#: flags the largest configured scale (5000 km, the coarsest end of the filter
#: range) as "beyond mesh validity" / non-monotonic for several individual
#: prediction samples (see spectra.nc's `warnings` attribute); `spectra.nc` stores
#: the sample-averaged cumulative spectrum, so that per-sample corruption survives
#: averaging and additionally drags the *second*-to-last raw point off the smooth
#: trend (checked directly against the dev cache: for sst/ssh "prediction", E jumps
#: by 10-20x at the second-to-last point and by many orders of magnitude at the
#: last one, versus a steady ~1.3-1.4x per-step growth everywhere else). Dropping
#: just the last raw sample still leaves that second, contaminated jump behind, so
#: both are dropped; the resulting S = -dE/dk curve then decays smoothly all the
#: way to its new endpoint (~3150 km) for every channel checked. Display-side only
#: -- spectra.nc itself is never modified.
_DROP_COARSE_SAMPLES = 2

_KE_PATTERN = re.compile(r"^KE_([a-zA-Z]+)_(\d+)_([a-zA-Z]+)_(\d+)$")


def _spectral_density(cum_spectra_1d: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """`(lambda_mid, S)` from one cumulative spectrum over ascending `scale` (km).

    Drops the coarsest `_DROP_COARSE_SAMPLES` raw scale samples first -- see its
    docstring for why; a no-op if that would leave fewer than 2 samples to diff.
    """
    lam = cum_spectra_1d["scale"].values.astype(np.float64)
    e = cum_spectra_1d.values.astype(np.float64)
    if 0 < _DROP_COARSE_SAMPLES < lam.size - 1:
        lam = lam[:-_DROP_COARSE_SAMPLES]
        e = e[:-_DROP_COARSE_SAMPLES]
    k = 2.0 * np.pi / lam
    d_e = np.diff(e)
    d_k = np.diff(k)
    s = -d_e / d_k
    lam_mid = np.sqrt(lam[:-1] * lam[1:])
    return lam_mid, s


def _plot_spectral_curve(ax, lam_mid, s, *, color, label, linewidth=1.4, zorder=2) -> None:
    """Plot one spectral-density curve, hard-trimmed at the mesh's trustworthy
    resolution (`_MESH_SCALE_KM`): wavelengths below the cutoff are simply not
    plotted at all (Task 18) -- the native-mesh filter is not trusted there, so
    nothing is drawn for that range rather than a faded/de-emphasized guess (unlike
    the coarse-end artifact, which is dropped outright upstream in
    `_spectral_density` for an unrelated reason)."""
    keep = lam_mid >= _MESH_SCALE_KM
    ax.plot(
        lam_mid[keep], s[keep], color=color, linestyle="-", linewidth=linewidth,
        label=label, zorder=zorder,
    )


def _channel_title(channel: str, nz1) -> str:
    m = _KE_PATTERN.match(channel)
    if m:
        u_name, u_suf, v_name, _v_suf = m.groups()
        try:
            _, depth, _ = parse_channel(f"{u_name}_{u_suf}", nz1)
            return f"KE ({u_name}, {v_name} @ {depth:g} m)"
        except ValueError:
            return channel
    return style.channel_label(channel, nz1)


def build_f09(cfg: dict, cache_paths: dict) -> list[Path]:
    path = cache_paths.get("spectra")
    if path is None or not Path(path).exists():
        print("[skip] f09: missing cache input 'spectra'")
        return []

    ds = xr.open_dataset(path)
    mesh = get_mesh(cfg["mesh_path"])

    channels = list(ds.channel.values)
    bands = list(ds.band.values)
    leads = list(ds.lead.values)
    if not channels or not leads:
        print("[skip] f09: spectra.nc has no channels/leads")
        return []

    n_bands, n_channels = len(bands), len(channels)
    figsize = (4.4 * n_channels, 3.8 * n_bands)
    fig, axes = plt.subplots(
        n_bands, n_channels, figsize=figsize, squeeze=False, constrained_layout=True
    )

    for bi, band in enumerate(bands):
        for ci, channel in enumerate(channels):
            ax = axes[bi, ci]
            da = ds["cum_spectra"].sel(band=band, channel=channel)

            # Single ground-truth curve (Task 23): the per-lead target spectra are
            # nominally lead-independent (same truth field; only the paired
            # prediction sample changes with lead) and empirically near-identical
            # -- plotting both separately (pre-Task-23) drew two near-coincident
            # black lines, which read as visual clutter rather than a meaningful
            # comparison. Averaged here across `lead` in cumulative (E) space,
            # before the coarse-bin drop + differencing inside `_spectral_density`.
            # That choice (vs. averaging the per-lead S curves after differencing)
            # is arbitrary but not lossy: S = -dE/dk is linear in E, and every lead
            # shares the same `scale` (hence `k`) grid, so diff-of-mean and
            # mean-of-diff are numerically identical here; cumulative space just
            # lets this reuse `_spectral_density` unchanged (one diff instead of
            # `len(leads)` diffs + an average).
            target_mean = da.sel(kind="target").mean(dim="lead")
            lam_mid, s = _spectral_density(target_mean)
            _plot_spectral_curve(
                ax, lam_mid, s, color="black", label="ground truth", linewidth=1.4, zorder=3,
            )
            for lead in leads:
                lam_mid, s = _spectral_density(da.sel(kind="prediction", lead=lead))
                _plot_spectral_curve(
                    ax, lam_mid, s, color=style.lead_color(int(lead)),
                    label=f"prediction (lead {lead})", linewidth=1.4, zorder=2,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()  # decreasing wavelength to the right, consistently

            # Hard cutoff (Task 18): nothing below the mesh-trustworthy resolution
            # is plotted at all (see `_plot_spectral_curve`), and the axis itself
            # ends exactly at the cutoff -- pin only the fine-wavelength (right,
            # post-inversion) edge to `_MESH_SCALE_KM`; the coarse-wavelength
            # (left) edge stays whatever the curve data + its own autoscale margin
            # produced, unchanged from before (the coarse-end 2-bin drop in
            # `_spectral_density` is untouched).
            ax.set_xlim(right=_MESH_SCALE_KM)

            # Log-scale x-axis default (decade-only) major locator places a tick
            # only at powers of 10, so a ~1-decade span like this one (~200-3300
            # km) gets just a single labeled tick (10^3). Add per-decade
            # subdivisions instead, and format the labels as plain km integers
            # ("200") rather than exponents. Locators/formatters only affect tick
            # placement/labels, not data limits, so this is safe after the
            # xlim-pinning above. Minor ticks are suppressed -- unlabeled, and
            # visually noisy at this scale -- rather than left at their default
            # (2..9)*10^n placement. Same `subs` as before the Task-18 hard cutoff
            # (1/2/3/5 per decade): verified this still yields exactly 6 visible
            # labels (200/300/500/1000/2000/3000) over the now-reduced [200, ~3400
            # km] range, so no further thinning was needed.
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.0f}"))
            ax.xaxis.set_minor_locator(mticker.NullLocator())

            ax.set_xlabel(f"wavelength [km]  (resolved scales > {_MESH_SCALE_KM:.0f} km)")
            if ci == 0:
                ax.set_ylabel("spectral density [arb. units]")
            title = _channel_title(channel, mesh.nz1)
            if n_bands > 1:
                title = f"{title}  --  {band}"
            ax.set_title(title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Wavelength spectra: target vs. prediction")

    return style.savefig(fig, "f09_wavelength_spectra", fig_dir(cfg))
