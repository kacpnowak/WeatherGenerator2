"""Native-mesh spectra via `implicit_filter`.

Power (scalar channels) and kinetic-energy (velocity pairs) spectra are computed
directly on the FESOM CORE2 mesh with `implicit_filter.FesomFilter` -- no regridding,
no zonal FFT. (`paper_eval/regrid.py` implements the earlier regrid+FFT approach; it is
kept as an optional visual-check utility but is not part of this pipeline.)

`FesomFilter.prepare_from_file` assembles the mesh operator (~18s for CORE2); that cost
is paid once and cached to disk (`load_filter`) rather than repeated per process/job.
Each `compute_spectra_scalar`/`compute_spectra_velocity` call costs roughly
`n_scales * (30-80s)` on a shared CPU node -- callers (`compute_spectra.py`) are expected
to budget for that and parallelize across jobs, not scales.

On first use, JAX's CPU backend sizes (at least) one internal thread pool to the
visible CPU count (observed: several hundred threads on a 256-core node for a single
process) -- harmless for one interactive process, but a problem once several worker
processes exist at once, as `compute_spectra.py` does. Its own parallelism is
process-level (one job per worker), so per-process multithreading only adds
contention, not speed. This module disables JAX's Eigen op pool specifically (which
also measurably speeds up a single call -- likely thread-coordination overhead the
sparse solves here don't benefit from); it must happen before `implicit_filter` (and
therefore JAX) is imported, since the pool is sized on first use, not at import time.
Set `XLA_FLAGS` yourself beforehand to override.

That flag alone is NOT sufficient to keep several concurrent worker processes under a
per-user thread/process `ulimit` -- JAX's CPU *client* itself (`PjRtCpuClient`) sizes a
separate pool the same way, unaffected by this flag ("pthread_create() failed", crashed
a 2-worker run here before the fix). `compute_spectra.py`'s `init_worker` additionally
pins each worker to a small slice of CPUs before its first computation, which is what
actually keeps that second pool -- and therefore total thread count -- bounded.
"""

import logging
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

from implicit_filter import FesomFilter, transform_velocity_to_nodes  # noqa: E402

logger = logging.getLogger(__name__)


def load_filter(mesh_path: str | Path, cache_path: str | Path) -> FesomFilter:
    """Load a `FesomFilter` from `cache_path` if present, else prepare it from
    `mesh_path` (mesh operator assembly, ~18s for CORE2) and save it to `cache_path`
    for next time.

    `FesomFilter.save_to_file` appends `.npz` itself (`np.savez` semantics), so the
    stem (with any trailing `.npz` stripped) is what gets passed to it.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("loading FesomFilter from cache: %s", cache_path)
        return FesomFilter.load_from_file(str(cache_path))

    logger.info("no cached filter at %s; preparing from mesh %s", cache_path, mesh_path)
    f = FesomFilter()
    f.prepare_from_file(str(mesh_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stem = str(cache_path)
    if stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    f.save_to_file(stem)
    return f


def scales_km(spec_cfg: dict) -> np.ndarray:
    """Ascending filter wavelengths (km), logspace over
    `[scale_min_km, scale_max_km]` with `n_scales` points (per `cfg["spectra"]`)."""
    return np.logspace(
        np.log10(spec_cfg["scale_min_km"]), np.log10(spec_cfg["scale_max_km"]), spec_cfg["n_scales"]
    )


def wavenumbers(lam_km: np.ndarray) -> np.ndarray:
    """Filter wavenumbers `Kc = 2*pi / lambda`, same order as `lam_km` (descending,
    since `lam_km` is ascending) -- matches `implicit_filter`'s own examples."""
    return 2.0 * np.pi / np.asarray(lam_km)


def band_node_mask(node_lat: np.ndarray, band: list | None) -> np.ndarray | None:
    """Per-node boolean mask for `compute_spectra_*`'s `mask` argument (True =
    EXCLUDED from the spectra integration; does not affect the filtering itself).

    `band=None` -> None (global mesh, nothing excluded).
    `band=[south, north]` -> True for nodes OUTSIDE `[south, north]` (degrees lat).
    """
    if band is None:
        return None
    south, north = band
    lat = np.asarray(node_lat)
    return (lat < south) | (lat > north)


def cumulative_spectra_scalar(
    filt: FesomFilter,
    field: np.ndarray,
    Kc: np.ndarray,
    order: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cumulative scalar power spectrum on mesh nodes: length `len(Kc) + 1`, index 0
    is the unfiltered total, entries 1..n correspond to `Kc` order."""
    field = np.asarray(field, dtype=np.float64)
    spec = filt.compute_spectra_scalar(order, Kc, field, mask=mask)
    return np.asarray(spec, dtype=np.float64)


def cumulative_spectra_velocity(
    filt: FesomFilter,
    u_elem: np.ndarray,
    v_elem: np.ndarray,
    Kc: np.ndarray,
    order: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cumulative kinetic-energy spectrum from element-based velocity: interpolated to
    nodes first (`transform_velocity_to_nodes`), then filtered there. Length
    `len(Kc) + 1`, index 0 is the unfiltered total, entries 1..n correspond to `Kc`
    order."""
    u_elem = np.asarray(u_elem, dtype=np.float64)
    v_elem = np.asarray(v_elem, dtype=np.float64)
    un, vn = transform_velocity_to_nodes(u_elem, v_elem, filt)
    spec = filt.compute_spectra_velocity(order, Kc, un, vn, mask=mask)
    return np.asarray(spec, dtype=np.float64)


def validate_cumulative(spec: np.ndarray, lam_km: np.ndarray) -> list[str]:
    """Sanity warnings for one cumulative spectrum (never raises):

    - non-monotonic: `spec[1:]` should be non-decreasing over ascending `lam_km`
      (larger filter scales admit progressively more of the total variance/energy).
    - beyond-mesh-validity: any `spec[1:]` entry exceeding `1.05 * spec[0]` (the
      unfiltered total) signals filter breakdown at scales too coarse for the mesh
      (see `implicit_filter` docs); the offending wavelength(s) are named.
    """
    warnings: list[str] = []
    band = spec[1:]
    lam_km = np.asarray(lam_km)

    diffs = np.diff(band)
    bad = np.nonzero(diffs < 0)[0]
    if bad.size:
        offending = ", ".join(f"{lam_km[i + 1]:.0f}km" for i in bad)
        warnings.append(f"non-monotonic cumulative spectrum at scales: {offending}")

    limit = 1.05 * spec[0]
    over = np.nonzero(band > limit)[0]
    if over.size:
        offending = ", ".join(f"{lam_km[i]:.0f}km" for i in over)
        warnings.append(
            f"cumulative spectrum exceeds 1.05x unfiltered total at scales: {offending} "
            "(beyond mesh validity)"
        )
    return warnings
