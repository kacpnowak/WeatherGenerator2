"""CLI: compute native-mesh spectra (`implicit_filter`) for a time-spread subset of the
(init-filtered) samples.

    .venv/bin/python -m paper_eval.compute_spectra --config paper_eval/configs/default.yaml \\
        [--n-samples N] [--n-scales N] [--n-workers N]

Spectra are expensive (`paper_eval/spectra.py`: ~n_scales * 30-80s per job on a shared
CPU node), so this stage runs on a small, time-spread subset of samples rather than the
full scoring set, with process-level parallelism across jobs (one `compute_spectra_*`
call per worker task).

Each worker's first JAX computation creates its own CPU thread pool sized to the
process's CPU affinity (observed: ~750 threads/process on a 256-core node when
unrestricted -- `paper_eval.spectra`'s `XLA_FLAGS` fix only covers JAX's Eigen op
pool, not this one; confirmed directly, `pthread_create() failed` crashed a 2-worker
run here before this fix was added). `init_worker` therefore pins each worker to a
small, distinct slice of the visible CPUs before touching the filter, capping threads
per worker to a safe, roughly constant count regardless of `n_workers`.
"""

import argparse
import dataclasses
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

from paper_eval import spectra
from paper_eval.config import cache_dir, load_config
from paper_eval.merged_zarr import open_merged_ranks
from paper_eval.meshtools import load_mesh

logger = logging.getLogger(__name__)

_KINDS = ("prediction", "target")

#: Max CPUs pinned per worker (see `init_worker`) -- small and fixed rather than
#: `available // n_workers`, since on a node with far more cores than workers an even
#: split would still hand each worker hundreds of cores and reproduce the thread
#: exhaustion this pinning exists to avoid.
_MAX_CPUS_PER_WORKER = 4

# ── Worker-global state, populated once by init_worker ────────────────────────────────
# (ProcessPoolExecutor initializer pattern: one call per worker process; also callable
# directly, e.g. for a single-process run.)
_cfg = None
_root = None
_filt = None
_lam_km = None
_Kc = None
_band_masks: dict[str, np.ndarray | None] = {}
_channel_specs: dict[str, tuple[str, tuple[str, ...]]] = {}


def _build_channel_specs(spec_cfg: dict) -> dict[str, tuple[str, tuple[str, ...]]]:
    """spec_channel name -> ("scalar", (channel,)) for scalar_channels, or
    ("velocity", (u, v)) for each velocity pair, keyed by "KE_{u}_{v}"."""
    specs: dict[str, tuple[str, tuple[str, ...]]] = {
        ch: ("scalar", (ch,)) for ch in spec_cfg["scalar_channels"]
    }
    for u, v in spec_cfg["velocity_pairs"]:
        specs[f"KE_{u}_{v}"] = ("velocity", (u, v))
    return specs


def _pin_cpus(n_workers: int, slot_counter) -> None:
    """Pin this worker process to a small, distinct slice of the visible CPUs, so its
    first JAX computation (which sizes a thread pool off the process's CPU affinity --
    see module docstring) can't blow up the per-user thread/process ulimit.

    `slot_counter` is a `multiprocessing.Value("i", 0)` (with its own lock) shared by
    the whole pool: each worker atomically claims the next 0-based slot, so slices
    don't overlap (unlike a fixed CPU range, which would serialize every worker onto
    the same few cores instead of spreading them out).
    """
    with slot_counter.get_lock():
        slot = slot_counter.value
        slot_counter.value += 1

    available = sorted(os.sched_getaffinity(0))
    per_worker = max(1, min(len(available) // max(1, n_workers), _MAX_CPUS_PER_WORKER))
    start = (slot * per_worker) % len(available)
    my_cpus = {available[(start + i) % len(available)] for i in range(per_worker)}
    os.sched_setaffinity(0, my_cpus)


def init_worker(cfg: dict, n_workers: int, slot_counter) -> None:
    """Pin this worker's CPU affinity (`_pin_cpus`), then open the merged zarr root and
    load the `implicit_filter` FesomFilter (from cache), and precompute wavenumbers,
    band masks and channel specs, into module globals.

    Runs once per worker process; idempotent (safe to call again in the same process).
    A spawned worker is a fresh interpreter -- it does not inherit the parent's
    `logging.basicConfig` call, so `process_job`'s per-job log line would otherwise be
    silently dropped; configure logging here too.
    """
    global _cfg, _root, _filt, _lam_km, _Kc, _band_masks, _channel_specs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _pin_cpus(n_workers, slot_counter)

    _cfg = cfg
    _root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    spec_cfg = cfg["spectra"]
    _filt = spectra.load_filter(cfg["mesh_path"], spec_cfg["filter_cache"])
    _lam_km = spectra.scales_km(spec_cfg)
    _Kc = spectra.wavenumbers(_lam_km)

    mesh = load_mesh(cfg["mesh_path"])
    _band_masks = {
        name: spectra.band_node_mask(mesh.lat, band) for name, band in spec_cfg["bands"].items()
    }
    _channel_specs = _build_channel_specs(spec_cfg)


@dataclasses.dataclass
class JobResult:
    """Result of one (sample, lead, kind, spec_channel) job: a cumulative spectrum
    (length n_scales + 1, index 0 = unfiltered total) per configured band, plus any
    `validate_cumulative` warnings (with job context prepended)."""

    sample: int
    lead: int
    kind: str
    spec_channel: str
    band_spectra: dict[str, np.ndarray]
    warnings: list[str]
    elapsed: float


def process_job(job: tuple[int, int, str, str]) -> JobResult:
    """Compute one (sample, lead, kind, spec_channel) job's cumulative spectrum for
    every configured band (one `compute_spectra_*` call per band)."""
    if _root is None:
        raise RuntimeError("init_worker() must be called before process_job()")

    sample, lead, kind, spec_channel = job
    t0 = time.perf_counter()

    kind_type, cols = _channel_specs[spec_channel]
    order = _cfg["spectra"]["filter_order"]

    stream = "FESOM_NODE" if kind_type == "scalar" else "FESOM_ELEM"
    group = _root[f"{sample}/{stream}/{lead}/{kind}"]
    channels = list(group.attrs["channels"])
    arr = np.asarray(group["data"])
    if kind == "prediction":
        assert arr.shape[-1] == 1, f"expected ens==1, got {arr.shape[-1]}"
        arr = arr[..., 0]

    band_spectra: dict[str, np.ndarray] = {}
    warns: list[str] = []
    for band_name, mask in _band_masks.items():
        if kind_type == "scalar":
            (name,) = cols
            field = arr[:, channels.index(name)]
            spec = spectra.cumulative_spectra_scalar(_filt, field, _Kc, order, mask=mask)
        else:
            u_name, v_name = cols
            u_elem = arr[:, channels.index(u_name)]
            v_elem = arr[:, channels.index(v_name)]
            spec = spectra.cumulative_spectra_velocity(_filt, u_elem, v_elem, _Kc, order, mask=mask)
        band_spectra[band_name] = spec
        for w in spectra.validate_cumulative(spec, _lam_km):
            warns.append(
                f"sample={sample} lead={lead} kind={kind} channel={spec_channel} "
                f"band={band_name}: {w}"
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "job sample=%d lead=%d kind=%s channel=%s: %.1fs (%d band(s))",
        sample,
        lead,
        kind,
        spec_channel,
        elapsed,
        len(_band_masks),
    )
    return JobResult(sample, lead, kind, spec_channel, band_spectra, warns, elapsed)


def _select_samples(cfg: dict, root, n_samples: int) -> list[int]:
    """Time-spread subset of the init-filtered sample list: `n_samples` evenly spaced
    indices into `root.store.samples` (already time-sorted) after dropping samples
    outside `cfg["init_filter"]` -- the SAME rule as the scoring pass: start/end compare
    against `init_times`; a null bound is open."""
    samples_all = list(root.store.samples)
    init_times = root.store.init_times
    start = cfg["init_filter"].get("start")
    end = cfg["init_filter"].get("end")
    start_dt = np.datetime64(start) if start else None
    end_dt = np.datetime64(end) if end else None

    filtered = [
        s
        for s in samples_all
        if (start_dt is None or init_times[s] >= start_dt)
        and (end_dt is None or init_times[s] <= end_dt)
    ]
    if not filtered:
        raise ValueError(f"no samples survive init_filter {cfg['init_filter']!r}")

    n_samples = min(n_samples, len(filtered))
    pick = sorted(set(int(i) for i in np.linspace(0, len(filtered) - 1, num=n_samples).round()))
    selected = [filtered[i] for i in pick]
    logger.info(
        "samples: %d total, %d after init_filter, %d selected (time-spread), init %s..%s",
        len(samples_all),
        len(filtered),
        len(selected),
        init_times[selected[0]],
        init_times[selected[-1]],
    )
    return selected


def _reduce(
    results: list[JobResult],
    cfg: dict,
    lam_km: np.ndarray,
    samples_used: list[int],
    init_times_iso: list[str],
) -> xr.Dataset:
    """Mean over samples -> `cum_spectra(kind, lead, band, channel, scale)`, dropping
    each job's index-0 unfiltered total (kept only for `validate_cumulative`)."""
    spec_cfg = cfg["spectra"]
    leads = list(spec_cfg["leads"])
    bands = list(spec_cfg["bands"])
    channels = list(spec_cfg["scalar_channels"]) + [
        f"KE_{u}_{v}" for u, v in spec_cfg["velocity_pairs"]
    ]

    acc: dict[tuple[str, int, str, str], list[np.ndarray]] = {}
    warnings: list[str] = []
    for r in results:
        for band_name, spec in r.band_spectra.items():
            acc.setdefault((r.kind, r.lead, band_name, r.spec_channel), []).append(spec[1:])
        warnings.extend(r.warnings)

    n_scales = len(lam_km)
    data = np.empty(
        (len(_KINDS), len(leads), len(bands), len(channels), n_scales), dtype=np.float64
    )
    for ik, kind in enumerate(_KINDS):
        for il, lead in enumerate(leads):
            for ib, band in enumerate(bands):
                for ic, channel in enumerate(channels):
                    specs = acc.get((kind, lead, band, channel))
                    if not specs:
                        raise ValueError(
                            f"no results for kind={kind} lead={lead} band={band} channel={channel}"
                        )
                    data[ik, il, ib, ic, :] = np.mean(specs, axis=0)

    return xr.Dataset(
        {"cum_spectra": (("kind", "lead", "band", "channel", "scale"), data)},
        coords={
            "kind": list(_KINDS),
            "lead": leads,
            "band": bands,
            "channel": channels,
            "scale": lam_km,
            "wavelength_km": ("scale", lam_km),
        },
        attrs={
            "n_samples": len(samples_used),
            "sample_ids": [int(s) for s in samples_used],
            "init_times": init_times_iso,
            "filter_order": int(spec_cfg["filter_order"]),
            "scale_min_km": float(spec_cfg["scale_min_km"]),
            "scale_max_km": float(spec_cfg["scale_max_km"]),
            "n_scales": int(n_scales),
            "run_id": cfg["run_id"],
            "warnings": warnings,
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument(
        "--n-samples", type=int, default=None, help="Overrides cfg['spectra']['n_samples']."
    )
    parser.add_argument(
        "--n-scales", type=int, default=None, help="Overrides cfg['spectra']['n_scales']."
    )
    parser.add_argument(
        "--n-workers", type=int, default=None, help="Overrides cfg['spectra']['n_workers']."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    spec_cfg = cfg["spectra"]
    if args.n_samples is not None:
        spec_cfg["n_samples"] = args.n_samples
    if args.n_scales is not None:
        spec_cfg["n_scales"] = args.n_scales
    if args.n_workers is not None:
        spec_cfg["n_workers"] = args.n_workers

    t_start = time.perf_counter()

    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    samples_used = _select_samples(cfg, root, spec_cfg["n_samples"])
    init_times_iso = [str(root.store.init_times[s]) for s in samples_used]

    leads = list(spec_cfg["leads"])
    channel_specs = _build_channel_specs(spec_cfg)
    jobs = [
        (s, lead, kind, ch)
        for s in samples_used
        for lead in leads
        for kind in _KINDS
        for ch in channel_specs
    ]
    logger.info(
        "jobs: %d (samples=%d leads=%d kinds=2 channels=%d bands=%d)",
        len(jobs),
        len(samples_used),
        len(leads),
        len(channel_specs),
        len(spec_cfg["bands"]),
    )

    n_workers = max(1, min(spec_cfg["n_workers"], len(jobs)))
    ctx = multiprocessing.get_context("spawn")
    slot_counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(cfg, n_workers, slot_counter),
    ) as pool:
        results = list(pool.map(process_job, jobs))

    lam_km = spectra.scales_km(spec_cfg)
    ds = _reduce(results, cfg, lam_km, samples_used, init_times_iso)

    out_path = cache_dir(cfg) / "spectra.nc"
    ds.to_netcdf(out_path)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "wrote %s (%.1fs total, %d jobs, %d workers)", out_path, elapsed, len(jobs), n_workers
    )
    if ds.attrs["warnings"]:
        logger.warning("validate_cumulative warnings: %d", len(ds.attrs["warnings"]))
        for w in ds.attrs["warnings"]:
            logger.warning("  %s", w)


if __name__ == "__main__":
    main()
