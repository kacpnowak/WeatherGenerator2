"""CLI: damped-persistence baseline (Task 28) -- sufficient statistics for f01.

    .venv/bin/python -m paper_eval.compute_damped \\
        --config paper_eval/configs/oe3i7whz.yaml [--channels sst,sss,ssh] \\
        [--n-workers N] [--samples ...] [--leads ...]

Fourth reference forecast for figure f01, on top of the day-of-year climatology:

    xhat(t0 + L) = clim(doy(t0+L)) + alpha(L) * [x(t0) - clim(doy(t0))]

i.e. persist the initial anomaly a0 = x(t0) - clim(doy(t0)), damped by a lead-dependent
factor alpha(L). alpha = 1 is anomaly persistence, alpha = 0 is climatology.

Rather than re-reading the 1.1 TB result store once per candidate alpha, this stores
the four sufficient statistics of the area-weighted MSE of ANY alpha, accumulated over
wet points, per (region, sample, lead, channel):

    S00 = sum(w*a0*a0), S0L = sum(w*a0*aL), SLL = sum(w*aL*aL), W = sum(w)

with aL = x(t0+L) - clim(doy(t0+L)). Then for any alpha (chosen post hoc, e.g. fit per
lead as alpha*(L) = sum_samples(S0L) / sum_samples(S00)):

    MSE(alpha) = ( alpha^2 * S00 - 2*alpha*S0L + SLL ) / W

FESOM_NODE only (the only stream f01 needs): reuses `scores.py`'s worker-globals
pattern verbatim (`scores.init_worker` populates `scores._root`/`_mask`/`_region_w`/
`_clim`/`_src_idx`/`_src_mean`/`_src_std`/`_channels` for the "fesom" geometry branch,
including the `_load_norm_stats`-based source denormalization and by-name source/
target channel alignment) rather than re-deriving any of that; only the accumulation
loop below is new. FESOM_ELEM and the "eerie" geometry are out of scope (the brief:
EERIE would need the cell-gather machinery `scores._process_block_eerie` uses, not
implemented here).
"""

import argparse
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from paper_eval import scores
from paper_eval.compute import _select_samples
from paper_eval.config import cache_dir, load_config, parse_int_list
from paper_eval.merged_zarr import open_merged_ranks

logger = logging.getLogger(__name__)

#: Stream this run accumulates; set per invocation via --stream (worker global, so it
#: survives the spawn-pool boundary through `init_worker`'s cfg).
STREAM = "FESOM_NODE"

# ── Worker-global state, populated once by init_worker (own globals: which channels
#: this run scores and cfg["regions"]'s order; everything mesh/store/climatology-
#: related lives in `scores.py`'s own globals, populated via `scores.init_worker`) ────
_channels: list[str] = []
_regions: list[str] = []


def init_worker(cfg: dict, channels: list[str]) -> None:
    """Pool initializer: delegate to `scores.init_worker` for the mesh/store/
    climatology/normalization state (restricting `cfg["streams"]` to `[STREAM]` so
    FESOM_ELEM's mesh/climatology/normalization are never loaded -- this baseline
    only needs FESOM_NODE), then record the channel subset and region order this
    run accumulates.
    """
    global _channels, _regions, STREAM
    STREAM = cfg.get("_damped_stream", STREAM)
    _channels = list(channels)
    _regions = list(cfg["regions"])
    scores.init_worker({**cfg, "streams": [STREAM]})


def process_block(sample_ids: list[int]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate (S00, S0L, SLL, W), each `(n_regions, n_block_samples, n_leads,
    n_channels)` float64, for one block of global sample ids.

    `a0`/`aL` are computed from `scores.py`'s own denormalized source
    (`_src_idx`/`_src_mean`/`_src_std`) and target fields, `_clean`-ed against the
    fixed FESOM_NODE wet mask (`scores._mask[STREAM]`) exactly as `_process_block_fesom`
    cleans its own fields -- so a dry point's NaN fill cannot poison the weighted sums
    below, and `S00`/`S0L`/`SLL`/`W` reduce to the *same* per-(region, sample, lead,
    channel) weighted sums `score_block`'s climatology-model row accumulates (this is
    the exact identity the brief's `MSE(alpha=0) == cached climatology RMSE^2` check
    verifies).
    """
    root = scores._root
    channels_full = scores._channels[STREAM]
    mask_full = scores._mask[STREAM]  # (n_channels_full, npts) bool
    region_w_full = scores._region_w[STREAM]  # region -> (n_channels_full, npts) float32
    clim = scores._clim[STREAM]
    src_idx = scores._src_idx[STREAM]
    src_mean = scores._src_mean[STREAM]
    src_std = scores._src_std[STREAM]
    leads = scores._leads

    sel = [channels_full.index(c) for c in _channels]
    region_w = {r: rw[sel].astype(np.float64) for r, rw in region_w_full.items()}  # (nchan, npts)

    n_samples, n_leads, n_regions, nchan = len(sample_ids), len(leads), len(_regions), len(_channels)
    S00 = np.zeros((n_regions, n_samples, n_leads, nchan))
    S0L = np.zeros((n_regions, n_samples, n_leads, nchan))
    SLL = np.zeros((n_regions, n_samples, n_leads, nchan))
    W = np.zeros((n_regions, n_samples, n_leads, nchan))

    for i, s in enumerate(sample_ids):
        init_date = root.store.init_times[s]

        src_raw = np.asarray(root[f"{s}/{STREAM}/0/source/data"])
        if src_idx is not None:
            src_raw = src_raw[:, src_idx]
        # Same denormalization scores.py's persistence baseline applies.
        src_phys = (src_raw * src_std + src_mean).astype(np.float32)
        src_clean = scores._clean(src_phys, mask_full)

        c0 = scores._clean(clim.get(init_date), mask_full)
        a0_full = src_clean.astype(np.float64) - c0.astype(np.float64)  # (npts, nchan_full)
        a0 = a0_full[:, sel].T  # (nchan, npts)

        for j, lead in enumerate(leads):
            tar_group = root[f"{s}/{STREAM}/{lead}/target"]
            if list(tar_group.attrs["channels"]) != channels_full:
                raise ValueError(
                    f"sample {s} stream {STREAM} lead {lead}: target channels drifted "
                    "from the reference list"
                )
            tar = scores._clean(np.asarray(tar_group["data"]), mask_full)

            cL = scores._clean(clim.get(init_date + np.timedelta64(lead, "D")), mask_full)
            aL_full = tar.astype(np.float64) - cL.astype(np.float64)
            aL = aL_full[:, sel].T  # (nchan, npts)

            for k, region in enumerate(_regions):
                rw = region_w[region]
                S00[k, i, j, :] = (rw * a0 * a0).sum(axis=1)
                S0L[k, i, j, :] = (rw * a0 * aL).sum(axis=1)
                SLL[k, i, j, :] = (rw * aL * aL).sum(axis=1)
                W[k, i, j, :] = rw.sum(axis=1)

    return list(sample_ids), S00, S0L, SLL, W


def _run_blocks(
    cfg: dict, channels: list[str], blocks: list[list[int]], n_workers: int
) -> list[tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Score `blocks` in-process (`n_workers == 1`) or via a spawned
    `ProcessPoolExecutor`, mirroring `compute.py._run_blocks`'s pattern."""
    t0 = time.perf_counter()
    results = []
    if n_workers == 1:
        init_worker(cfg, channels)
        for i, block in enumerate(blocks):
            results.append(process_block(block))
            logger.info(
                "block %d/%d done (%d samples): %.1fs elapsed",
                i + 1, len(blocks), len(block), time.perf_counter() - t0,
            )
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=init_worker, initargs=(cfg, channels),
        ) as pool:
            for i, r in enumerate(pool.map(process_block, blocks)):
                results.append(r)
                logger.info(
                    "block %d/%d done (%d samples): %.1fs elapsed",
                    i + 1, len(blocks), len(blocks[i]), time.perf_counter() - t0,
                )
    return results


def reduce_blocks(cfg: dict, channels: list[str], results: list) -> xr.Dataset:
    """Concatenate block results (time-ordered slices of the time-sorted sample
    list, per `_select_samples`) into one `(region, sample, lead, channel)` dataset."""
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    leads, _ = scores._compute_leads(cfg)
    regions = list(cfg["regions"])

    sample_ids = [s for r in results for s in r[0]]
    S00 = np.concatenate([r[1] for r in results], axis=1)
    S0L = np.concatenate([r[2] for r in results], axis=1)
    SLL = np.concatenate([r[3] for r in results], axis=1)
    W = np.concatenate([r[4] for r in results], axis=1)
    init_times = np.array([root.store.init_times[s] for s in sample_ids], dtype="datetime64[ns]")

    dims = ("region", "sample", "lead", "channel")
    return xr.Dataset(
        {
            "S00": (dims, S00),
            "S0L": (dims, S0L),
            "SLL": (dims, SLL),
            "W": (dims, W),
        },
        coords={
            "region": regions,
            "sample": sample_ids,
            "init_time": ("sample", init_times),
            "lead": leads,
            "channel": channels,
        },
        attrs={
            "run_id": cfg["run_id"],
            "n_samples": len(sample_ids),
            "note": (
                "Damped-persistence baseline (Task 28): xhat(t0+L) = clim(doy(t0+L)) + "
                "alpha(L) * [x(t0) - clim(doy(t0))], a0 = x(t0) - clim(doy(t0)), "
                "aL = x(t0+L) - clim(doy(t0+L)). S00/S0L/SLL/W are the area-weighted "
                "sufficient statistics of MSE(alpha) = (alpha^2*S00 - 2*alpha*S0L + SLL) / W "
                "over wet points; alpha is NOT stored here -- it is fit post hoc, e.g. "
                "alpha*(L) = sum_samples(S0L) / sum_samples(S00) per region/channel."
            ),
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument(
        "--channels", default=None, help="Comma-separated channel names. "
        "Default: cfg['seasons']['channels'] if present, else sst,sss,ssh."
    )
    parser.add_argument("--n-workers", type=int, default=None, help="Overrides cfg['compute']['n_workers'].")
    parser.add_argument(
        "--samples", default=None,
        help="Positional indices into the time-sorted, init-filtered sample list "
        "(e.g. '0:5' or '3,17,42'). Default: all.",
    )
    parser.add_argument("--leads", default=None, help="Comma-separated leads to score (e.g. '1,15,30').")
    parser.add_argument(
        "--stream", default="FESOM_NODE", choices=["FESOM_NODE", "FESOM_ELEM"],
        help="Stream to accumulate. FESOM_ELEM supplies the u/v statistics that combine "
        "into the vector |U| baseline (the statistics are additive across components).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if cfg.get("geometry", "fesom") == "eerie":
        logger.error(
            "geometry 'eerie' is not supported by compute_damped.py -- the EERIE path "
            "would need the sparse-point cell-gather machinery (scores._process_block_eerie), "
            "which is out of scope for this baseline (FESOM geometry only)."
        )
        raise SystemExit(1)

    if args.leads is not None:
        cfg["_leads_override"] = parse_int_list(args.leads)
    if args.n_workers is not None:
        cfg["compute"]["n_workers"] = args.n_workers

    global STREAM
    STREAM = args.stream
    cfg["_damped_stream"] = STREAM          # crosses the spawn boundary via init_worker

    default_channels = (
        list(cfg.get("seasons", {}).get("channels", ["sst", "sss", "ssh"]))
        if STREAM == "FESOM_NODE" else ["u_2", "v_2"]
    )
    channels = (
        [c.strip() for c in args.channels.split(",")] if args.channels is not None
        else default_channels
    )

    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    try:
        selected = _select_samples(cfg, root, parse_int_list(args.samples))
    except ValueError as e:
        logger.error("%s", e)
        raise SystemExit(1) from None

    t_start = time.perf_counter()
    n_workers = max(1, min(cfg["compute"]["n_workers"], len(selected)))
    blocks = [
        [int(s) for s in block]
        for block in np.array_split(np.array(selected, dtype=np.int64), n_workers)
    ]
    logger.info("blocks: %d (n_workers=%d), channels=%s", len(blocks), n_workers, channels)

    results = _run_blocks(cfg, channels, blocks, n_workers)
    ds = reduce_blocks(cfg, channels, results)

    out_path = cache_dir(cfg) / f"damped_{STREAM}.nc"
    ds.to_netcdf(out_path)

    elapsed = time.perf_counter() - t_start
    logger.info("wrote %s (%s, %.1fs total)", out_path, ds.nbytes, elapsed)


if __name__ == "__main__":
    main()
