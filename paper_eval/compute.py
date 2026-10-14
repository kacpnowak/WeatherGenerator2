"""CLI: stage-1 orchestrator -- scores, spatial maps and showcase snapshots for the
paper evaluation pipeline.

    .venv/bin/python -m paper_eval.compute --config paper_eval/configs/default.yaml \\
        [--samples 0:5|3,17,42] [--leads 1,15,30] [--streams FESOM_NODE[,FESOM_ELEM]] \\
        [--n-workers N]

Wires `paper_eval.scores` (per-block worker: `init_worker`/`process_block`/
`reduce_blocks`) into a parallel pass over the time-sorted, init-filtered sample list,
then extracts the showcase snapshot (`paper_eval.snapshots.extract`) in the parent.
Writes every stage-1 cache (`scores_*.nc`, `maps_*.nc`, `snapshots_*.nc`, `meta.json`)
under `cache_dir(cfg)`, overwriting silently -- these are regenerable caches, not
inputs to anything upstream.

`--n-workers 1` (or a selection smaller than the configured worker count) runs
in-process instead of spawning a pool: a debug/small-run path that avoids
`ProcessPoolExecutor` overhead.

Only config loading and sample selection (`_prepare`) get a clean, traceback-free
CLI error on `FileNotFoundError`/`ValueError` -- bad `--config`, bad `--samples`,
or a selection the init filter empties out. Everything downstream (scoring, reduce,
snapshots, writes) is expected to raise a normal traceback on failure: those are
real bugs (e.g. `scores.py`'s wet-point NaN check), not bad input, and swallowing
them would be actively harmful inside an sbatch job.
"""

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from paper_eval import scores, snapshots
from paper_eval.config import cache_dir, load_config, parse_int_list
from paper_eval.merged_zarr import open_merged_ranks
from paper_eval.scores import BlockResult

logger = logging.getLogger(__name__)


def _select_samples(cfg: dict, root, sample_idx: list[int] | None) -> list[int]:
    """Global sample ids to score: `root.store.samples` (already time-sorted), with
    samples outside `cfg["init_filter"]` dropped, then `sample_idx` applied as
    positional indices into the surviving (still time-sorted) list.

    `sample_idx=None` selects every surviving sample. Raises `ValueError` with a clear
    message (no traceback wanted at the CLI level -- see `main`) if nothing survives
    the init filter or an index in `sample_idx` is out of range.
    """
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

    if sample_idx is None:
        selected = filtered
    else:
        bad = [i for i in sample_idx if not (0 <= i < len(filtered))]
        if bad:
            raise ValueError(
                f"--samples index/indices {bad} out of range: {len(filtered)} sample(s) "
                f"survive init_filter {cfg['init_filter']!r} (valid range "
                f"0..{len(filtered) - 1})"
            )
        selected = [filtered[i] for i in sample_idx]

    if not selected:
        raise ValueError("--samples selected zero samples")

    logger.info(
        "samples: %d total, %d after init_filter, %d selected, init %s..%s",
        len(samples_all),
        len(filtered),
        len(selected),
        init_times[selected[0]],
        init_times[selected[-1]],
    )
    return selected


def _run_blocks(
    cfg: dict, blocks: list[list[int]], n_workers: int
) -> tuple[list[BlockResult], dict | None]:
    """Score `blocks` in-process (`n_workers == 1`, the debug path) or via a spawned
    `ProcessPoolExecutor` initialized once per worker with `scores.init_worker` --
    `executor.map` preserves block order.

    Each `BlockResult`'s (large) `map_sums` is folded into a running total as soon
    as it arrives (`scores.merge_map_sums`) and then stripped (`r.map_sums = {}`)
    before the block is kept for score concatenation -- the parent never holds more
    than the running total plus the block just merged, not every block's map_sums
    at once. Returns `(score_results, merged_maps)` for `scores.reduce_blocks`.

    Logs one line per completed block with elapsed time since the pass started."""
    results: list[BlockResult] = []
    merged: dict | None = None
    t0 = time.perf_counter()

    def _fold(r: BlockResult) -> None:
        nonlocal merged
        merged = scores.merge_map_sums(merged, r)
        r.map_sums = {}
        results.append(r)

    if n_workers == 1:
        scores.init_worker(cfg)
        for i, block in enumerate(blocks):
            _fold(scores.process_block(block))
            logger.info(
                "block %d/%d done (%d samples): %.1fs elapsed",
                i + 1,
                len(blocks),
                len(block),
                time.perf_counter() - t0,
            )
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=scores.init_worker,
            initargs=(cfg,),
        ) as pool:
            for i, r in enumerate(pool.map(scores.process_block, blocks)):
                _fold(r)
                logger.info(
                    "block %d/%d done (%d samples): %.1fs elapsed",
                    i + 1,
                    len(blocks),
                    len(blocks[i]),
                    time.perf_counter() - t0,
                )
    return results, merged


def _fmt_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _prepare(args: argparse.Namespace) -> tuple[dict, object, list[int]]:
    """Load the config, apply CLI overrides, open the merged root and resolve the
    sample selection -- every error this can raise (`FileNotFoundError` from a bad
    `--config`/`results_dir`, `ValueError` from `_select_samples`) is an expected,
    clean input-validation failure. `main` wraps only this call in its clean-error
    handler; everything downstream (scoring, reduce, snapshots, writes) is expected
    to raise a normal traceback on failure -- those are real bugs, not bad input.
    """
    cfg = load_config(args.config)
    if args.leads is not None:
        cfg["_leads_override"] = parse_int_list(args.leads)
    if args.streams is not None:
        cfg["streams"] = [s.strip() for s in args.streams.split(",")]
    if args.n_workers is not None:
        cfg["compute"]["n_workers"] = args.n_workers

    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    selected = _select_samples(cfg, root, parse_int_list(args.samples))
    return cfg, root, selected


def _run(cfg: dict, root, selected: list[int]) -> None:
    t_start = time.perf_counter()

    n_workers = max(1, min(cfg["compute"]["n_workers"], len(selected)))
    blocks = [
        [int(s) for s in block]
        for block in np.array_split(np.array(selected, dtype=np.int64), n_workers)
    ]
    logger.info("blocks: %d (n_workers=%d)", len(blocks), n_workers)

    block_results, merged_maps = _run_blocks(cfg, blocks, n_workers)

    datasets = scores.reduce_blocks(block_results, merged_maps, cfg)
    datasets.update(snapshots.extract(cfg, root))

    out_dir = cache_dir(cfg)
    written: list[Path] = []
    for key, ds in datasets.items():
        path = out_dir / f"{key}.nc"
        ds.to_netcdf(path)
        written.append(path)

    leads = [int(x) for x in datasets[f"scores_{cfg['streams'][0]}"]["lead"].values]
    elapsed = time.perf_counter() - t_start
    meta = {
        "config": cfg,
        "sample_ids": [int(s) for s in selected],
        "init_times": [str(root.store.init_times[s]) for s in selected],
        "leads": leads,
        "wall_time_seconds": elapsed,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    written.append(meta_path)

    logger.info("=== summary ===")
    for path in written:
        logger.info("  %s (%s)", path, _fmt_size(path.stat().st_size))
    logger.info("total wall time: %.1fs", elapsed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument(
        "--samples",
        default=None,
        help="Positional indices into the time-sorted, init-filtered sample list "
        "(e.g. '0:5' or '3,17,42'). Default: all.",
    )
    parser.add_argument(
        "--leads",
        default=None,
        help="Comma-separated leads to score (e.g. '1,15,30'). Overrides "
        "cfg['leads']['max']-derived default via cfg['_leads_override'].",
    )
    parser.add_argument(
        "--streams",
        default=None,
        help="Comma-separated stream names (e.g. 'FESOM_NODE,FESOM_ELEM'). "
        "Replaces cfg['streams'].",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None, help="Overrides cfg['compute']['n_workers']."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        cfg, root, selected = _prepare(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        raise SystemExit(1) from None

    _run(cfg, root, selected)


if __name__ == "__main__":
    main()
