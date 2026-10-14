"""CLI: build a day-of-year climatology for one stream from its AWICM3 truth archive.

    .venv/bin/python -m paper_eval.build_clim --config paper_eval/configs/default.yaml \\
        --stream FESOM_NODE [--ref-start 2207] [--ref-end 2208] [--force]

The channel list (and its order) is read from the inference results store rather than
the truth archive's own `colnames`, so the climatology's channel axis is guaranteed to
match the run being scored against it.
"""

import argparse
import logging
import sys

from paper_eval.climatology import build_climatology
from paper_eval.config import clim_dir, load_config
from paper_eval.merged_zarr import open_merged_ranks

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument("--stream", required=True, help="e.g. FESOM_NODE or FESOM_ELEM.")
    parser.add_argument("--ref-start", type=int, default=None, help="Overrides cfg['climatology']['ref_start'].")
    parser.add_argument("--ref-end", type=int, default=None, help="Overrides cfg['climatology']['ref_end'].")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing climatology output.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    ref_start = args.ref_start if args.ref_start is not None else cfg["climatology"]["ref_start"]
    ref_end = args.ref_end if args.ref_end is not None else cfg["climatology"]["ref_end"]

    out_path = clim_dir(cfg) / f"clim_{args.stream}_{ref_start}-{ref_end}.zarr"
    if out_path.exists() and not args.force:
        print(f"Refusing to overwrite existing climatology: {out_path} (pass --force to overwrite)")
        sys.exit(1)

    logger.info("reading channel list for %s from results store %s", args.stream, cfg["results_dir"])
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    s0 = root.store.samples[0]
    channels = list(root[f"{s0}/{args.stream}/1/target"].attrs["channels"])

    truth_dir = cfg["truth"][args.stream]
    logger.info(
        "building climatology: stream=%s truth_dir=%s ref=%d-%d out=%s",
        args.stream, truth_dir, ref_start, ref_end, out_path,
    )
    build_climatology(
        truth_dir=truth_dir,
        channels=channels,
        ref_start=ref_start,
        ref_end=ref_end,
        out_path=out_path,
    )
    logger.info("done: %s", out_path)


if __name__ == "__main__":
    main()
