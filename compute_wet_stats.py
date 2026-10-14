# ruff: noqa: T201
"""Wet-point normalisation statistics for the depth-dependent FESOM channels.

FESOM stores dry points (land, below the sea floor) as 0. The archive's normalisation statistics
(`data.attrs["means"]` / `["std"]`) include those zeros, so deep channels get absurd statistics
(salt_450: mean 23.9 psu, std 15.0, against 34.7 and 0.8 over the wet points). This script
computes mean and std per channel over the WET points only, with the wet mask taken from the mesh
(never from the data values), for

  --stream node:  temp_<depth> and salt_<depth>   (FESOM_NODE)
  --stream elem:  u_<depth> and v_<depth>         (FESOM_ELEM)

and writes them as JSON for the `normalization_stats` stream option of the FESOM reader:

  {"_meta": {...}, "salt_450": {"mean": 34.74, "std": 0.82}, ...}

Surface channels (sst, sss, ssh, ...) keep the archive statistics and are not written.

One day is sampled every `--every` days of each year (default 73: 5 days per year) from the
training years (2000-2208; 2209 is held out). Usage:

  .venv/bin/python compute_wet_stats.py --stream node --out config/streams/fesom/wet_stats_node.json
  .venv/bin/python compute_wet_stats.py --stream elem --out config/streams/fesom/wet_stats_elem.json

A tiny smoke setting: `--first-year 2000 --last-year 2001 --every 365 --out <tmp>.json`.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from paper_eval.meshtools import load_mesh, masks_and_weights  # noqa: E402

ARCHIVE = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
MESH = Path(
    "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
)

# per stream: the mesh stream name, the regex of the depth-dependent channels, and the channel
# used to check that the archive rows follow the mesh order
STREAMS = {
    "node": ("FESOM_NODE", re.compile(r"^(temp|salt)_\d+$"), "salt_450"),
    "elem": ("FESOM_ELEM", re.compile(r"^(u|v)_\d+$"), "u_450"),
}
FIRST_YEAR, LAST_YEAR = 2000, 2208  # 2209 is held out


def check_row_order(block, colnames, channels, mask, check_channel):
    """Assert that the archive rows follow the mesh order: dry points hold 0 (or NaN).

    `block` is one day (npts, ncols). Prints the counts for `check_channel` and, over all
    depth channels, the exceptions. A handful of points where the mesh says dry but the archive
    holds a value (the mesh/data boundary nodes, e.g. node 125225) are tolerated, as is the NaN
    that the archive holds at some dry points of shallow salt levels; a wrong row order or mask
    would put the exceptions in the thousands.
    """
    col = {name: i for i, name in enumerate(colnames)}
    ci = channels.index(check_channel)
    values = block[:, col[check_channel]]
    dry, wet = ~mask[ci], mask[ci]
    n_dry_zero = int((values[dry] == 0.0).sum())
    n_wet_nonzero = int((values[wet] != 0.0).sum())
    print(
        f"row-order check on {check_channel}: dry {dry.sum()} (exactly 0: {n_dry_zero}, "
        f"other: {dry.sum() - n_dry_zero}), wet {wet.sum()} (nonzero: {n_wet_nonzero})"
    )
    assert n_dry_zero >= (1 - 1e-4) * dry.sum(), f"{check_channel}: mesh-dry points are not 0"
    assert n_wet_nonzero >= (1 - 1e-4) * wet.sum(), f"{check_channel}: mesh-wet points are 0"

    dry_other = dry_nan = wet_zero = wet_nan = n_dry = n_wet = 0
    for i, name in enumerate(channels):
        v = block[:, col[name]]
        d, w = v[~mask[i]], v[mask[i]]
        dry_nan += int(np.isnan(d).sum())
        dry_other += int(((d != 0.0) & ~np.isnan(d)).sum())
        wet_zero += int((w == 0.0).sum())
        wet_nan += int(np.isnan(w).sum())
        n_dry, n_wet = n_dry + d.size, n_wet + w.size
    print(
        f"all {len(channels)} depth channels: dry points {n_dry} (nonzero non-NaN: {dry_other}, "
        f"NaN: {dry_nan}), wet points {n_wet} (exactly 0: {wet_zero}, NaN: {wet_nan})"
    )
    assert dry_other <= 1e-4 * n_dry, "archive holds values at many mesh-dry points"
    assert wet_zero + wet_nan <= 1e-4 * n_wet, "archive holds 0 or NaN at many mesh-wet points"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stream", choices=sorted(STREAMS), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output JSON")
    parser.add_argument("--every", type=int, default=73, help="sample one day every N days")
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_YEAR)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--mesh", type=Path, default=MESH)
    args = parser.parse_args()
    assert args.every >= 1 and args.first_year <= args.last_year

    mesh_stream, pattern, check_channel = STREAMS[args.stream]
    mesh = load_mesh(args.mesh)
    npts = mesh.lon.shape[0] if args.stream == "node" else mesh.elem_area.shape[0]

    def open_year(year):
        path = args.archive / f"ocean_{args.stream}" / f"ocean_{args.stream}_{year}.zarr"
        return zarr.open_group(str(path), mode="r")["data"]

    first = open_year(args.first_year)
    old_mean = dict(zip(first.attrs["colnames"][2:], first.attrs["means"], strict=True))
    old_std = dict(zip(first.attrs["colnames"][2:], first.attrs["std"], strict=True))
    assert list(first.attrs["colnames"][:2]) == ["lon", "lat"]
    channels = [c for c in first.attrs["colnames"] if pattern.match(c)]
    mask, _ = masks_and_weights(mesh, mesh_stream, channels)
    print(f"{args.stream}: {len(channels)} depth channels, {npts} points per day, mesh {args.mesh}")

    # sums of (x - pivot) and (x - pivot)^2 in float64; the pivot (the archive mean) keeps the
    # variance free of cancellation for channels with a large mean and a small spread
    pivot = np.array([old_mean[c] for c in channels])
    s1, s2 = np.zeros(len(channels)), np.zeros(len(channels))
    count = np.zeros(len(channels), dtype=np.int64)
    n_nonfinite, n_days, checked = 0, 0, False
    t0 = time.time()

    for year in range(args.first_year, args.last_year + 1):
        data = open_year(year)
        colnames = list(data.attrs["colnames"])
        col = {name: i for i, name in enumerate(colnames)}
        cols = [col[c] for c in channels]
        n_rows = data.shape[0]
        assert n_rows % npts == 0, f"{year}: {n_rows} rows is not a multiple of {npts} points"

        for day in range(0, n_rows // npts, args.every):
            block = data[day * npts : (day + 1) * npts, :]  # one day, all columns
            if not checked:
                check_row_order(block, colnames, channels, mask, check_channel)
                checked = True
            for i, c in enumerate(cols):
                v = block[mask[i], c].astype(np.float64)
                finite = np.isfinite(v)
                n_nonfinite += int(v.size - finite.sum())
                v = v[finite] - pivot[i]
                s1[i] += v.sum()
                s2[i] += np.square(v).sum()
                count[i] += v.size
            n_days += 1
        print(f"  {year}: {n_days} days so far, {time.time() - t0:.0f} s", flush=True)

    # the deepest level (6125 m) has no wet point at all: nothing to compute, and the channel
    # (0 everywhere, so a target of NaN and a source of 0 under the fill options) keeps the
    # archive statistics
    all_dry = [c for c, n in zip(channels, count, strict=True) if n == 0]
    if all_dry:
        print(f"no wet points, left out of the output: {', '.join(all_dry)}")
    safe_count = np.maximum(count, 1)
    mean_dev = s1 / safe_count
    mean = pivot + mean_dev
    std = np.sqrt(np.maximum(s2 / safe_count - mean_dev**2, 0.0))
    has_wet = count > 0
    assert np.isfinite(mean[has_wet]).all() and np.isfinite(std[has_wet]).all()
    assert (std[has_wet] > 0).all()
    if n_nonfinite:
        print(f"WARNING: {n_nonfinite} non-finite wet values were skipped")

    result = {
        "_meta": {
            "stream": args.stream,
            "years": [args.first_year, args.last_year],
            "every": args.every,
            "n_days": n_days,
            "no_wet_points": all_dry,
            "wet_mask": "mesh (nlevels_nod2D / nlevels)",
            "mesh": str(args.mesh),
            "archive": str(args.archive / f"ocean_{args.stream}"),
        }
    }
    for i, c in enumerate(channels):
        if has_wet[i]:
            result[c] = {"mean": float(mean[i]), "std": float(std[i])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"\n{'channel':>10} {'old mean':>12} {'new mean':>12} {'old std':>12} {'new std':>12}")
    for i, c in enumerate(channels):
        if has_wet[i]:
            print(f"{c:>10} {old_mean[c]:12.5g} {mean[i]:12.5g} {old_std[c]:12.5g} {std[i]:12.5g}")
    print(f"\nwrote {args.out} ({n_days} days, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
