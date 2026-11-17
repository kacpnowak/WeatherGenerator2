"""Stability gate for a free rollout: is every prediction finite, and does the sea-surface
temperature stay within a few degrees of the truth?

Free rollouts of the fvars models blew up at days 50-80 while their 15-day training and
validation windows looked fine, so no model counts until a clean rollout past that window
passes. The gate needs no climatology: it reads `temp_2` of the TEMP stream from the
rollout output (`merged_zarr`), the truth from the ocean_node archive, and weights the
error by area over the wet points (`meshtools`).

    python -m paper_eval.rollout_gate --results-dir <dir> --mini-epoch 0 \
        --archive <ifs-fesom>/ocean_node --mesh <ifs-fesom>/fesom.mesh.diag.nc --every 10

Exit status 0 when the gate passes, 1 when it fails. A rollout that stops before
`--min-lead` (default 120 days, past the days 50-80 blow-up window) fails, so a crashed run
cannot pass. Finiteness is checked at every lead, the SST RMSE at every `--every`-th.
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

from paper_eval.merged_zarr import open_merged_ranks
from paper_eval.meshtools import PointIndex, load_mesh, masks_and_weights
from paper_eval.scores import full_point_coords

#: Streams whose predictions must be finite at every checked lead.
STREAMS = ("TEMP", "SALT", "SURFACE", "U", "V")


def gate_verdict(
    leads: NDArray,
    sst_rmse: NDArray,
    all_finite: bool,
    max_sst_rmse: float = 3.0,
    min_lead: int = 120,
    nonfinite: str = "",
) -> tuple[bool, str]:
    """Pass iff the rollout reaches `min_lead`, every output is finite and every SST RMSE is
    finite and below `max_sst_rmse`.

    A short (crashed or killed) rollout must not pass vacuously, so an empty series or one
    whose last lead is below `min_lead` fails first. The message names the first failing
    lead (an RMSE at exactly the threshold fails); `nonfinite` says where a prediction was
    not finite (e.g. "stream SALT, lead 7").
    """
    leads = np.asarray(leads)
    rmse = np.asarray(sst_rmse, dtype=np.float64)
    if leads.size == 0:
        return False, f"FAIL: no leads checked (a rollout of at least {min_lead} leads is required)"
    if int(leads.max()) < min_lead:
        short = min_lead - int(leads.max())
        return False, (
            f"FAIL: rollout stops at lead {int(leads.max())}, {short} short of the required "
            f"{min_lead}"
        )
    bad = ~(np.isfinite(rmse) & (rmse < max_sst_rmse))
    if bad.any():
        i = int(np.argmax(bad))
        what = "not finite" if not np.isfinite(rmse[i]) else f"not below {max_sst_rmse:g}"
        return False, f"FAIL: SST RMSE {rmse[i]:.3g} {what} at lead {int(leads[i])}"
    if not all_finite:
        where = f" ({nonfinite})" if nonfinite else ""
        return False, f"FAIL: a prediction is not finite{where}"
    return True, f"PASS: all outputs finite, SST RMSE below {max_sst_rmse:g} at every lead"


def weighted_rmse(pred: NDArray, truth: NDArray, wet: NDArray, w: NDArray) -> float:
    """Area-weighted RMSE over the wet points. NaN at a dry point is ignored (its value is
    unconstrained); NaN at a wet point gives NaN."""
    wt = np.where(wet, w, 0.0).astype(np.float64)
    err2 = np.where(wet, (pred.astype(np.float64) - truth) ** 2, 0.0)
    return float(np.sqrt((wt * err2).sum() / wt.sum()))


def select_leads(available: Iterable[int], every: int) -> list[int]:
    """Every `every`-th lead, plus the last one so the end of the rollout is always checked."""
    leads = sorted(int(x) for x in available)
    picked = [x for x in leads if x % every == 0]
    if leads and leads[-1] not in picked:
        picked.append(leads[-1])
    return picked


def truth_day(init_time: np.datetime64, lead: int) -> np.datetime64:
    """Calendar day of the truth for `lead`: the init date plus `lead` days (the alignment
    `scores._process_block_fesom` uses for the climatology; one forecast step is one day)."""
    return (np.datetime64(init_time, "D") + np.timedelta64(int(lead), "D")).astype("datetime64[D]")


class TruthArchive:
    """Daily fields from a directory of `<name>_<year>.zarr` stores (`data` has one block of
    `npts` rows per day, columns named by `data.attrs["colnames"]`; `dates` stamps each row)."""

    def __init__(self, archive: str | Path):
        self.dir = Path(archive)
        self._years: dict[int, tuple] = {}

    def _year(self, year: int) -> tuple:
        if year not in self._years:
            path = self.dir / f"{self.dir.name}_{year}.zarr"
            g = zarr.open_group(str(path), mode="r")
            attrs = g["data"].attrs.asdict()
            npts = int(attrs.get("nod2", attrs.get("n_points", 0)))
            if npts <= 0:
                raise ValueError(f"{path}: data.attrs has neither 'nod2' nor 'n_points'")
            days = np.asarray(g["dates"][::npts, 0]).astype("datetime64[D]")
            self._years[year] = (g["data"], list(attrs["colnames"]), npts, days)
        return self._years[year]

    def field(self, day: np.datetime64, channel: str) -> NDArray:
        """(npts,) float32 values of `channel` on `day`."""
        year = int(day.astype("datetime64[Y]").astype(int)) + 1970
        data, colnames, npts, days = self._year(year)
        d = int(np.searchsorted(days, day))
        if d >= days.size or days[d] != day:
            raise KeyError(f"{day} not in the {year} archive store of {self.dir}")
        return np.asarray(data[d * npts : (d + 1) * npts, colnames.index(channel)])


def _available_leads(root, sample: int, stream: str) -> list[int]:
    return [
        int(f)
        for f, g in root[f"{sample}/{stream}"].groups()
        if "prediction" in set(g.group_keys())
    ]


def run_gate(args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Open the rollout, check finiteness at every lead and the SST RMSE at the selected
    leads, return (passed, report lines)."""
    root = open_merged_ranks(args.results_dir, args.mini_epoch)
    store = root.store
    samples = list(store.samples) if args.samples is None else list(args.samples)
    mesh = load_mesh(args.mesh)
    truth = TruthArchive(args.archive)
    channel = args.channel

    # Leads every sample has: a rollout that died early leaves fewer, and the gate says so.
    common = set.intersection(*(set(_available_leads(root, s, args.stream)) for s in samples))
    all_leads = sorted(common)
    leads = select_leads(all_leads, args.every)

    # The mesh decides wet and dry, as everywhere in paper_eval.
    mask, w = masks_and_weights(mesh, "node", [channel])
    wet, area = mask[0], w[0]
    points = PointIndex(full_point_coords(root, samples[0], args.stream))

    # Finite at every lead, not only the scored ones: a blow-up can fall between them.
    nonfinite = ""
    bad_lead: dict[int, bool] = {}
    for lead in all_leads:
        for s in samples:
            for stream in args.streams:
                data = np.asarray(root[f"{s}/{stream}/{lead}/prediction"]["data"])
                if not np.isfinite(data).all():
                    bad_lead[lead] = True
                    nonfinite = nonfinite or f"stream {stream}, lead {lead}, sample {s}"
        if nonfinite:
            break  # the first non-finite lead is enough

    sst_rmse = np.full(len(leads), -np.inf)
    for s in samples:
        for j, lead in enumerate(leads):
            group = root[f"{s}/{args.stream}/{lead}/prediction"]
            ch = list(group.attrs["channels"]).index(channel)
            field = np.asarray(group["data"][:, ch, 0])
            field, sampled = points.densify(field, group["coords"])
            wet_l = wet if sampled is None else wet & sampled
            tru = truth.field(truth_day(store.init_times[s], lead), channel)
            r = weighted_rmse(field, tru, wet_l, area)
            # NaN must survive the max over samples
            sst_rmse[j] = r if np.isnan(r) or np.isnan(sst_rmse[j]) else max(sst_rmse[j], r)

    lines = [
        f"lead {lead:4d}  {channel} RMSE {sst_rmse[j]:8.4f}"
        f"  streams finite: {not bad_lead.get(lead, False)}"
        for j, lead in enumerate(leads)
    ]
    ok, msg = gate_verdict(
        np.array(leads),
        sst_rmse,
        not nonfinite,
        args.max_sst_rmse,
        args.min_lead,
        nonfinite,
    )
    return ok, [*lines, msg]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--results-dir", required=True, help="directory with validation_chkpt*_rank*.zip"
    )
    p.add_argument("--mini-epoch", type=int, default=0)
    p.add_argument(
        "--archive", required=True, help="ocean_node truth directory (ocean_node_<year>.zarr)"
    )
    p.add_argument("--mesh", required=True, help="fesom.mesh.diag.nc")
    p.add_argument("--every", type=int, default=10, help="check every N-th lead (and the last)")
    p.add_argument("--stream", default="TEMP", help="stream holding the SST channel")
    p.add_argument("--channel", default="temp_2")
    p.add_argument(
        "--streams", nargs="+", default=list(STREAMS), help="streams checked for finiteness"
    )
    p.add_argument(
        "--samples", type=int, nargs="+", default=None, help="global sample ids (default all)"
    )
    p.add_argument("--max-sst-rmse", type=float, default=3.0)
    p.add_argument(
        "--min-lead", type=int, default=120, help="fail if the rollout stops before this lead"
    )
    args = p.parse_args(argv)

    ok, lines = run_gate(args)
    sys.stdout.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
