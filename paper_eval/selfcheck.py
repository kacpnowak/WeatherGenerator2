"""CLI: independent verification of the paper evaluation stage-1 cache.

    .venv/bin/python -m paper_eval.selfcheck --config paper_eval/configs/default.yaml

Re-derives a handful of quantities directly from the raw inputs (merged inference
zarr, mesh file via xarray, climatology zarr, truth-store attrs) with code paths
INDEPENDENT of meshtools.py/metrics.py/scores.py/climatology.py -- only
`paper_eval.config` (path helpers) and `paper_eval.merged_zarr` (store access) are
imported from this package -- and compares the result against the stage-1 cache
built by `compute.py` (``scores_FESOM_NODE.nc``, ``maps_FESOM_NODE.nc``,
``meta.json``).

Prints one PASS/FAIL/SKIP line per check (a)-(h) with the numbers behind the
verdict, then a summary line. A check whose required cache/climatology file is
absent (e.g. maps checks when the cache has no map leads) prints SKIP and does not
affect the exit code. Exit 0 iff every check PASSes or SKIPs; exit 1 if any FAILs.

``--cache-dir`` is a hidden, test-only override: read the stage-1 cache from this
directory instead of ``cache_dir(cfg)`` (``results_dir``/``mesh_path``/``truth``/
climatology paths still come from ``--config`` unchanged). It exists so the
failure-detection demo can point at a scratch COPY of the cache -- with one value
corrupted -- without touching ``--config`` or the real cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from paper_eval.config import cache_dir, clim_paths, load_config, normalization_overrides
from paper_eval.merged_zarr import open_merged_ranks

#: Stream used for checks (a)-(g): the only one whose points are in mesh nod2 order,
#: so mesh-derived wet/dry masks and area weights (nod_area, nlevels_nod2D) apply.
#: Default for legacy configs; `cfg["selfcheck"]["stream"]` overrides it (per-variable
#: layout: TEMP, which also carries temp_450).
STREAM = "FESOM_NODE"
#: Surface (mesh-wet everywhere) channel of STREAM the checks use;
#: `cfg["selfcheck"]["surface_channel"]` overrides it (per-variable layout: temp_2,
#: which is what sst was a copy of).
SURFACE_CHANNEL = "sst"

#: Cumulative days before each month (index 1..12) in a FIXED leap calendar (Feb
#: always has 29 days). Independent reimplementation of the (month, day) -> 1..366
#: map used by paper_eval.climatology.doy_index -- not imported from there.
_CUM_DAYS_BEFORE_MONTH = (0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

#: rtol required by the brief for score-recompute comparisons; atol is a small
#: absolute floor (the cache stores float32, ~1e-7 relative precision at O(1)
#: magnitudes) so a value near zero (e.g. bias) isn't spuriously flagged by rtol
#: alone.
_RTOL = 1e-5
_ATOL = 1e-6

#: Fix round 1 (coordinator adjudication): a mesh/data boundary artifact -- node
#: 125225 (lon=-153.99, lat=-76.99, Antarctic coast) -- carries a real temp_450
#: value at a mesh-dry point in every sample/lead checked, yet is harmless (zero
#: weight at dry points in every score). A hard FAIL on this in check (b) would cry
#: wolf on every future run against pristine data, so up to this many dry-point
#: "neither 0.0 nor NaN" values are tolerated as a known boundary-artifact count;
#: exceeding it still FAILs. A NaN at a WET point remains a hard FAIL regardless
#: (the dangerous direction -- unchanged).
_DRY_ANOMALY_TOLERANCE = 10

CHECK_NAMES = {
    "a": "Score recompute",
    "b": "Fill/mask consistency",
    "c": "Persistence ordering",
    "d": "ACC decay",
    "e": "Lead-day relation",
    "f": "Climatology-model ACC",
    "g": "Maps wet/dry",
    "h": "Meta consistency",
    "i": "Baseline maps",
    "j": "Region partition",
}

#: Task 14 map variables -- their absence means the cache predates that task (old
#: caches lack them entirely), which check (i) treats as SKIP, not FAIL.
_TASK14_MAP_VARS = ("rmse_persistence", "rmse_climatology", "acc")


class SkipCheckError(Exception):
    """Raised by a check function when a required cache/climatology file is
    absent -- caught by `main` and reported as SKIP (does not fail the run)."""


@dataclass
class CheckResult:
    id: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    message: str


@dataclass
class Ctx:
    cfg: dict
    cache: Path
    root: object
    mesh: dict
    scores_ds: xr.Dataset
    #: The scores dataset exactly as cached -- unlike `scores_ds`, still carries the
    #: Task 20 `region` dimension when the cache has one. Only check (j) reads this;
    #: every other check uses `scores_ds` (already narrowed to region="global"), so
    #: they need no region-awareness of their own -- old and new caches look
    #: identical to them.
    scores_ds_raw: xr.Dataset
    s_star: int
    lead: int
    max_lead: int
    level_idx_450: int
    dry_mask_450: np.ndarray  # (nod2,) bool, mesh-derived, sample-independent
    stream: str = STREAM  # node stream checked (must carry SURFACE_CHANNEL and temp_450)
    sfc: str = SURFACE_CHANNEL  # its all-wet surface channel
    #: mesh positions of the (s*, L) target's points, in stored order: arange(nod2) for
    #: a full target, the matched subset for one written with `max_num_targets`
    idx: np.ndarray | None = None


def _target_positions(root, s: int, stream: str, lead: int) -> np.ndarray:
    """Mesh positions of the (s, lead) target's points, by exact coordinate match
    against the fstep-0 source (always the full point set in mesh order) -- an
    independent re-derivation of `meshtools.PointIndex`'s nearest-point lookup."""
    src = np.ascontiguousarray(np.asarray(root[f"{s}/{stream}/0/source"]["coords"]))
    tar = np.ascontiguousarray(np.asarray(root[f"{s}/{stream}/{lead}/target"]["coords"]))
    if tar.shape[0] == src.shape[0]:
        return np.arange(src.shape[0])
    key = {row.tobytes(): i for i, row in enumerate(src)}
    idx = np.array([key.get(row.tobytes(), -1) for row in tar])
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} target point(s) not in the source point set")
    return idx


# ── Small independent re-derivations (deliberately not the metrics.py/meshtools.py
#    code paths) ─────────────────────────────────────────────────────────────────
def _doy(date) -> int:
    """Fixed (month, day) -> 1..366 day-of-year for `date` (any np.datetime64-like;
    time-of-day is truncated). Feb 29 is always 60, Mar 1 is always 61."""
    d = np.datetime64(date, "D")
    month_start = d.astype("datetime64[M]")
    month = int(month_start.astype("int64") % 12) + 1
    day = int((d - month_start).astype("int64")) + 1
    return _CUM_DAYS_BEFORE_MONTH[month] + day


def _weighted_rmse_bias(err: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Area-weighted RMSE/bias via plain np.average -- a different code path from
    metrics.score_block's manual sum(W*x)/sum(W) accumulation."""
    rmse = float(np.sqrt(np.average(err**2, weights=w)))
    bias = float(np.average(err, weights=w))
    return rmse, bias


def _weighted_acc(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> float:
    """Area-weighted ACC via np.average; algebraically equal to metrics.py's
    sum-based formula (the weight sums cancel) but expressed with different code."""
    num = np.average(f * o, weights=w)
    den = np.sqrt(np.average(f**2, weights=w) * np.average(o**2, weights=w))
    return float(num / den)


def _level_idx(nz1: np.ndarray, depth_suffix: int) -> int:
    matches = np.nonzero(nz1.astype(int) == depth_suffix)[0]
    if matches.size == 0:
        raise ValueError(f"no nz1 depth matches suffix {depth_suffix}")
    return int(matches[0])


def _load_mesh(mesh_path: str | Path) -> dict:
    """Read only what the checks need straight from the mesh file via xarray."""
    with xr.open_dataset(mesh_path) as ds:
        return {
            "nod_area0": ds["nod_area"].values[0].astype(np.float64),  # (nod2,) m^2
            "nz1": ds["nz1"].values,
            "n_wet_layers_nod": ds["nlevels_nod2D"].values - 1,
            "lon": ds["lon"].values,  # (nod2,) degrees -- for check (b) anomaly examples
            "lat": ds["lat"].values,
        }


def _norm_stats(cfg: dict, stream: str, channel: str) -> tuple[float, float]:
    """(mean, std) for `channel`, read from the first truth-year store's `data`
    attrs (colnames/means/std), aligned by NAME (colname order differs across
    year stores). std<=1e-5 -> 1.0, matching the training reader's no-op rule.

    If the evaluated run sets the stream option `normalization_stats` (wet-point
    statistics; `paper_eval.config.normalization_overrides`) and the JSON has an entry
    for `channel`, that entry replaces the archive statistics, as in the reader."""
    truth_dir = Path(cfg["truth"][stream])
    files = sorted(truth_dir.glob("*.zarr"))
    if not files:
        raise SkipCheckError(f"no truth store (*.zarr) in {truth_dir}")
    g = zarr.open_group(str(files[0]), mode="r")
    attrs = g["data"].attrs.asdict()
    colnames = list(attrs["colnames"])
    means = np.asarray(attrs["means"], dtype=np.float64)
    std = np.asarray(attrs["std"], dtype=np.float64)
    offset = len(colnames) - len(means)
    pos = colnames.index(channel) - offset
    mean, s = float(means[pos]), float(std[pos])
    if s <= 1e-5:
        s = 1.0
    override = (normalization_overrides(cfg, stream) or {}).get(channel)
    if override is not None:
        mean, s = float(override["mean"]), float(override["std"])
        assert np.isfinite(mean) and np.isfinite(s) and s > 0.0, (
            f"invalid normalization override for {channel!r}: mean={mean}, std={s}"
        )
    return mean, s


# ── Checks (a)-(h) ──────────────────────────────────────────────────────────────
def check_a(ctx: Ctx) -> str:
    """Score recompute: emulator rmse/bias/acc and persistence rmse for
    (s*, sst, lead L), vs the cached scores."""
    s, lead, stream = ctx.s_star, ctx.lead, ctx.stream
    tar_g = ctx.root[f"{s}/{stream}/{lead}/target"]
    pred_g = ctx.root[f"{s}/{stream}/{lead}/prediction"]
    channels = list(tar_g.attrs["channels"])
    ci = channels.index(ctx.sfc)
    tar = np.asarray(tar_g["data"])[:, ci].astype(np.float64)
    pred = np.asarray(pred_g["data"])[:, ci, 0].astype(np.float64)
    idx = ctx.idx  # mesh positions of the stored (possibly subsampled) points
    w = ctx.mesh["nod_area0"][idx]

    rmse, bias = _weighted_rmse_bias(pred - tar, w)

    init_time = ctx.root.store.init_times[s]
    valid_date = init_time + np.timedelta64(int(lead), "D")
    doy = _doy(valid_date)
    # first climatology source holding the channel (legacy configs: the one zarr)
    for clim_path in clim_paths(ctx.cfg, stream):
        if not clim_path.exists():
            raise SkipCheckError(f"{clim_path} not found")
        clim_g = zarr.open_group(str(clim_path), mode="r")
        clim_channels = list(clim_g.attrs["channels"])
        if ctx.sfc in clim_channels:
            break
    cci = clim_channels.index(ctx.sfc)
    clim_field = np.asarray(clim_g["clim"][doy - 1])[idx, cci].astype(np.float64)
    acc = _weighted_acc(pred - clim_field, tar - clim_field, w)

    src_g = ctx.root[f"{s}/{stream}/0/source"]
    src_channels = list(src_g.attrs["channels"])
    si = src_channels.index(ctx.sfc)
    src_raw = np.asarray(src_g["data"])[idx, si].astype(np.float64)
    mean, std = _norm_stats(ctx.cfg, stream, ctx.sfc)
    src_phys = src_raw * std + mean
    rmse_p, _ = _weighted_rmse_bias(src_phys - tar, w)

    cached = ctx.scores_ds["score"].sel(sample=s, lead=lead, channel=ctx.sfc)
    wanted = {
        "emulator rmse": (rmse, float(cached.sel(model="emulator", metric="rmse"))),
        "emulator bias": (bias, float(cached.sel(model="emulator", metric="bias"))),
        "emulator acc": (acc, float(cached.sel(model="emulator", metric="acc"))),
        "persistence rmse": (rmse_p, float(cached.sel(model="persistence", metric="rmse"))),
    }
    detail = "; ".join(
        f"{name} recomputed={got:.6g} cached={want:.6g}" for name, (got, want) in wanted.items()
    )
    bad = [
        name for name, (got, want) in wanted.items()
        if not np.isclose(got, want, rtol=_RTOL, atol=_ATOL)
    ]
    assert not bad, f"s={s} {ctx.sfc} lead={lead}: mismatch on {bad} -- {detail}"
    return f"s={s} {ctx.sfc} lead={lead}: {detail}"


def check_b(ctx: Ctx) -> str:
    """Fill/mask consistency: temp_450 target at (s*, L) is 0.0/NaN at mesh-dry
    points and never NaN at mesh-wet points.

    A dry-point value that is neither 0.0 nor NaN is tolerated up to
    `_DRY_ANOMALY_TOLERANCE` (a known mesh/data boundary artifact, e.g. node 125225
    -- see that constant's comment) and only FAILs beyond it; a NaN at a WET point
    is always a hard FAIL (the dangerous direction, never tolerated)."""
    s, lead = ctx.s_star, ctx.lead
    tar_g = ctx.root[f"{s}/{ctx.stream}/{lead}/target"]
    channels = list(tar_g.attrs["channels"])
    ci = channels.index("temp_450")
    tar = np.asarray(tar_g["data"])[:, ci]

    dry = ctx.dry_mask_450[ctx.idx]  # at the stored (possibly subsampled) points
    wet = ~dry
    dry_idx = ctx.idx[np.nonzero(dry)[0]]
    dry_vals, wet_vals = tar[dry], tar[wet]
    n_zero = int(np.sum(dry_vals == 0.0))
    n_nan = int(np.sum(np.isnan(dry_vals)))
    other_local = (dry_vals != 0.0) & ~np.isnan(dry_vals)
    n_other = int(other_local.sum())
    n_wet_nan = int(np.sum(np.isnan(wet_vals)))

    examples = [
        f"node {i} (lon={ctx.mesh['lon'][i]:.2f}, lat={ctx.mesh['lat'][i]:.2f})"
        for i in dry_idx[other_local][:3]
    ]

    msg = (
        f"temp_450 (level_idx={ctx.level_idx_450}) s={s} lead={lead}: "
        f"{dry_vals.size} dry pts (zero={n_zero}, nan={n_nan}, other={n_other}); "
        f"{wet_vals.size} wet pts (nan={n_wet_nan})"
    )
    if n_other:
        msg += f"; other-value example(s): {', '.join(examples)}"
    msg += f"; dry-anomaly tolerance={_DRY_ANOMALY_TOLERANCE} (known: node 125225 Antarctic coast)"

    assert n_wet_nan == 0, f"{msg} -- {n_wet_nan} wet-point NaN(s)"
    assert n_other <= _DRY_ANOMALY_TOLERANCE, (
        f"{msg} -- {n_other} dry-point value(s) neither 0.0 nor NaN exceeds "
        f"tolerance {_DRY_ANOMALY_TOLERANCE}"
    )
    return msg


def _agg_rmse(ds: xr.Dataset, model: str, channel: str, lead: int) -> float:
    """Sample-aggregated rmse: sqrt(mean(rmse^2)) over the sample dim."""
    sel = ds["score"].sel(model=model, metric="rmse", channel=channel, lead=lead)
    return float(np.sqrt((sel**2).mean()))


def _mean_acc(ds: xr.Dataset, model: str, channel: str, lead: int) -> float:
    """Sample-aggregated acc: plain mean over the sample dim."""
    return float(ds["score"].sel(model=model, metric="acc", channel=channel, lead=lead).mean())


def check_c(ctx: Ctx) -> str:
    """Persistence ordering: sample-aggregated (sqrt(mean(rmse^2))) sst rmse at
    lead 1 < lead L, and lead-1 value < 1.0 degC."""
    ds = ctx.scores_ds
    r1 = _agg_rmse(ds, "persistence", ctx.sfc, 1)
    rl = _agg_rmse(ds, "persistence", ctx.sfc, ctx.lead)
    msg = f"persistence {ctx.sfc} rmse: lead1={r1:.4f} lead{ctx.lead}={rl:.4f} degC"
    assert r1 < rl, f"{msg} -- expected lead1 < lead{ctx.lead}"
    assert r1 < 1.0, f"{msg} -- expected lead1 < 1.0 degC"
    return msg


def check_d(ctx: Ctx) -> str:
    """ACC decay: emulator sst acc (plain mean over samples) strictly decreases
    across the available leads {1, L, max}."""
    ds = ctx.scores_ds
    avail = [int(x) for x in ds["lead"].values]
    leads = list(dict.fromkeys(lead for lead in (1, ctx.lead, ctx.max_lead) if lead in avail))
    accs = [_mean_acc(ds, "emulator", ctx.sfc, lead) for lead in leads]
    pairs = zip(leads, accs, strict=True)
    msg = f"emulator {ctx.sfc} acc: " + ", ".join(f"lead{lead}={a:.4f}" for lead, a in pairs)
    decreasing = all(accs[i] > accs[i + 1] for i in range(len(accs) - 1))
    assert decreasing, f"{msg} -- not strictly decreasing"
    return msg


def check_e(ctx: Ctx) -> str:
    """Lead-day relation: for s* and fsteps {1, L}, target `times[0]` date minus
    init_time date equals the forecast step (days)."""
    s = ctx.s_star
    init_d = np.datetime64(ctx.root.store.init_times[s], "D")
    fsteps = list(dict.fromkeys([1, ctx.lead]))
    parts = []
    for fstep in fsteps:
        times = ctx.root[f"{s}/{ctx.stream}/{fstep}/target"]["times"]
        t0 = np.datetime64(times[0], "D")
        delta = int((t0 - init_d).astype("int64"))
        parts.append(f"fstep{fstep}: delta={delta}d")
        assert delta == fstep, (
            f"s={s} fstep={fstep}: times[0] date {t0} - init {init_d} = {delta}d != {fstep}"
        )
    return f"s={s} init={init_d}: " + ", ".join(parts)


def check_f(ctx: Ctx) -> str:
    """Climatology-model ACC: all-NaN; climatology-model rmse: all finite."""
    ds = ctx.scores_ds
    acc = ds["score"].sel(model="climatology", metric="acc").values
    rmse = ds["score"].sel(model="climatology", metric="rmse").values
    n_non_nan_acc = int(np.sum(~np.isnan(acc)))
    n_non_finite_rmse = int(np.sum(~np.isfinite(rmse)))
    msg = f"climatology acc all-NaN over {acc.size} vals; rmse all-finite over {rmse.size} vals"
    assert n_non_nan_acc == 0, f"{msg} -- {n_non_nan_acc} non-NaN acc value(s) found"
    assert n_non_finite_rmse == 0, f"{msg} -- {n_non_finite_rmse} non-finite rmse value(s) found"
    return msg


def check_g(ctx: Ctx) -> str:
    """Maps wet/dry: sst rmse finite everywhere (min > 0); temp_450 rmse finite at
    every mesh-wet node (NaN, if any, confined to mesh-dry nodes). SKIP if the maps
    cache is absent or lead L isn't among its leads."""
    maps_path = ctx.cache / f"maps_{ctx.stream}.nc"
    if not maps_path.exists():
        raise SkipCheckError(f"{maps_path} not found")
    with xr.open_dataset(maps_path) as ds:
        avail = [int(x) for x in ds["lead"].values]
        if ctx.lead not in avail:
            raise SkipCheckError(f"lead {ctx.lead} not in {maps_path} leads {avail}")
        sst_rmse = ds["rmse"].sel(lead=ctx.lead, channel=ctx.sfc).values
        t450_rmse = ds["rmse"].sel(lead=ctx.lead, channel="temp_450").values

    # only the points the target holds at (s*, L) (all of them unless subsampled)
    sampled = np.zeros(ctx.dry_mask_450.size, dtype=bool)
    sampled[ctx.idx] = True
    dry, wet = ctx.dry_mask_450 & sampled, ~ctx.dry_mask_450 & sampled
    sst_rmse = sst_rmse[sampled]
    sst_finite = bool(np.all(np.isfinite(sst_rmse)))
    sst_min = float(np.min(sst_rmse))
    t450_wet_nan = int(np.sum(np.isnan(t450_rmse[wet])))
    t450_dry_nan = int(np.sum(np.isnan(t450_rmse[dry])))

    msg = (
        f"lead={ctx.lead}: {ctx.sfc} rmse finite={sst_finite} min={sst_min:.3g}; "
        f"temp_450 nan at wet={t450_wet_nan}, at dry={t450_dry_nan}/{int(dry.sum())}"
    )
    assert sst_finite, f"{msg} -- {ctx.sfc} rmse has non-finite value(s)"
    assert sst_min > 0, f"{msg} -- {ctx.sfc} rmse global min not > 0"
    assert t450_wet_nan == 0, f"{msg} -- temp_450 rmse NaN at a wet node"
    return msg


def check_h(ctx: Ctx) -> str:
    """Meta consistency: meta.json sample ids == each stream's scores nc sample
    coord; n_samples attr of maps nc == number of samples in scores nc."""
    meta_path = ctx.cache / "meta.json"
    if not meta_path.exists():
        raise SkipCheckError(f"{meta_path} not found")
    meta = json.loads(meta_path.read_text())
    meta_ids = [int(s) for s in meta["sample_ids"]]

    parts, ran_any = [], False
    for stream in ctx.cfg["streams"]:
        path = ctx.cache / f"scores_{stream}.nc"
        if not path.exists():
            continue
        ran_any = True
        with xr.open_dataset(path) as ds:
            ids = [int(x) for x in ds["sample"].values]
        assert ids == meta_ids, (
            f"scores_{stream}.nc sample coord ({len(ids)} ids) != "
            f"meta.json sample_ids ({len(meta_ids)} ids)"
        )
        parts.append(f"{stream} sample ids match ({len(ids)})")
    if not ran_any:
        raise SkipCheckError("no scores_*.nc found for any configured stream")

    maps_path = ctx.cache / f"maps_{ctx.stream}.nc"
    if maps_path.exists():
        with xr.open_dataset(maps_path) as ds:
            n_samples = int(ds.attrs["n_samples"])
        assert n_samples == len(meta_ids), (
            f"maps_{ctx.stream}.nc n_samples attr ({n_samples}) != "
            f"scores sample count ({len(meta_ids)})"
        )
        parts.append(f"maps n_samples == {n_samples}")
    else:
        parts.append(f"maps_{ctx.stream}.nc absent (n_samples cross-check skipped)")
    return "; ".join(parts)


def check_i(ctx: Ctx) -> str:
    """Baseline maps (Task 14): sst `acc` lies within [-1.000001, 1.000001] at all
    (sst has no dry points -- a 2-D channel, mesh-wet everywhere) nodes and is
    non-NaN on > 99% of them; sst `rmse_persistence` is >= 0 and finite everywhere;
    and the area-weighted (`nod_area[0]`) mean of pointwise `rmse_persistence**2`
    matches the sample-aggregated persistence sst rmse**2 from the scores cache at
    the same lead within 5% -- the same statistic (mean over samples and points),
    computed in the opposite order. SKIP if the maps cache lacks the Task 14
    variables (older caches predate them)."""
    maps_path = ctx.cache / f"maps_{ctx.stream}.nc"
    if not maps_path.exists():
        raise SkipCheckError(f"{maps_path} not found")
    with xr.open_dataset(maps_path) as ds:
        missing = [v for v in _TASK14_MAP_VARS if v not in ds.data_vars]
        if missing:
            raise SkipCheckError(f"{maps_path} lacks {missing} (pre-Task-14 cache)")
        avail = [int(x) for x in ds["lead"].values]
        if ctx.lead not in avail:
            raise SkipCheckError(f"lead {ctx.lead} not in {maps_path} leads {avail}")
        acc = ds["acc"].sel(lead=ctx.lead, channel=ctx.sfc).values.astype(np.float64)[ctx.idx]
        rmse_p = ds["rmse_persistence"].sel(lead=ctx.lead, channel=ctx.sfc).values.astype(
            np.float64
        )[ctx.idx]

    n_pts = acc.size
    acc_nan = np.isnan(acc)
    n_acc_nan = int(acc_nan.sum())
    frac_nonnan = 1.0 - n_acc_nan / n_pts
    # NaN comparisons are False in numpy, so an out-of-range check must explicitly
    # exclude NaN entries (handled separately above) to avoid double-counting them.
    n_acc_oob = int(np.sum((np.abs(acc) > 1.000001) & ~acc_nan))

    rmse_p_bad = ~np.isfinite(rmse_p) | (rmse_p < 0)
    n_rmse_p_bad = int(rmse_p_bad.sum())

    w = ctx.mesh["nod_area0"][ctx.idx]
    map_mean_rmse2 = float(np.average(rmse_p**2, weights=w))

    persistence_rmse = ctx.scores_ds["score"].sel(
        model="persistence", metric="rmse", channel=ctx.sfc, lead=ctx.lead
    ).values.astype(np.float64)
    sample_mean_rmse2 = float(np.mean(persistence_rmse**2))
    rel_diff = abs(map_mean_rmse2 - sample_mean_rmse2) / sample_mean_rmse2

    msg = (
        f"lead={ctx.lead} {ctx.sfc}: acc oob={n_acc_oob}/{n_pts}, non-NaN={frac_nonnan:.4%}; "
        f"rmse_persistence bad(neg/non-finite)={n_rmse_p_bad}/{n_pts}; area-weighted "
        f"mean(rmse_persistence^2)={map_mean_rmse2:.6g} vs sample-aggregated "
        f"persistence rmse^2={sample_mean_rmse2:.6g} (rel diff={rel_diff:.4%})"
    )
    assert n_acc_oob == 0, f"{msg} -- acc out of [-1.000001, 1.000001] at a node"
    assert frac_nonnan > 0.99, f"{msg} -- acc non-NaN fraction <= 99%"
    assert n_rmse_p_bad == 0, f"{msg} -- rmse_persistence negative or non-finite at a node"
    assert rel_diff <= 0.05, f"{msg} -- exceeds 5% tolerance"
    return msg


def check_j(ctx: Ctx) -> str:
    """Region partition (Task 20): for (s*, sst, lead L, model=emulator), the
    global/nh/sh identity ``rmse_glob^2 * W_glob == rmse_nh^2 * W_nh + rmse_sh^2 *
    W_sh`` holds within 1e-6 relative error, where ``W_r`` is the summed area
    weight of region ``r`` -- recomputed independently here from mesh
    lat/nod_area (NOT `meshtools.masks_and_weights`/`scores._region_mask`), using
    `cfg["regions"]`'s lat bounds. sst is a 2-D, mesh-wet-everywhere channel (see
    check (b)), so its weight is simply `nod_area[0]` at every node, unmasked.

    This holds because `metrics.score_block`'s rmse satisfies
    ``rmse^2 * sum(W) == sum(W * err^2)`` exactly, and the region weights partition
    the mesh (every node's weight counts toward exactly one of nh/sh, see
    `scores._region_mask`), so the nh/sh weighted sums of squared error add up to
    the global one.

    SKIPs if the cached scores have no `region` dimension (pre-Task-20 cache).
    """
    if "region" not in ctx.scores_ds_raw.dims:
        raise SkipCheckError("cached scores have no region dim (pre-Task-20 cache)")

    s, lead = ctx.s_star, ctx.lead
    lat = ctx.mesh["lat"][ctx.idx]
    w = ctx.mesh["nod_area0"][ctx.idx]

    def region_weight(bounds) -> float:
        if bounds is None:
            m = np.ones_like(lat, dtype=bool)
        else:
            south, north = bounds
            m = (lat >= south) & (lat < north)
        return float(w[m].sum())

    W = {name: region_weight(bounds) for name, bounds in ctx.cfg["regions"].items()}

    def rmse(region: str) -> float:
        return float(
            ctx.scores_ds_raw["score"].sel(
                sample=s, lead=lead, channel=ctx.sfc, metric="rmse", model="emulator",
                region=region,
            )
        )

    rmse_glob, rmse_nh, rmse_sh = rmse("global"), rmse("nh"), rmse("sh")
    lhs = rmse_glob**2 * W["global"]
    rhs = rmse_nh**2 * W["nh"] + rmse_sh**2 * W["sh"]
    rel_err = abs(lhs - rhs) / abs(lhs) if lhs else abs(lhs - rhs)

    msg = (
        f"s={s} {ctx.sfc} lead={lead}: rmse(global={rmse_glob:.6g}, nh={rmse_nh:.6g}, "
        f"sh={rmse_sh:.6g}); W(global={W['global']:.6g}, nh={W['nh']:.6g}, "
        f"sh={W['sh']:.6g}); glob^2*W_glob={lhs:.6g} vs nh^2*W_nh+sh^2*W_sh={rhs:.6g} "
        f"(rel_err={rel_err:.3g})"
    )
    assert rel_err < 1e-6, f"{msg} -- exceeds 1e-6 relative tolerance"
    return msg


CHECKS = [("a", check_a), ("b", check_b), ("c", check_c), ("d", check_d),
          ("e", check_e), ("f", check_f), ("g", check_g), ("h", check_h),
          ("i", check_i), ("j", check_j)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=argparse.SUPPRESS,  # hidden test-only override, see module docstring
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        cache = Path(args.cache_dir) if args.cache_dir else cache_dir(cfg)
        check_cfg = cfg.get("selfcheck") or {}
        stream = check_cfg.get("stream", STREAM)
        sfc = check_cfg.get("surface_channel", SURFACE_CHANNEL)
        scores_path = cache / f"scores_{stream}.nc"
        if not scores_path.exists():
            raise FileNotFoundError(f"{scores_path} not found -- nothing to verify")
        scores_ds_raw = xr.open_dataset(scores_path)
        # Backward/forward compat (Task 20): a cache with a `region` dimension is
        # narrowed to region="global" here, once, so every pre-existing check (a,
        # c, d, f, g, h, i) keeps reading `scores_ds` exactly as before -- a cache
        # without the dimension (pre-Task-20) passes through unchanged. Only check
        # (j) needs every region, via `scores_ds_raw`.
        scores_ds = (
            scores_ds_raw.sel(region="global")
            if "region" in scores_ds_raw.dims
            else scores_ds_raw
        )
        root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
        mesh = _load_mesh(cfg["mesh_path"])
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None

    s_star = int(scores_ds["sample"].values[0])
    avail_leads = [int(x) for x in scores_ds["lead"].values]
    lead = 15 if 15 in avail_leads else max(avail_leads)
    max_lead = max(avail_leads)

    level_idx_450 = _level_idx(mesh["nz1"], 450)
    dry_mask_450 = mesh["n_wet_layers_nod"] <= level_idx_450

    ctx = Ctx(
        cfg=cfg, cache=cache, root=root, mesh=mesh, scores_ds=scores_ds,
        scores_ds_raw=scores_ds_raw,
        s_star=s_star, lead=lead, max_lead=max_lead,
        level_idx_450=level_idx_450, dry_mask_450=dry_mask_450,
        stream=stream, sfc=sfc, idx=_target_positions(root, s_star, stream, lead),
    )

    results = []
    for cid, fn in CHECKS:
        try:
            msg = fn(ctx)
            results.append(CheckResult(cid, "PASS", msg))
        except SkipCheckError as e:
            results.append(CheckResult(cid, "SKIP", str(e)))
        except AssertionError as e:
            results.append(CheckResult(cid, "FAIL", str(e)))
        except Exception as e:  # unexpected bug in a check -- still report, don't crash
            results.append(CheckResult(cid, "FAIL", f"unexpected error: {e!r}"))

    for r in results:
        print(f"[{r.status}] ({r.id}) {CHECK_NAMES[r.id]}: {r.message}")

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed (of {len(results)} checks)")

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
