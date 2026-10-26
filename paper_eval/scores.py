"""Stage-1 scoring worker: per-sample scores and spatial accumulators for the paper
evaluation pipeline.

Runs inside a `ProcessPoolExecutor` (Task 8's `compute.py` owns the pool): each worker
calls `init_worker(cfg)` once, as the pool initializer, then `process_block(sample_ids)`
once per task. Everything needed to score a block -- the merged zarr root, mesh masks
and weights, climatology lookups -- is opened once per worker into module globals and
reused across calls; nothing here spawns its own parallelism.

Spectra are NOT computed in this pass (moved to a separate implicit_filter-based
stage): no regridding, no PSD here.

Two geometries: `cfg.get("geometry", "fesom")` picks the branch. "fesom" (default,
every pre-Task-24 config) is FESOM's native CORE2 mesh: a fixed per-stream node/element
index shared by every sample and forecast step, with masks/weights/climatology looked
up once in `init_worker` (`_init_worker_fesom`/`_process_block_fesom`/
`_reduce_blocks_fesom`) -- unchanged behavior, verified by a bitwise regression check
against the production cache. "eerie" (`init_worker`/`process_block`/`reduce_blocks`'s
other branch, `_init_worker_eerie`/`_process_block_eerie`/`_reduce_blocks_eerie`) is
run dbuzcpgf's OCEAN_2D/OCEAN_3D streams: `eerie_grid.EerieGrid`'s fixed 0.25 deg
lat/lon grid, sparse target/prediction points re-drawn every (sample, forecast step),
no fixed mask/weight -- see `eerie_grid.py`'s module docstring and this module's
`_init_worker_eerie`/`_process_block_eerie`/`_reduce_blocks_eerie`.
"""

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from paper_eval import eerie_grid
from paper_eval.climatology import ChannelClimLookup, ClimLookup
from paper_eval.config import (
    apply_normalization_overrides,
    clim_dir,
    clim_paths,
    normalization_overrides,
    stream_entity,
    stream_layout,
)
from paper_eval.merged_zarr import open_merged_ranks
from paper_eval.meshtools import PointIndex, load_mesh, masks_and_weights
from paper_eval.metrics import METRICS, score_block

#: Model index order used throughout `scores`/`map_sums`.
MODELS = ("emulator", "persistence", "climatology")

#: Spatial accumulator fields kept per (stream, map lead) in `BlockResult.map_sums`.
#: The last five (persistence/climatology squared error, ACC cross-products) feed
#: `reduce_blocks`'s `rmse_persistence`/`rmse_climatology`/`acc` map variables.
_MAP_FIELDS = (
    "sum_err2", "sum_err", "sum_p", "sum_p2", "sum_t", "sum_t2",
    "sum_perr2", "sum_cerr2", "sum_fo", "sum_f2", "sum_o2",
)

#: EERIE geometry's BASE map accumulator fields: the same six sums as the first six of
#: `_MAP_FIELDS` plus `sum_perr2` (persistence squared error, feeds
#: `rmse_persistence`) -- always accumulated, regardless of `climatology.enabled`. A
#: per-(channel, cell) `count` int32 array (NOT float64, kept out of this tuple) is
#: accumulated alongside these and is the divisor `reduce_blocks` uses in place of
#: FESOM's fixed `n_total` (a given cell is covered by only a small fraction of any
#: one sample's sparse points, not every sample).
_MAP_FIELDS_EERIE = ("sum_err2", "sum_err", "sum_p", "sum_p2", "sum_t", "sum_t2", "sum_perr2")

#: EERIE geometry's CLIMATOLOGY-DEPENDENT map accumulator fields (Task 25): the same
#: four climatology cross-products FESOM's `_MAP_FIELDS` always carries, added to
#: `_MAP_FIELDS_EERIE`'s allocation only when `cfg["climatology"]["enabled"]` is true
#: (there is no climatology field to accumulate against when it's false) --
#: `_reduce_blocks_eerie` computes real `rmse_climatology`/`acc` when these are
#: present in a block's `map_sums`, else fills NaN directly (unchanged Task 24
#: behavior), so the `enabled: false` path never allocates or touches these at all.
_MAP_FIELDS_EERIE_CLIM = ("sum_cerr2", "sum_fo", "sum_f2", "sum_o2")

#: Above this fraction of valid (finite-target) points whose persistence gather
#: (`_gather_persistence`) misses the source grid (`cell not covered by source`),
#: `_process_block_eerie` raises rather than silently tolerating a systematic
#: coverage problem (e.g. a grid-alignment bug) -- verified data fact: run dbuzcpgf's
#: actual miss rate is exactly 0 (source covers every wet cell sampled by target), so
#: this is a generous sanity bound, not a tuned expectation.
_PERSISTENCE_MISS_TOLERANCE = 0.005


@dataclasses.dataclass
class BlockResult:
    """Result of scoring one (time-ordered) block of global sample ids."""

    sample_ids: list[int]
    #: stream -> (3 models, n_regions, 3 metrics, n_block_samples, n_leads, nchan)
    #: float64 -- region order is `cfg["regions"]`'s config order (global first)
    scores: dict[str, np.ndarray]
    #: (stream, lead) -> {field: (nchan, npts) float64}, field in `_MAP_FIELDS`
    map_sums: dict[tuple[str, int], dict[str, np.ndarray]]
    #: samples accumulated into map_sums (== len(sample_ids))
    n_map_samples: int


# ── Worker-global state, populated once by init_worker ────────────────────────────────
# (ProcessPoolExecutor initializer pattern: one call per worker process; also callable
# directly, e.g. in a single-process test.)
_cfg = None
#: `cfg.get("geometry", "fesom")`, cached by `init_worker` so `process_block` doesn't
#: need to look it up (and stays valid after `_cfg` is reassigned by a later
#: `init_worker` call in the same process, e.g. `--n-workers 1`'s in-process path).
_geometry: str = "fesom"
_root = None
_mesh = None
_channels: dict[str, list[str]] = {}
_mask: dict[str, np.ndarray] = {}
_w: dict[str, np.ndarray] = {}
_clim: dict[str, ClimLookup] = {}
_src_idx: dict[str, np.ndarray | None] = {}
_src_mean: dict[str, np.ndarray] = {}
_src_std: dict[str, np.ndarray] = {}
#: stream -> its full point set (lookup for targets subsampled with `max_num_targets`)
_points: dict[str, PointIndex] = {}
_leads: list[int] = []
_map_leads: list[int] = []
#: Region name order, config order (`cfg["regions"]`), global first.
_regions: list[str] = []
#: stream -> region -> (nchan, npts) float32 = `_w[stream] * region_mask` -- the
#: per-region weight variant `process_block` feeds to `score_block` in place of
#: `_w[stream]` (the wet mask itself is region-independent and stays unchanged).
_region_w: dict[str, dict[str, np.ndarray]] = {}

#: EERIE-only worker globals (populated by `_init_worker_eerie`, unused/empty on the
#: "fesom" branch). `_eerie_grid` is shared by every EERIE stream of a run (one
#: physical 0.25 deg earth grid): unlike `_mask`/`_w`/`_region_w` above, there is no
#: per-stream fixed mask/weight to cache here -- `_process_block_eerie` builds
#: validity/weights fresh per (sample, lead) since the sparse points themselves
#: change every time (see `eerie_grid.py`'s module docstring).
_eerie_grid: eerie_grid.EerieGrid | None = None
#: `(stream,)` keys already `[warn]`-logged by `_process_block_eerie`'s persistence
#: miss check, so a nonzero (but within-tolerance) miss count is reported once per
#: worker per stream rather than once per (sample, lead).
_persistence_miss_warned: set[str] = set()


def _clean(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-fill NaNs at masked-out (dry) points before scoring.

    `score_block` weights every point by `w * mask`, but a weight of zero does not
    stop `0 * NaN` from producing NaN (IEEE-754) -- so a NaN fill value at a dry
    point (observed in the real archives, e.g. FESOM_NODE 'salt_35'..'salt_95')
    silently poisons every weighted sum for that channel, even though the mesh mask
    already excludes the point. Dry points are excluded from `score_block`'s sums
    regardless of their value, so replacing NaN there with 0 is safe. NaN at a
    *wet* point would be a genuine data problem and is deliberately left in place
    (raising) rather than silently hidden.
    """
    nan_at_wet = np.isnan(field) & mask.T
    if nan_at_wet.any():
        raise ValueError("NaN found at a wet (masked-in) point -- real data problem")
    return np.where(np.isnan(field), 0.0, field)


def _region_mask(lat: np.ndarray, bounds: list[float] | None) -> np.ndarray:
    """Per-point region membership from point latitude (degrees).

    `bounds=None` -> every point (e.g. the `global` region: whole mesh).
    `bounds=[south, north]` -> half-open `[south, north)` -- south-inclusive,
    north-exclusive -- so two touching regions (e.g. `nh: [0, 90]`, `sh: [-90,
    0]`) partition the mesh with no point double-counted or orphaned at the
    shared boundary: a point at exactly `lat=0` belongs to `nh`, not `sh`. No
    real mesh point sits at exactly `lat=90` (verified against the CORE2 mesh:
    max lat ~89.96), so the exclusive north bound never drops a pole point.
    """
    lat = np.asarray(lat)
    if bounds is None:
        return np.ones(lat.shape, dtype=bool)
    south, north = bounds
    return (lat >= south) & (lat < north)


def _load_norm_stats(cfg: dict, stream: str, src_channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(mean_vec, std_vec) float32, aligned to `src_channels` order -- the same
    per-channel normalization constants the training reader applied to build
    `source`, read once from the first truth-year store's `data` array attrs
    (never read per sample).

    The truth store's `colnames` carries two leading, non-normalized coordinate
    columns ('lon', 'lat') that `means`/`std` don't cover (`len(colnames) ==
    len(means) + 2`); `offset` accounts for that when looking a channel's name up
    in `colnames` to index into `means`/`std`.

    The archive statistics are what the reader used unless the evaluated run sets the
    stream option `normalization_stats` (wet-point statistics; see
    `paper_eval.config.normalization_overrides`): then the reader replaced them,
    per channel, by the JSON's values, and so does this -- in the reader's order
    (archive near-zero-std rule first, overrides after). Without the option the
    result is exactly the archive statistics. `snapshots.py` and, through
    `scores.init_worker`, `compute_damped.py` take their statistics from here.
    """
    truth_dir = Path(cfg["truth"][stream])
    files = sorted(truth_dir.glob("*.zarr"))
    if not files:
        raise FileNotFoundError(f"no truth store (*.zarr) found in {truth_dir}")

    g = zarr.open_group(str(files[0]), mode="r")
    attrs = g["data"].attrs.asdict()
    colnames = list(attrs["colnames"])
    means = np.asarray(attrs["means"], dtype=np.float32)
    std = np.asarray(attrs["std"], dtype=np.float32)
    offset = len(colnames) - len(means)

    mean_vec = np.empty(len(src_channels), dtype=np.float32)
    std_vec = np.empty(len(src_channels), dtype=np.float32)
    for i, name in enumerate(src_channels):
        pos = colnames.index(name) - offset
        if not (0 <= pos < len(means)):
            raise ValueError(
                f"stream {stream!r}: channel {name!r} has no normalization stats "
                f"(colnames/means offset mismatch in {files[0]})"
            )
        mean_vec[i] = means[pos]
        std_vec[i] = std[pos]

    # Training reader's rule: a near-zero std is treated as 1.0 (no-op) rather than
    # inverted, to avoid blowing up a constant (or near-constant) channel.
    std_vec = np.where(std_vec <= 1e-5, 1.0, std_vec).astype(np.float32)

    overrides = normalization_overrides(cfg, stream)
    if overrides:
        # `src_channels` names the entries of the aligned vectors (the lon/lat offset
        # of the archive's colnames is already resolved above).
        mean_o, std_o = apply_normalization_overrides(src_channels, mean_vec, std_vec, overrides)
        mean_vec, std_vec = mean_o.astype(np.float32), std_o.astype(np.float32)
    return mean_vec, std_vec


def full_point_coords(root, sample, stream) -> np.ndarray:
    """(npts, 2) [lat, lon] of `stream`'s full point set: the fstep-1 target's coords
    (as always) unless the target is a subset (`max_num_targets`), then the fstep-0
    source's, which always covers every point in mesh order."""
    tar_coords = root[f"{sample}/{stream}/1/target"]["coords"]
    src_coords = root[f"{sample}/{stream}/0/source"]["coords"]
    if tar_coords.shape[0] == src_coords.shape[0]:
        return np.asarray(tar_coords)
    return np.asarray(src_coords)


def _compute_leads(cfg: dict) -> tuple[list[int], list[int]]:
    """(leads, map_leads) from cfg, honouring the CLI-subset override.

    Shared by `init_worker` (in every worker) and `reduce_blocks` (in the parent) so
    both sides agree on the lead lists without the parent seeing worker globals.
    """
    override = cfg.get("_leads_override")
    leads = list(override) if override is not None else list(range(1, cfg["leads"]["max"] + 1))
    map_leads = [lead for lead in cfg["leads"]["map_leads"] if lead in leads]
    return leads, map_leads


def init_worker(cfg: dict) -> None:
    """Dispatch to `_init_worker_fesom` or `_init_worker_eerie` by
    `cfg.get("geometry", "fesom")` -- see this module's docstring. Runs once per
    worker process (`ProcessPoolExecutor` initializer); idempotent.
    """
    global _cfg, _geometry
    _cfg = cfg
    _geometry = cfg.get("geometry", "fesom")
    if _geometry == "eerie":
        _init_worker_eerie(cfg)
    else:
        _init_worker_fesom(cfg)


def _init_worker_fesom(cfg: dict) -> None:
    """Open the merged zarr root, mesh masks/weights and climatology lookups for every
    configured stream, into module globals.

    Runs once per worker process; idempotent (safe to call again in the same process).
    """
    global _cfg, _root, _mesh, _channels, _mask, _w, _clim
    global _src_idx, _src_mean, _src_std, _leads, _map_leads, _regions, _region_w, _points

    _cfg = cfg
    _root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    _mesh = load_mesh(cfg["mesh_path"])

    _channels, _mask, _w, _clim = {}, {}, {}, {}
    _src_idx, _src_mean, _src_std, _points = {}, {}, {}, {}
    _regions = list(cfg["regions"])
    _region_w = {}
    s0 = _root.store.samples[0]
    for stream in cfg["streams"]:
        channels = list(_root[f"{s0}/{stream}/1/target"].attrs["channels"])
        _channels[stream] = channels

        # Per-variable streams (TEMP, SALT, ...) map to their mesh entity through
        # `stream_layout`; FESOM_NODE/FESOM_ELEM resolve to node/elem as before. The
        # mesh mask is the only wet/dry authority: dry predictions (unconstrained for
        # a model trained on mesh-masked targets) get zero weight in every score.
        mask, w = masks_and_weights(_mesh, stream_entity(cfg, stream), channels)
        _mask[stream] = mask
        _w[stream] = w

        # Region weight variants: `w * region_mask`, broadcast over channels. The
        # region mask is built from point latitude, taken from the zarr `coords`
        # array of the reference target group (`[:, 0]` = lat, same convention
        # `reduce_blocks` uses for `maps_{stream}`'s `lat` coord) rather than
        # `_mesh` directly -- this covers FESOM_ELEM (element centroids) the same
        # way as FESOM_NODE (node points) with no per-stream branching here.
        pts = full_point_coords(_root, s0, stream)
        _points[stream] = PointIndex(pts)
        lat = pts[:, 0]
        _region_w[stream] = {
            region: (w * _region_mask(lat, bounds)[np.newaxis, :]).astype(np.float32)
            for region, bounds in cfg["regions"].items()
        }

        if stream_layout(cfg, stream) is not None:
            # channels picked by name from the stream's climatology sources
            _clim[stream] = ChannelClimLookup(clim_paths(cfg, stream), channels)
        else:
            _clim[stream] = ClimLookup(clim_paths(cfg, stream)[0], channels)

        # The source (model input) can carry channels the target/prediction don't --
        # observed for FESOM_ELEM, whose source has one extra column ('v_210') not
        # present in target/prediction. Align source columns to the reference channel
        # order by name so the persistence baseline lines up with mask/w/target,
        # which are all built from `channels`.
        src_channels = list(_root[f"{s0}/{stream}/0/source"].attrs["channels"])
        if src_channels == channels:
            _src_idx[stream] = None
        else:
            missing = [c for c in channels if c not in src_channels]
            if missing:
                raise ValueError(
                    f"stream {stream!r}: source is missing channel(s) required by "
                    f"target/prediction: {missing}"
                )
            _src_idx[stream] = np.array([src_channels.index(c) for c in channels])

        # `source` is the model's normalized input, not a physical field -- denormalize
        # it (fix round 1) using the training reader's own per-channel mean/std,
        # stored in the truth archive's attrs, before it can serve as a persistence
        # baseline against physical-unit target/prediction. Stats are built aligned to
        # `src_channels` (source's own order) then reduced to reference order with the
        # same `_src_idx` used for the source data itself.
        mean_full, std_full = _load_norm_stats(cfg, stream, src_channels)
        if _src_idx[stream] is not None:
            _src_mean[stream] = mean_full[_src_idx[stream]]
            _src_std[stream] = std_full[_src_idx[stream]]
        else:
            _src_mean[stream] = mean_full
            _src_std[stream] = std_full

    _leads, _map_leads = _compute_leads(cfg)


def process_block(sample_ids: list[int]) -> BlockResult:
    """Dispatch to `_process_block_fesom` or `_process_block_eerie` by the geometry
    `init_worker` cached in `_geometry` -- see this module's docstring."""
    if _root is None:
        raise RuntimeError("init_worker() must be called before process_block()")
    if _geometry == "eerie":
        return _process_block_eerie(sample_ids)
    return _process_block_fesom(sample_ids)


def _process_block_fesom(sample_ids: list[int]) -> BlockResult:
    """Score one block of global sample ids against persistence and climatology
    baselines, accumulating spatial sums at `_map_leads`."""

    n_samples, n_leads = len(sample_ids), len(_leads)
    streams = _cfg["streams"]

    scores = {
        stream: np.empty(
            (len(MODELS), len(_regions), len(METRICS), n_samples, n_leads, len(_channels[stream])),
            dtype=np.float64,
        )
        for stream in streams
    }
    map_sums = {
        (stream, lead): {
            field: np.zeros((len(_channels[stream]), _mask[stream].shape[1]), dtype=np.float64)
            for field in _MAP_FIELDS
        }
        for stream in streams
        for lead in _map_leads
    }

    for i, s in enumerate(sample_ids):
        init_date = _root.store.init_times[s]
        for stream in streams:
            channels_ref = _channels[stream]
            mask = _mask[stream]
            region_w = _region_w[stream]

            src_raw = np.asarray(_root[f"{s}/{stream}/0/source/data"])
            if _src_idx[stream] is not None:
                src_raw = src_raw[:, _src_idx[stream]]
            # Denormalize: source is stored z-score normalized by the training reader;
            # persistence needs it in the same physical units as target/prediction.
            src_phys = (src_raw * _src_std[stream] + _src_mean[stream]).astype(np.float32)
            src_clean = _clean(src_phys, mask)

            for j, lead in enumerate(_leads):
                tar_group = _root[f"{s}/{stream}/{lead}/target"]
                pred_group = _root[f"{s}/{stream}/{lead}/prediction"]
                if list(tar_group.attrs["channels"]) != channels_ref:
                    raise ValueError(
                        f"sample {s} stream {stream} lead {lead}: target channels drifted "
                        "from the reference list"
                    )
                if list(pred_group.attrs["channels"]) != channels_ref:
                    raise ValueError(
                        f"sample {s} stream {stream} lead {lead}: prediction channels "
                        "drifted from the reference list"
                    )

                tar = np.asarray(tar_group["data"])
                pred = np.asarray(pred_group["data"])
                if pred.shape[-1] != 1:
                    raise ValueError(
                        f"sample {s} stream {stream} lead {lead}: expected ens==1, "
                        f"got {pred.shape[-1]}"
                    )
                pred = pred[..., 0]

                # A target subsampled with `max_num_targets` is scattered onto the full
                # point set (NaN where not sampled); only the sampled wet points are
                # scored. Full-size targets pass through unchanged (`mask_l is mask`).
                mask_l = mask
                if tar.shape[0] != _points[stream].npts:
                    if not np.array_equal(tar_group["coords"][:], pred_group["coords"][:]):
                        raise ValueError(
                            f"sample {s} stream {stream} lead {lead}: target and prediction "
                            "points differ"
                        )
                    tar, sampled = _points[stream].densify(tar, tar_group["coords"])
                    pred, _ = _points[stream].densify(pred, pred_group["coords"])
                    mask_l = mask & sampled[np.newaxis, :]

                c = _clim[stream].get(init_date + np.timedelta64(lead, "D"))

                # score_block's weighted sums are NaN-fragile (0 * NaN == NaN even
                # though the point's weight is 0) -- clean copies feed the kernel,
                # raw ones feed the (unmasked, per brief) spatial accumulators below.
                tar_clean, pred_clean, c_clean = (
                    _clean(tar, mask_l), _clean(pred, mask_l), _clean(c, mask_l)
                )

                # One score_block call per (model, region): same pred/tar/clim fields
                # and wet mask, only the weights vary (region_w[region] = w *
                # region_mask, see init_worker) -- CPU-only, ~3x score_block calls.
                for k, region in enumerate(_regions):
                    rw = region_w[region]
                    scores[stream][0, k, :, i, j, :] = score_block(pred_clean, tar_clean, c_clean, mask_l, rw)
                    scores[stream][1, k, :, i, j, :] = score_block(src_clean, tar_clean, c_clean, mask_l, rw)
                    scores[stream][2, k, :, i, j, :] = score_block(c_clean, tar_clean, c_clean, mask_l, rw)

                if lead in _map_leads:
                    sums = map_sums[(stream, lead)]
                    err = (pred - tar).T.astype(np.float64)
                    pred_t = pred.T.astype(np.float64)
                    tar_t = tar.T.astype(np.float64)
                    sums["sum_err"] += err
                    sums["sum_err2"] += err**2
                    sums["sum_p"] += pred_t
                    sums["sum_p2"] += pred_t**2
                    sums["sum_t"] += tar_t
                    sums["sum_t2"] += tar_t**2

                    # Baseline-error and ACC accumulators (Task 14), same raw
                    # (un-`_clean`-ed) fields as above -- dry-point NaN/0 behaviour is
                    # masked downstream at plot time (D3), not here.
                    perr = (src_phys - tar).T.astype(np.float64)
                    cerr = (c - tar).T.astype(np.float64)
                    f_prime = (pred - c).T.astype(np.float64)
                    o_prime = (tar - c).T.astype(np.float64)
                    sums["sum_perr2"] += perr**2
                    sums["sum_cerr2"] += cerr**2
                    sums["sum_fo"] += f_prime * o_prime
                    sums["sum_f2"] += f_prime**2
                    sums["sum_o2"] += o_prime**2

    return BlockResult(
        sample_ids=list(sample_ids), scores=scores, map_sums=map_sums, n_map_samples=n_samples
    )


def merge_map_sums(agg: dict | None, r: BlockResult) -> dict:
    """Incrementally fold one `BlockResult.map_sums` into a running total, so the
    parent (`compute.py`) never holds more than the running total plus the block
    just merged -- not every block's map_sums simultaneously (map_sums is now 11
    (nchan, npts) float64 accumulators per (stream, lead), the memory-heavy part of
    a `BlockResult`).

    `agg is None` (first call, e.g. before the first block): adopts `r.map_sums`
    itself in place as the running total (no copy) and seeds a running sample count
    under the reserved key `"n_map_samples"` (a plain int, alongside the
    `(stream, lead)` tuple keys -- no collision). Every later call adds `r`'s arrays
    into `agg`'s in place and bumps that count. Callers must strip `r.map_sums`
    (e.g. `r.map_sums = {}`) themselves right after this call -- `merge_map_sums`
    does not mutate `r`.

    The running count lets `reduce_blocks` normalize the merged sums without
    needing the (by-then-stripped) per-block `map_sums`. When `r.map_sums` is empty
    (no map leads configured), this degenerates to just tracking the count.
    """
    if agg is None:
        agg = r.map_sums
        agg["n_map_samples"] = r.n_map_samples
        return agg
    for key, fields in r.map_sums.items():
        dst = agg[key]
        for field, arr in fields.items():
            dst[field] += arr
    agg["n_map_samples"] += r.n_map_samples
    return agg


def reduce_blocks(
    score_results: list[BlockResult], merged_maps: dict | None, cfg: dict
) -> dict[str, xr.Dataset]:
    """Dispatch to `_reduce_blocks_fesom` or `_reduce_blocks_eerie` by
    `cfg.get("geometry", "fesom")` -- see this module's docstring. Runs in the
    parent (`compute.py`), not a worker, so it reads `cfg` directly rather than the
    (worker-only) `_geometry` global.
    """
    if cfg.get("geometry", "fesom") == "eerie":
        return _reduce_blocks_eerie(score_results, merged_maps, cfg)
    return _reduce_blocks_fesom(score_results, merged_maps, cfg)


def _reduce_blocks_fesom(
    score_results: list[BlockResult], merged_maps: dict | None, cfg: dict
) -> dict[str, xr.Dataset]:
    """Concatenate block results (already time-ordered slices of the time-sorted
    sample list) into one (scores, maps) dataset pair per stream.

    `score_results` are the per-block results used for the `scores_{stream}`
    concatenation (`sample_ids`, `scores`) -- callers strip their (large) `map_sums`
    after folding each into `merged_maps` via `merge_map_sums`, so `score_results`
    must NOT be read for map data here. `merged_maps` is that already-merged dict
    (`(stream, lead) -> {field: array}` plus the running `"n_map_samples"` count),
    or `None` when no map lead was ever scored.

    Opens its own merged root for coords/init_times -- runs in the parent, not in a
    worker.

    `scores_{stream}` is always present, for every configured stream. `maps_{stream}`
    is present only when at least one map lead was actually scored -- i.e. when the
    effective `map_leads` (`cfg["leads"]["map_leads"]` intersected with the effective
    `leads`, see `_compute_leads`) is non-empty. This can be empty when a CLI `--leads`
    subset is fully disjoint from `map_leads` (e.g. `_leads_override=[5]` against the
    default `map_leads=[15, 30]`); in that case there is no lead dimension and no
    accumulated sums to build a maps dataset from, so it is omitted rather than
    invented empty. Callers must use `.get(f"maps_{stream}")` (or check membership),
    not assume the key exists.
    """
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    leads, map_leads = _compute_leads(cfg)
    regions = list(cfg["regions"])
    if map_leads and merged_maps is None:
        raise ValueError("merged_maps must not be None when map leads are configured")
    n_total = merged_maps["n_map_samples"] if merged_maps is not None else 0

    out = {}
    for stream in cfg["streams"]:
        s0 = root.store.samples[0]
        channels = list(root[f"{s0}/{stream}/1/target"].attrs["channels"])

        sample_ids = [s for r in score_results for s in r.sample_ids]
        # sample is axis 3 of each block's (model, region, metric, sample, lead,
        # channel) scores array -- see process_block/BlockResult.
        score_arr = np.concatenate([r.scores[stream] for r in score_results], axis=3)
        init_times = np.array(
            [root.store.init_times[s] for s in sample_ids], dtype="datetime64[ns]"
        )

        out[f"scores_{stream}"] = xr.Dataset(
            {
                "score": (
                    ("model", "region", "metric", "sample", "lead", "channel"),
                    score_arr.astype(np.float32),
                )
            },
            coords={
                "model": list(MODELS),
                "region": regions,
                "metric": list(METRICS),
                "sample": sample_ids,
                "init_time": ("sample", init_times),
                "lead": leads,
                "channel": channels,
            },
        )

        # No map lead was scored (e.g. a CLI --leads subset fully disjoint from
        # leads.map_leads) -- there is nothing to build a maps dataset from (no lead
        # dimension, no accumulated sums), so skip it entirely rather than inventing an
        # empty-dims dataset. See `reduce_blocks`' docstring.
        if not map_leads:
            continue

        pts = full_point_coords(root, s0, stream)
        dims = ("lead", "channel", "ipoint")
        by_var = {
            name: []
            for name in (
                "rmse", "bias", "pred_mean", "pred_var", "target_mean", "target_var",
                "rmse_persistence", "rmse_climatology", "acc",
            )
        }
        for lead in map_leads:
            sums = merged_maps[(stream, lead)]
            pred_mean = sums["sum_p"] / n_total
            target_mean = sums["sum_t"] / n_total
            by_var["rmse"].append(np.sqrt(sums["sum_err2"] / n_total))
            by_var["bias"].append(sums["sum_err"] / n_total)
            by_var["pred_mean"].append(pred_mean)
            by_var["pred_var"].append(np.clip(sums["sum_p2"] / n_total - pred_mean**2, 0, None))
            by_var["target_mean"].append(target_mean)
            by_var["target_var"].append(
                np.clip(sums["sum_t2"] / n_total - target_mean**2, 0, None)
            )
            by_var["rmse_persistence"].append(np.sqrt(sums["sum_perr2"] / n_total))
            by_var["rmse_climatology"].append(np.sqrt(sums["sum_cerr2"] / n_total))
            # sum_f2 * sum_o2 is 0 at a degenerate point (e.g. a dry cell where both
            # prediction and target equal climatology identically) -> 0/0 -> NaN,
            # silently (brief's requirement, mirrors metrics.score_block's acc).
            with np.errstate(invalid="ignore", divide="ignore"):
                acc = sums["sum_fo"] / np.sqrt(sums["sum_f2"] * sums["sum_o2"])
            by_var["acc"].append(acc)

        out[f"maps_{stream}"] = xr.Dataset(
            {name: (dims, np.stack(arrs).astype(np.float32)) for name, arrs in by_var.items()},
            coords={
                "lead": map_leads,
                "channel": channels,
                "lat": ("ipoint", pts[:, 0]),
                "lon": ("ipoint", pts[:, 1]),
            },
            attrs={"n_samples": n_total, "run_id": cfg["run_id"]},
        )

    return out


# ── EERIE geometry branch (Task 24) ────────────────────────────────────────────────────
def _init_worker_eerie(cfg: dict) -> None:
    """EERIE geometry counterpart of `_init_worker_fesom`: open the merged zarr root
    and the fixed 0.25 deg grid (`eerie_grid.load_grid`, shared by every EERIE stream
    -- both OCEAN_2D and OCEAN_3D are the same physical earth grid), plus per-stream
    reference channel lists and normalization stats.

    No mesh, no fixed mask/weight -- every EERIE stream's points are re-drawn per
    (sample, forecast step), so masks/weights are built fresh in `_process_block_eerie`
    instead of cached here. `ClimLookup` (Task 25) IS loaded here, per stream, when
    `cfg["climatology"]["enabled"]` is true -- same `ClimLookup`, same
    `clim_dir(cfg) / f"clim_{stream}_{ref_start}-{ref_end}.zarr"` path convention
    `_init_worker_fesom` already uses (the EERIE clim zarr, built by
    `build_clim_eerie.py`, is written in that exact layout -- see
    `paper_eval/build_clim_eerie.py`'s module docstring); left empty when disabled, so
    `_clim_lookup` returns `None` (see `_clim_lookup`'s docstring).
    """
    global _root, _channels, _src_idx, _src_mean, _src_std, _leads, _map_leads, _regions
    global _eerie_grid, _persistence_miss_warned, _clim

    _root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    _eerie_grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])

    _channels, _src_idx, _src_mean, _src_std, _clim = {}, {}, {}, {}, {}
    _regions = list(cfg["regions"])
    _persistence_miss_warned = set()

    s0 = _root.store.samples[0]
    for stream in cfg["streams"]:
        channels = list(_root[f"{s0}/{stream}/1/target"].attrs["channels"])
        _channels[stream] = channels

        # Unlike FESOM_ELEM's source (one extra 'v_210' column not in target/
        # prediction), EERIE's source channel list equals target/prediction's
        # exactly (verified against run dbuzcpgf) -- no by-name reindexing needed;
        # `_src_idx[stream] = None` mirrors `_init_worker_fesom`'s "no reorder
        # needed" case so `_gather_persistence` can stay geometry-agnostic-shaped.
        src_channels = list(_root[f"{s0}/{stream}/0/source"].attrs["channels"])
        if src_channels != channels:
            raise ValueError(
                f"stream {stream!r}: source channels {src_channels} != "
                f"target/prediction reference {channels} -- EERIE geometry assumes "
                "identical source/target channel lists (no FESOM-style extra-column "
                "handling implemented here)"
            )
        _src_idx[stream] = None

        mean_vec, std_vec = eerie_grid.load_norm_stats(cfg["truth"][stream], channels)
        _src_mean[stream] = mean_vec
        _src_std[stream] = std_vec

        # Task 25: load the EERIE clim zarr the same way _init_worker_fesom does --
        # `ClimLookup`'s (366, npts, nchan) layout is generic enough to read it
        # unmodified (build_clim_eerie.py writes exactly that layout on the full
        # grid, npts == _eerie_grid.n_cells). Left unpopulated when disabled, so a
        # later `_clim_lookup(stream, ...)` call returns None for this stream, same
        # as Task 24.
        if cfg["climatology"].get("enabled", False):
            y0, y1 = cfg["climatology"]["ref_start"], cfg["climatology"]["ref_end"]
            clim_path = clim_dir(cfg) / f"clim_{stream}_{y0}-{y1}.zarr"
            _clim[stream] = ClimLookup(clim_path, channels)

    _leads, _map_leads = _compute_leads(cfg)


def _clim_lookup(
    stream: str, init_date: np.datetime64, lead: int, coords: np.ndarray
) -> np.ndarray | None:
    """EERIE climatology hook (designed in Task 24, wired in Task 25).

    `None` when `cfg["climatology"]["enabled"]` is false (checked here rather than
    assumed, so a config that flips it on without `_init_worker_eerie` having loaded
    `_clim[stream]` fails with a clear `KeyError` from the dict lookup below instead
    of silently scoring with no climatology -- `_init_worker_eerie` and this function
    read the exact same `cfg["climatology"]["enabled"]` flag, so that can only happen
    if `init_worker` was skipped/stale).

    Otherwise: `_clim[stream].get(init_date + lead days)` (LRU-cached by
    day-of-year) is the day's climatology on the FULL grid, `(n_cells, nchan)`
    float32; gathered at `coords` via `_eerie_grid.cell_index` -- the same gather
    convention `_gather_persistence`/`_gather_at` use for the persistence baseline --
    giving `(npts, nchan)`, matching `target`'s own shape.
    """
    if not _cfg["climatology"].get("enabled", False):
        return None
    target_date = init_date + np.timedelta64(lead, "D")
    full_grid = _clim[stream].get(target_date)  # (n_cells, nchan) f32, LRU-cached, read-only
    cell_idx = _eerie_grid.cell_index(coords)
    return full_grid[cell_idx]  # fancy-index gather -> a fresh (npts, nchan) array


def _gather_persistence(grid: eerie_grid.EerieGrid, stream: str, sample: int) -> np.ndarray:
    """Read+denormalize `stream`'s fstep-0 source for `sample` (once per sample, per
    the brief) and scatter it onto the full grid keyed by cell index.

    Returns `(grid.n_cells, nchan)` float32, NaN at any cell the source doesn't
    cover -- ready for `_gather_at`'s per-lead gather. Source coords are the same
    every sample for a given stream in practice (verified against run dbuzcpgf), but
    `cell_index` is rebuilt every call rather than cached across samples, per the
    brief ("build cell_index for source cells once per sample").
    """
    src_group = _root[f"{sample}/{stream}/0/source"]
    src_channels = list(src_group.attrs["channels"])
    if src_channels != _channels[stream]:
        raise ValueError(
            f"sample {sample} stream {stream!r}: source channels drifted from the "
            f"reference list {_channels[stream]}"
        )
    src_raw = np.asarray(src_group["data"])
    # Denormalize: source is stored z-score normalized by the training reader;
    # persistence needs it in the same physical units as target/prediction (same
    # denormalization as `_load_norm_stats`'s FESOM counterpart).
    src_phys = (src_raw * _src_std[stream] + _src_mean[stream]).astype(np.float32)

    src_coords = np.asarray(src_group["coords"])
    src_cell_idx = grid.cell_index(src_coords)
    if np.unique(src_cell_idx).size != src_cell_idx.size:
        raise ValueError(
            f"sample {sample} stream {stream!r}: source has duplicate grid cell "
            "indices -- expected exactly one source point per wet cell"
        )

    src_by_cell = np.full((grid.n_cells, src_phys.shape[1]), np.nan, dtype=np.float32)
    src_by_cell[src_cell_idx] = src_phys
    return src_by_cell


def _gather_at(
    src_by_cell: np.ndarray, cell_idx: np.ndarray, tar: np.ndarray, mask: np.ndarray, stream: str
) -> np.ndarray:
    """Gather `src_by_cell` (from `_gather_persistence`) at `cell_idx` -- `(npts,
    nchan)`, matching `tar`'s shape; this lead's persistence baseline.

    Verified data fact: run dbuzcpgf's actual miss rate is exactly 0 (the source
    covers every wet cell target/prediction ever sample). A miss -- a valid (per
    `mask`) target point whose cell the source doesn't cover -- is nonetheless
    tolerated up to `_PERSISTENCE_MISS_TOLERANCE` (else raised: a miss rate that
    high is a likely grid-alignment bug, not the expected tiny gap) and reported
    once per worker per stream (`_persistence_miss_warned`). A missed point's
    persistence value falls back to the target's own value there (a defensible
    "persistence == truth" default that keeps the gathered field NaN-free without
    inventing an error) rather than either poisoning `score_block`'s weighted sums
    (`_clean`'s `0 * NaN == NaN` problem) or hard-failing on a documented, tolerated
    gap.
    """
    gathered = src_by_cell[cell_idx]  # (npts, nchan)
    miss = np.isnan(gathered) & mask.T
    if miss.any():
        n_miss = int(miss.any(axis=1).sum())
        n_valid = int(mask.any(axis=0).sum())
        frac = n_miss / max(1, n_valid)
        if frac > _PERSISTENCE_MISS_TOLERANCE:
            raise ValueError(
                f"stream {stream!r}: persistence gather miss rate {frac:.4%} "
                f"({n_miss}/{n_valid} valid target points) exceeds tolerance "
                f"{_PERSISTENCE_MISS_TOLERANCE:.2%} -- likely a grid-alignment bug, "
                "not the tiny expected gap"
            )
        if stream not in _persistence_miss_warned:
            print(
                f"[warn] EERIE persistence gather: stream {stream!r}: {n_miss}/"
                f"{n_valid} valid target points missed the source grid (cell not "
                "covered by source) at least once -- falling back to the target's "
                "own value there; reported once per worker per stream"
            )
            _persistence_miss_warned.add(stream)
        gathered = np.where(np.isnan(gathered), tar, gathered)
    return gathered


def _process_block_eerie(sample_ids: list[int]) -> BlockResult:
    """EERIE geometry counterpart of `_process_block_fesom`.

    Per (sample, lead): validity mask starts as finite(target) per channel (D3-style
    "the validity mask is the only authority" -- here there is no fixed mesh to derive
    wetness from at all, only the target field itself), then -- when climatology is
    enabled -- is narrowed to `finite(target) & finite(clim_gathered)`: a sampled wet
    point whose gathered climatology cell happens to be NaN (its truth-land status at
    that grid cell disagrees with the climatology reference period's -- not observed
    in practice, see task-25 report, but not assumed impossible) drops out of every
    downstream sum for that point rather than reaching `_clean` with a NaN `clim` at a
    "valid" point (which would raise). Weights = cos(lat) of that lead's own re-drawn
    points; region masks built fresh from those same points (`_region_mask`, shared
    with the FESOM branch, since it only needs per-point latitude). Persistence is
    `stream`'s fstep-0 source, read+denormalized once per sample, gathered at each
    lead's target points via `_eerie_grid.cell_index` (`_gather_persistence`/
    `_gather_at`) -- its OWN miss-rate check uses the target-only mask, unaffected by
    whether climatology is enabled (persistence has nothing to do with climatology).
    Climatology is `_clim_lookup`'s hook (Task 24 designed it, Task 25 wires it up):
    `None` when disabled -> acc all-NaN via `score_block`'s documented `clim=None`
    contract, climatology-model row filled NaN directly since there is no `clim`
    field to score it against at all; a real `(npts, nchan)` gathered field when
    enabled -> every score real, including the climatology model's own rmse/bias
    (its own acc is still NaN, but for a different, purely mathematical reason: `f' =
    pred - clim = clim - clim = 0` identically when "predicting" with climatology
    itself, `acc`'s `0/0`).

    Map accumulators (`_map_leads`) are on the FULL grid (`_eerie_grid.n_cells`),
    scattered via `np.add.at` over each lead's `cell_index` -- per-channel `count`
    int32 (validity differs per channel, e.g. OCEAN_3D's sub-bathymetry NaN)
    alongside the float64 sums (`_MAP_FIELDS_EERIE`, plus `_MAP_FIELDS_EERIE_CLIM`
    when climatology is enabled); `_reduce_blocks_eerie` divides by `count` (NaN
    where `count == 0`) rather than FESOM's fixed sample-count divisor, since a
    full-grid cell is sampled by only a small fraction of a sample's sparse points,
    not every sample.
    """
    grid = _eerie_grid
    n_samples, n_leads = len(sample_ids), len(_leads)
    streams = _cfg["streams"]
    clim_enabled = bool(_cfg["climatology"].get("enabled", False))

    scores = {
        stream: np.empty(
            (len(MODELS), len(_regions), len(METRICS), n_samples, n_leads, len(_channels[stream])),
            dtype=np.float64,
        )
        for stream in streams
    }
    eerie_map_fields = _MAP_FIELDS_EERIE + (_MAP_FIELDS_EERIE_CLIM if clim_enabled else ())
    map_sums = {
        (stream, lead): {
            **{
                field: np.zeros((len(_channels[stream]), grid.n_cells), dtype=np.float64)
                for field in eerie_map_fields
            },
            "count": np.zeros((len(_channels[stream]), grid.n_cells), dtype=np.int32),
        }
        for stream in streams
        for lead in _map_leads
    }

    for i, s in enumerate(sample_ids):
        init_date = _root.store.init_times[s]
        for stream in streams:
            channels_ref = _channels[stream]
            src_by_cell = _gather_persistence(grid, stream, s)

            for j, lead in enumerate(_leads):
                tar_group = _root[f"{s}/{stream}/{lead}/target"]
                pred_group = _root[f"{s}/{stream}/{lead}/prediction"]
                if list(tar_group.attrs["channels"]) != channels_ref:
                    raise ValueError(
                        f"sample {s} stream {stream} lead {lead}: target channels drifted "
                        "from the reference list"
                    )
                if list(pred_group.attrs["channels"]) != channels_ref:
                    raise ValueError(
                        f"sample {s} stream {stream} lead {lead}: prediction channels "
                        "drifted from the reference list"
                    )

                tar = np.asarray(tar_group["data"])
                pred = np.asarray(pred_group["data"])
                if pred.shape[-1] != 1:
                    raise ValueError(
                        f"sample {s} stream {stream} lead {lead}: expected ens==1, "
                        f"got {pred.shape[-1]}"
                    )
                pred = pred[..., 0]

                coords = np.asarray(tar_group["coords"])
                lat_pts = coords[:, 0]
                cell_idx = grid.cell_index(coords)

                tar_mask = np.isfinite(tar).T  # (nchan, npts) -- target-only validity,
                # used unchanged for the persistence gather's own miss-rate check.
                w_row = np.cos(np.radians(lat_pts)).astype(np.float32)[np.newaxis, :]  # (1, npts)

                persistence = _gather_at(src_by_cell, cell_idx, tar, tar_mask, stream)
                c = _clim_lookup(stream, init_date, lead, coords)
                # With climatology enabled, narrow validity to points whose gathered
                # clim is also finite (see this function's docstring); disabled ->
                # unchanged from Task 24 (mask is exactly tar_mask).
                mask = tar_mask if c is None else (tar_mask & np.isfinite(c).T)

                # score_block's weighted sums are NaN-fragile (0 * NaN == NaN even
                # though the point's weight is 0) -- clean copies feed the kernel,
                # raw ones feed the (unmasked, per brief) spatial accumulators below.
                tar_clean = _clean(tar, mask)
                pred_clean = _clean(pred, mask)
                persistence_clean = _clean(persistence, mask)
                c_clean = _clean(c, mask) if c is not None else None

                # One score_block call per (model, region): pred/persistence/clim=None
                # and the shared validity mask, only the weights vary (region_mask *
                # cos-lat) -- mirrors `_process_block_fesom`'s per-(model, region) loop.
                for k, region in enumerate(_regions):
                    region_mask = _region_mask(lat_pts, _cfg["regions"][region])
                    rw = w_row * region_mask[np.newaxis, :]
                    scores[stream][0, k, :, i, j, :] = score_block(
                        pred_clean, tar_clean, c_clean, mask, rw
                    )
                    scores[stream][1, k, :, i, j, :] = score_block(
                        persistence_clean, tar_clean, c_clean, mask, rw
                    )
                    if c_clean is None:
                        # No climatology field to score a "climatology model" against
                        # at all (Task 24: climatology.enabled: false) -- NaN outright,
                        # not just an all-NaN acc row from a score_block call.
                        scores[stream][2, k, :, i, j, :] = np.nan
                    else:
                        scores[stream][2, k, :, i, j, :] = score_block(
                            c_clean, tar_clean, c_clean, mask, rw
                        )

                if lead in _map_leads:
                    sums = map_sums[(stream, lead)]
                    ch_idx, pt_idx = np.nonzero(mask)  # per-channel valid (point) positions
                    cells = cell_idx[pt_idx]

                    err = (pred - tar).T.astype(np.float64)
                    perr = (persistence - tar).T.astype(np.float64)
                    pred_t = pred.T.astype(np.float64)
                    tar_t = tar.T.astype(np.float64)

                    np.add.at(sums["count"], (ch_idx, cells), 1)
                    np.add.at(sums["sum_err"], (ch_idx, cells), err[ch_idx, pt_idx])
                    np.add.at(sums["sum_err2"], (ch_idx, cells), err[ch_idx, pt_idx] ** 2)
                    np.add.at(sums["sum_p"], (ch_idx, cells), pred_t[ch_idx, pt_idx])
                    np.add.at(sums["sum_p2"], (ch_idx, cells), pred_t[ch_idx, pt_idx] ** 2)
                    np.add.at(sums["sum_t"], (ch_idx, cells), tar_t[ch_idx, pt_idx])
                    np.add.at(sums["sum_t2"], (ch_idx, cells), tar_t[ch_idx, pt_idx] ** 2)
                    np.add.at(sums["sum_perr2"], (ch_idx, cells), perr[ch_idx, pt_idx] ** 2)

                    if c is not None:
                        cerr = (c - tar).T.astype(np.float64)
                        f_prime = (pred - c).T.astype(np.float64)
                        o_prime = (tar - c).T.astype(np.float64)
                        np.add.at(sums["sum_cerr2"], (ch_idx, cells), cerr[ch_idx, pt_idx] ** 2)
                        np.add.at(
                            sums["sum_fo"], (ch_idx, cells), (f_prime * o_prime)[ch_idx, pt_idx]
                        )
                        np.add.at(sums["sum_f2"], (ch_idx, cells), f_prime[ch_idx, pt_idx] ** 2)
                        np.add.at(sums["sum_o2"], (ch_idx, cells), o_prime[ch_idx, pt_idx] ** 2)

    return BlockResult(
        sample_ids=list(sample_ids), scores=scores, map_sums=map_sums, n_map_samples=n_samples
    )


def _reduce_blocks_eerie(
    score_results: list[BlockResult], merged_maps: dict | None, cfg: dict
) -> dict[str, xr.Dataset]:
    """EERIE geometry counterpart of `_reduce_blocks_fesom`.

    `maps_{stream}` is on the FULL grid (`ipoint` = `eerie_grid.EerieGrid.n_cells`,
    every (lat, lon) cell of the 0.25 deg grid, not just the cells any sample
    actually sampled) with a `count` (lead, channel, ipoint) int32 variable --
    the per-(channel, cell) accumulated-sample count, used as the divisor in place of
    `_reduce_blocks_fesom`'s fixed `n_total` (a cell no sample's sparse points ever
    landed on has `count == 0` and every other variable NaN there -- plain division
    under `np.errstate(invalid="ignore")` gives `0 / 0 -> NaN` directly, the same
    silent-NaN convention `metrics.score_block` already uses, no explicit `np.where`
    needed). `rmse_climatology`/`acc` are computed the same way `_reduce_blocks_fesom`
    does WHEN a block's `map_sums` carries `_MAP_FIELDS_EERIE_CLIM`'s
    `sum_cerr2`/`sum_fo`/`sum_f2`/`sum_o2` (`climatology.enabled: true`, Task 25);
    otherwise (Task 24's `enabled: false`, unchanged) they are NaN outright since
    there is no climatology accumulator to divide.
    """
    root = open_merged_ranks(cfg["results_dir"], cfg["mini_epoch"])
    leads, map_leads = _compute_leads(cfg)
    regions = list(cfg["regions"])
    if map_leads and merged_maps is None:
        raise ValueError("merged_maps must not be None when map leads are configured")

    grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])
    # Full-grid (lat, lon) per flat cell index (row * n_lon + col, see
    # `eerie_grid.EerieGrid.cell_index`) -- `np.repeat`/`np.tile` reproduce that same
    # row-major flattening as plain 1-D coordinate arrays for `maps_{stream}`'s
    # `ipoint` dim.
    lat_cells = np.repeat(grid.lat, grid.lon.size)
    lon_cells = np.tile(grid.lon, grid.lat.size)

    out = {}
    for stream in cfg["streams"]:
        s0 = root.store.samples[0]
        channels = list(root[f"{s0}/{stream}/1/target"].attrs["channels"])

        sample_ids = [s for r in score_results for s in r.sample_ids]
        # sample is axis 3 of each block's (model, region, metric, sample, lead,
        # channel) scores array -- see process_block/BlockResult.
        score_arr = np.concatenate([r.scores[stream] for r in score_results], axis=3)
        init_times = np.array(
            [root.store.init_times[s] for s in sample_ids], dtype="datetime64[ns]"
        )

        out[f"scores_{stream}"] = xr.Dataset(
            {
                "score": (
                    ("model", "region", "metric", "sample", "lead", "channel"),
                    score_arr.astype(np.float32),
                )
            },
            coords={
                "model": list(MODELS),
                "region": regions,
                "metric": list(METRICS),
                "sample": sample_ids,
                "init_time": ("sample", init_times),
                "lead": leads,
                "channel": channels,
            },
        )

        # See `_reduce_blocks_fesom`'s docstring: no map lead scored -> nothing to
        # build a maps dataset from, so skip it entirely.
        if not map_leads:
            continue

        dims = ("lead", "channel", "ipoint")
        by_var = {
            name: []
            for name in (
                "rmse", "bias", "pred_mean", "pred_var", "target_mean", "target_var",
                "rmse_persistence", "rmse_climatology", "acc",
            )
        }
        counts = []
        for lead in map_leads:
            sums = merged_maps[(stream, lead)]
            count = sums["count"]
            counts.append(count)
            with np.errstate(invalid="ignore", divide="ignore"):
                pred_mean = sums["sum_p"] / count
                target_mean = sums["sum_t"] / count
                by_var["rmse"].append(np.sqrt(sums["sum_err2"] / count))
                by_var["bias"].append(sums["sum_err"] / count)
                by_var["pred_mean"].append(pred_mean)
                by_var["pred_var"].append(np.clip(sums["sum_p2"] / count - pred_mean**2, 0, None))
                by_var["target_mean"].append(target_mean)
                by_var["target_var"].append(
                    np.clip(sums["sum_t2"] / count - target_mean**2, 0, None)
                )
                by_var["rmse_persistence"].append(np.sqrt(sums["sum_perr2"] / count))
                if "sum_cerr2" in sums:
                    by_var["rmse_climatology"].append(np.sqrt(sums["sum_cerr2"] / count))
                    acc = sums["sum_fo"] / np.sqrt(sums["sum_f2"] * sums["sum_o2"])
                    by_var["acc"].append(acc)
                else:
                    nan_field = np.full(count.shape, np.nan, dtype=np.float64)
                    by_var["rmse_climatology"].append(nan_field)
                    by_var["acc"].append(nan_field)

        out[f"maps_{stream}"] = xr.Dataset(
            {
                **{
                    name: (dims, np.stack(arrs).astype(np.float32))
                    for name, arrs in by_var.items()
                },
                "count": (dims, np.stack(counts).astype(np.int32)),
            },
            coords={
                "lead": map_leads,
                "channel": channels,
                "lat": ("ipoint", lat_cells),
                "lon": ("ipoint", lon_cells),
            },
            attrs={
                "n_samples": merged_maps["n_map_samples"],
                "run_id": cfg["run_id"],
            },
        )

    return out
