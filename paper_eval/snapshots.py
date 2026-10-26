"""Showcase snapshot extractor: full spatial fields (prediction/target/source) for the
single, human-picked forecast init used in the paper's showcase map figures.

Two geometries (`cfg.get("geometry", "fesom")`, matching `scores.py`'s dispatch): the
sample-selection logic (nearest available init to `cfg["showcase"]["init_time"]`) is
shared, geometry-agnostic; per-stream field extraction is not (`_extract_stream_fesom`
vs. `_extract_stream_eerie`) since EERIE's sparse target/prediction points are re-drawn
every forecast step -- there is no single shared `ipoint` coordinate across leads the
way FESOM's fixed mesh gives one for free.
"""

import numpy as np
import xarray as xr

from paper_eval import eerie_grid
from paper_eval.meshtools import PointIndex
from paper_eval.scores import _load_norm_stats, full_point_coords

#: Above this gap between the requested `showcase.init_time` and the nearest
#: available init, `extract` is silently substituting a very different sample --
#: worth a loud warning rather than a quiet nearest-neighbour pick (see F5: a
#: config left pointing at the old run's period after `run_id`/`results_dir` are
#: updated would otherwise snap to whatever init happens to be nearest, with no
#: indication the requested date was unavailable).
_INIT_MISMATCH_WARN_DAYS = 15


def extract(cfg: dict, root) -> dict[str, xr.Dataset]:
    """Nearest-init-time showcase snapshot, per configured stream.

    Picks the sample in `root.store.samples` whose init_time is closest to
    `cfg["showcase"]["init_time"]`, then extracts each stream's fields
    (`_extract_stream_fesom`/`_extract_stream_eerie`, by `cfg.get("geometry",
    "fesom")`).
    """
    samples = root.store.samples
    target_dt = np.datetime64(cfg["showcase"]["init_time"])
    init_times = np.array([root.store.init_times[s] for s in samples])
    idx = int(np.argmin(np.abs(init_times - target_dt)))
    sample, init_time = samples[idx], init_times[idx]

    gap_days = abs(init_time - target_dt) / np.timedelta64(1, "D")
    if gap_days > _INIT_MISMATCH_WARN_DAYS:
        print(
            f"[warn] snapshots.extract: nearest available init {init_time} is "
            f"{gap_days:.1f} days from requested showcase.init_time "
            f"{cfg['showcase']['init_time']!r} -- check init_filter/showcase.init_time "
            "against the run actually being evaluated"
        )

    geometry = cfg.get("geometry", "fesom")
    out = {}
    for stream in cfg["streams"]:
        if geometry == "eerie":
            ds = _extract_stream_eerie(cfg, root, stream, sample, init_time)
        else:
            ds = _extract_stream_fesom(cfg, root, stream, sample, init_time)
        out[f"snapshots_{stream}"] = ds

    return out


def _extract_stream_fesom(cfg: dict, root, stream: str, sample: int, init_time) -> xr.Dataset:
    """FESOM native-mesh showcase snapshot for one stream: fixed mesh points shared
    by `source`/`target`/`prediction` and every lead, so all three stack into one
    `ipoint` coordinate (unlike `_extract_stream_eerie`)."""
    avail_fsteps = {int(f) for f, _ in root[f"{sample}/{stream}"].groups()}
    leads = [lead for lead in cfg["showcase"]["leads"] if lead in avail_fsteps]

    # Reference channel list/order: target at fstep 1 (always present regardless
    # of which showcase leads are configured), also used to align `source`
    # (fstep 0) -- which can carry extra channels the model doesn't predict, e.g.
    # FESOM_ELEM's source has one more column ('v_210') than target/prediction.
    channels = list(root[f"{sample}/{stream}/1/target"].attrs["channels"])

    # `source` is the model's z-score normalized input, not a physical field --
    # denormalize with the training reader's own per-channel mean/std (same stats
    # path as scores.py's persistence baseline) before it feeds initial-condition
    # map panels; target/prediction are already physical and untouched.
    src_group = root[f"{sample}/{stream}/0/source"]
    src_channels = list(src_group.attrs["channels"])
    src_data = np.asarray(src_group["data"])
    src_idx = [src_channels.index(c) for c in channels] if src_channels != channels else None
    if src_idx is not None:
        src_data = src_data[:, src_idx]

    mean_full, std_full = _load_norm_stats(cfg, stream, src_channels)
    mean_vec, std_vec = (
        (mean_full[src_idx], std_full[src_idx]) if src_idx is not None else (mean_full, std_full)
    )
    src_phys = src_data * std_vec + mean_vec

    source = src_phys.T.astype(np.float32)  # (channel, ipoint)

    # full point set: a target subsampled with `max_num_targets` (different points every
    # lead) is scattered onto it, NaN where not sampled
    pts = full_point_coords(root, sample, stream)
    points = PointIndex(pts)
    lat, lon = pts[:, 0], pts[:, 1]

    pred_list, tar_list = [], []
    for lead in leads:
        tar_group = root[f"{sample}/{stream}/{lead}/target"]
        pred_group = root[f"{sample}/{stream}/{lead}/prediction"]

        tar = np.asarray(tar_group["data"])
        pred = np.asarray(pred_group["data"])
        assert pred.shape[-1] == 1, f"expected ens==1, got {pred.shape[-1]}"
        pred = pred[..., 0]
        tar, _ = points.densify(tar, tar_group["coords"])
        pred, _ = points.densify(pred, pred_group["coords"])

        tar_list.append(tar.T)
        pred_list.append(pred.T)

    return xr.Dataset(
        {
            "prediction": (
                ("lead", "channel", "ipoint"),
                np.stack(pred_list).astype(np.float32),
            ),
            "target": (("lead", "channel", "ipoint"), np.stack(tar_list).astype(np.float32)),
            "source": (("channel", "ipoint"), source),
        },
        coords={
            "lead": leads,
            "channel": channels,
            "lat": ("ipoint", lat),
            "lon": ("ipoint", lon),
        },
        attrs={
            "sample": int(sample),
            "init_time": str(init_time),
            "run_id": cfg["run_id"],
        },
    )


def _extract_stream_eerie(cfg: dict, root, stream: str, sample: int, init_time) -> xr.Dataset:
    """EERIE gridded showcase snapshot for one stream.

    `source` is fstep-0's full wet-cell grid, denormalized, AS-IS (its own
    `ipoint_source`/`source_lat`/`source_lon`, fixed for a given stream in practice
    but read fresh here, not assumed). `target`/`prediction` are stored AS-IS too,
    but -- unlike FESOM's fixed mesh -- their points are re-drawn every forecast
    step, so there is no single shared point coordinate to stack leads against:
    each lead's own target/prediction/coords are stored under lead-suffixed
    variable names (`target_<lead>`/`prediction_<lead>`/`lat_<lead>`/`lon_<lead>`),
    sharing only the generic-index `ipoint_sparse` dim (same size every lead, no
    coordinate values of its own -- see the dataset's `note` attr).
    """
    avail_fsteps = {int(f) for f, _ in root[f"{sample}/{stream}"].groups()}
    leads = [lead for lead in cfg["showcase"]["leads"] if lead in avail_fsteps]

    channels = list(root[f"{sample}/{stream}/1/target"].attrs["channels"])

    src_group = root[f"{sample}/{stream}/0/source"]
    src_channels = list(src_group.attrs["channels"])
    if src_channels != channels:
        raise ValueError(
            f"stream {stream!r}: source channels {src_channels} != target/prediction "
            f"reference {channels} -- EERIE geometry assumes identical source/target "
            "channel lists"
        )
    src_data = np.asarray(src_group["data"])
    src_coords = np.asarray(src_group["coords"])

    mean_vec, std_vec = eerie_grid.load_norm_stats(cfg["truth"][stream], channels)
    source = (src_data * std_vec + mean_vec).T.astype(np.float32)  # (channel, ipoint_source)

    data_vars = {"source": (("channel", "ipoint_source"), source)}
    coords = {
        "channel": channels,
        "source_lat": ("ipoint_source", src_coords[:, 0]),
        "source_lon": ("ipoint_source", src_coords[:, 1]),
    }
    for lead in leads:
        tar_group = root[f"{sample}/{stream}/{lead}/target"]
        pred_group = root[f"{sample}/{stream}/{lead}/prediction"]
        if list(tar_group.attrs["channels"]) != channels:
            raise ValueError(f"stream {stream!r} lead {lead}: target channels drifted")
        if list(pred_group.attrs["channels"]) != channels:
            raise ValueError(f"stream {stream!r} lead {lead}: prediction channels drifted")

        tar = np.asarray(tar_group["data"])
        pred = np.asarray(pred_group["data"])
        assert pred.shape[-1] == 1, f"expected ens==1, got {pred.shape[-1]}"
        pred = pred[..., 0]
        pts = np.asarray(tar_group["coords"])

        data_vars[f"target_{lead}"] = (("channel", "ipoint_sparse"), tar.T.astype(np.float32))
        data_vars[f"prediction_{lead}"] = (("channel", "ipoint_sparse"), pred.T.astype(np.float32))
        coords[f"lat_{lead}"] = ("ipoint_sparse", pts[:, 0])
        coords[f"lon_{lead}"] = ("ipoint_sparse", pts[:, 1])

    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "sample": int(sample),
            "init_time": str(init_time),
            "run_id": cfg["run_id"],
            "geometry": "eerie",
            "leads": list(leads),
            "note": (
                "EERIE geometry: target/prediction points are re-drawn every forecast "
                "step, so there is no single shared 'ipoint' coordinate across leads "
                "(unlike the FESOM snapshot layout) -- each lead's own "
                "target/prediction/coords are stored under lead-suffixed variable names "
                "(target_<lead>/prediction_<lead>/lat_<lead>/lon_<lead>), sharing only "
                "the generic-index 'ipoint_sparse' dim (no coordinate values of its "
                "own). 'source' is the fstep-0 full wet-cell grid, denormalized, with "
                "its own 'ipoint_source'/'source_lat'/'source_lon'."
            ),
        },
    )
