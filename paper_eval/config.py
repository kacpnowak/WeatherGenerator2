"""Config loading and small path helpers for the paper evaluation pipeline.

Kept deliberately minimal: a plain dict loaded from YAML, no dataclass schema,
no validation framework. Later CLIs (compute.py, plot.py, ...) do their own
argparse and use these helpers.

Also holds the lookup of the evaluated run's per-stream normalization overrides
(stream option `normalization_stats`), which `scores._load_norm_stats` and
`selfcheck._norm_stats` apply on top of the truth archive's statistics.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import yaml

#: Repository root (`paper_eval/..`): `models/<run_id>/` lives here, and the stream
#: option `normalization_stats` is a path relative to it (the training/inference
#: reader opens it relative to the working directory, which is the repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict.

    Raises FileNotFoundError with a clear message if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_int_list(spec: str | None) -> list[int] | None:
    """Parse a CLI-style spec into a list of ints.

    None -> None
    "1,15,30" -> [1, 15, 30]
    "0:5" -> [0, 1, 2, 3, 4]  (colon = half-open python range)
    """
    if spec is None:
        return None
    if ":" in spec:
        start, end = spec.split(":")
        return list(range(int(start), int(end)))
    return [int(x) for x in spec.split(",")]


def parse_samples(spec: str | None) -> list[int] | None:
    """Alias of parse_int_list (kept for CLI readability)."""
    return parse_int_list(spec)


def cache_dir(cfg: dict) -> Path:
    """<output.cache_dir>/<run_id>, created if missing."""
    path = Path(cfg["output"]["cache_dir"]) / cfg["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def clim_dir(cfg: dict) -> Path:
    """<output.cache_dir>/climatology, created if missing."""
    path = Path(cfg["output"]["cache_dir"]) / "climatology"
    path.mkdir(parents=True, exist_ok=True)
    return path


def regrid_dir(cfg: dict) -> Path:
    """<output.cache_dir>/regrid, created if missing."""
    path = Path(cfg["output"]["cache_dir"]) / "regrid"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fig_dir(cfg: dict) -> Path:
    """<output.fig_dir>/<run_id>, created if missing."""
    path = Path(cfg["output"]["fig_dir"]) / cfg["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


#: Mesh entity of the legacy streams (one stream per entity, every variable in it).
_LEGACY_STREAM_ENTITY = {"FESOM_NODE": "node", "FESOM_ELEM": "elem"}


def stream_layout(cfg: dict, stream: str) -> dict | None:
    """`cfg["stream_layout"][stream]`, or None for a stream without an entry.

    Per-variable streams (TEMP, SALT, SURFACE, U, V -- one stream per variable instead
    of one per mesh entity) need an entry `{entity: node|elem, clim: [source, ...]}`:
    `entity` picks the mesh points (wet masks, area weights) and `clim` the climatology
    zarrs (`clim_<source>_<ref_start>-<ref_end>.zarr`) their channels are read from by
    name. Legacy configs (FESOM_NODE/FESOM_ELEM) have no `stream_layout` and are
    handled exactly as before: entity from the stream name, one climatology zarr per
    stream whose channel list must match exactly.
    """
    return (cfg.get("stream_layout") or {}).get(stream)


def stream_entity(cfg: dict, stream: str) -> str:
    """Mesh entity ("node" or "elem") of `stream`: `stream_layout.<stream>.entity`, or
    the legacy FESOM_NODE/FESOM_ELEM mapping. ValueError for anything else."""
    layout = stream_layout(cfg, stream)
    entity = layout.get("entity") if layout is not None else _LEGACY_STREAM_ENTITY.get(stream)
    if entity not in ("node", "elem"):
        raise ValueError(
            f"stream {stream!r}: unknown mesh entity {entity!r} -- per-variable streams "
            "need `stream_layout.<stream>.entity: node|elem` in the paper_eval config"
        )
    return entity


def clim_paths(cfg: dict, stream: str) -> list[Path]:
    """Climatology zarr(s) for `stream`: one per `stream_layout.<stream>.clim` source, or
    the single legacy `clim_<stream>_<ref>.zarr`."""
    y0, y1 = cfg["climatology"]["ref_start"], cfg["climatology"]["ref_end"]
    layout = stream_layout(cfg, stream)
    sources = list(layout["clim"]) if layout is not None else [stream]
    return [clim_dir(cfg) / f"clim_{src}_{y0}-{y1}.zarr" for src in sources]


def model_config_path(cfg: dict) -> Path:
    """Path of the evaluated run's model config JSON, as the framework writes it:
    `models/<run_id>/model_<run_id>_chkpt<mini_epoch:05d>.json` (`_latest` for
    mini_epoch -1, no suffix for None; the suffixless file is the fallback, as in
    `weathergen.common.config.load_run_config`). For an inference run this file is
    the (merged) config that run's reader was built from, so its `streams` are the
    ones whose statistics normalized the stored `source`.

    `cfg["models_dir"]` (optional) overrides `<repo root>/models` -- test hook.
    """
    run_id = cfg["run_id"]
    mini_epoch = cfg.get("mini_epoch")
    if mini_epoch is None:
        suffix = ""
    elif mini_epoch == -1:
        suffix = "_latest"
    else:
        suffix = f"_chkpt{int(mini_epoch):05d}"
    run_dir = Path(cfg.get("models_dir", REPO_ROOT / "models")) / run_id
    path = run_dir / f"model_{run_id}{suffix}.json"
    if not path.exists() and (run_dir / f"model_{run_id}.json").exists():
        path = run_dir / f"model_{run_id}.json"
    return path


def normalization_overrides(cfg: dict, stream: str) -> dict | None:
    """The evaluated run's `normalization_stats` override for `stream`, or None.

    A run trained/inferred with wet-point statistics sets the stream option
    `normalization_stats: <json path, relative to the repo root>` (JSON format
    `{channel: {"mean": m, "std": s}, ..., "_meta": {...}}`); its reader then
    normalizes with those statistics instead of the truth archive's. Returns the
    parsed JSON dict, or None when the run's model config has no such option for
    `stream`, so callers keep the archive statistics unchanged.

    If the run has no model config JSON at all, a warning is emitted and None is
    returned (archive statistics, i.e. the behaviour before this option existed).
    A configured JSON path that does not exist raises FileNotFoundError.
    """
    config_path = model_config_path(cfg)
    if not config_path.exists():
        warnings.warn(
            f"no model config for run {cfg['run_id']!r} at {config_path}: cannot tell "
            f"whether stream {stream!r} used 'normalization_stats'; assuming the truth "
            "archive statistics",
            stacklevel=2,
        )
        return None
    with config_path.open() as f:
        stream_cfg = (json.load(f).get("streams") or {}).get(stream) or {}
    stats_path = stream_cfg.get("normalization_stats")
    if not stats_path:
        return None
    full_path = Path(cfg.get("repo_root", REPO_ROOT)) / stats_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"stream {stream!r} of run {cfg['run_id']!r} sets normalization_stats="
            f"{stats_path!r} ({config_path}), but {full_path} does not exist"
        )
    with full_path.open() as f:
        return json.load(f)


def apply_normalization_overrides(
    colnames: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    overrides: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the (mean, std) of the channels named in `overrides`.

    Same semantics as `weathergen.readers_extra.data_reader_fesom.
    apply_normalization_overrides` (replicated so paper_eval needs no weathergen
    import; `test_norm_stats.py` checks the two agree): `colnames` names the entries
    of `mean`/`std`; keys starting with "_" (metadata) and channels not in
    `colnames` are ignored; an applied override needs a finite mean and a finite,
    positive std (ValueError otherwise). Returns float64 copies; inputs unchanged.
    """
    new_mean = np.array(mean, dtype=np.float64)
    new_std = np.array(std, dtype=np.float64)
    if not len(colnames) == len(new_mean) == len(new_std):
        raise ValueError(
            f"{len(colnames)} column names but {len(new_mean)} means and {len(new_std)} stds"
        )
    column = {name: i for i, name in enumerate(colnames)}
    for name, stats in overrides.items():
        if name.startswith("_") or name not in column:
            continue
        ch_mean, ch_std = float(stats["mean"]), float(stats["std"])
        if not (np.isfinite(ch_std) and ch_std > 0.0):
            raise ValueError(f"Normalisation override for {name!r} needs std > 0, got {ch_std}")
        if not np.isfinite(ch_mean):
            raise ValueError(f"Normalisation override for {name!r} has a non-finite mean")
        new_mean[column[name]] = ch_mean
        new_std[column[name]] = ch_std
    return new_mean, new_std
