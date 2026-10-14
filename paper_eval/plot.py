"""CLI: build every paper figure from the stage-1/spectra caches.

    .venv/bin/python -m paper_eval.plot --config paper_eval/configs/default.yaml \\
        [--only f01,f04] [--cache-dir PATH]

Every figure builder reads ONLY the cached `scores_{stream}.nc`, `maps_{stream}.nc`,
`snapshots_{stream}.nc` and `spectra.nc` files -- never the raw zarr archives. A
figure whose required cache file(s) are missing prints a `[skip]` warning and
writes nothing -- that is not an error, so `--cache-dir` pointed at an empty (or
not-yet-populated) directory produces a clean, all-skipped run (exit 0). A figure
builder that instead *raises* (a genuine bug, not a missing-input skip) is caught
so it can't sink the rest of the run, but is tracked separately and turns the
final exit code non-zero -- an actual bug must not be silently reported as
success.

`--cache-dir` overrides the cache location; production runs omit it and use
`<output.cache_dir>/<run_id>` (`paper_eval.config.cache_dir`). Figures always go to
`<output.fig_dir>/<run_id>` (`paper_eval.config.fig_dir`), regardless of the cache
location.
"""

import argparse
from pathlib import Path

from paper_eval.config import cache_dir, load_config
from paper_eval.plotting import (
    curves,
    depth,
    eerie_maps,
    fesom_maps,
    seasonal,
    spectra_figs,
    variability,
)


def _by_geometry(fesom_fn, eerie_fn):
    """`callable(cfg, cache_paths)` dispatching on `cfg.get("geometry", "fesom")` --
    the map-figure families (f04/f05/f08/f10) have a wholly separate builder per
    geometry (`fesom_maps.py` vs. `eerie_maps.py`, see the latter's module
    docstring for why); this is the one place that picks between them, so neither
    builder needs to know the other exists.
    """
    def build(cfg: dict, cache_paths: dict) -> list[Path]:
        if cfg.get("geometry", "fesom") == "eerie":
            return eerie_fn(cfg, cache_paths)
        return fesom_fn(cfg, cache_paths)
    return build


def _build_f09(cfg: dict, cache_paths: dict) -> list[Path]:
    """f09 (native-mesh wavelength spectra) does not apply to the EERIE gridded
    geometry at all -- `compute_spectra.py` is never even run against an EERIE
    config (see `configs/dbuzcpgf.yaml`'s `spectra:` section), so `spectra.nc`
    never exists for one; skip with the specific reason up front rather than
    falling through to `spectra_figs.build_f09`'s generic "missing cache input"
    message.
    """
    if cfg.get("geometry", "fesom") == "eerie":
        print("[skip] f09: spectra disabled for sparse-sampled gridded runs")
        return []
    return spectra_figs.build_f09(cfg, cache_paths)


#: f01..f11, each `callable(cfg, cache_paths) -> list[Path]` of written figure files.
FIGURES = {
    "f01": curves.build_f01,
    "f02": curves.build_f02,
    "f03": curves.build_f03,
    "f04": _by_geometry(fesom_maps.build_f04, eerie_maps.build_f04),
    "f05": _by_geometry(fesom_maps.build_f05, eerie_maps.build_f05),
    "f06": depth.build_f06,
    "f07": depth.build_f07,
    "f08": _by_geometry(variability.build_f08, eerie_maps.build_f08),
    "f09": _build_f09,
    "f10": _by_geometry(fesom_maps.build_f10, eerie_maps.build_f10),
    "f12": variability.build_f12,   # FESOM only; skips cleanly elsewhere
    "f11": seasonal.build_f11,
}


def _build_cache_paths(cache_root: Path, cfg: dict) -> dict[str, Path]:
    """`{scores,maps,snapshots}_{stream}.nc` for every configured stream, plus
    `spectra.nc` -- built from `cfg["streams"]` rather than a hardcoded FESOM_NODE/
    FESOM_ELEM list, so it resolves correctly for any stream set (FESOM's
    `[FESOM_NODE, FESOM_ELEM]` or EERIE's `[OCEAN_2D, OCEAN_3D]`) without this
    module needing to know which geometry it's building for.
    """
    paths = {"spectra": cache_root / "spectra.nc"}
    for stream in cfg["streams"]:
        for kind in ("scores", "maps", "snapshots"):
            paths[f"{kind}_{stream}"] = cache_root / f"{kind}_{stream}.nc"
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to a paper_eval YAML config.")
    parser.add_argument("--only", default=None, help="Comma-separated figure ids, e.g. f01,f04.")
    parser.add_argument("--cache-dir", default=None, help="Override the stage-1 cache directory.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # cache_dir(cfg) has the side effect of creating <cache_dir>/<run_id>; only take
    # that (production) path when no override is given, so pointing --cache-dir at a
    # dev/quicktest or an empty verification directory never touches the real cache tree.
    cache_root = Path(args.cache_dir) if args.cache_dir else cache_dir(cfg)
    cache_paths = _build_cache_paths(cache_root, cfg)

    if args.only:
        requested = [f.strip() for f in args.only.split(",") if f.strip()]
        unknown = [f for f in requested if f not in FIGURES]
        if unknown:
            raise SystemExit(f"unknown figure id(s): {unknown}; known: {sorted(FIGURES)}")
    else:
        requested = sorted(FIGURES)

    print(f"cache dir: {cache_root}")
    print(f"figures:   {requested}")

    written: dict[str, list[Path]] = {}
    errored: list[str] = []
    for fig_id in requested:
        builder = FIGURES[fig_id]
        try:
            written[fig_id] = builder(cfg, cache_paths)
        except Exception as exc:  # a bug in one figure must not sink the whole run...
            print(f"[error] {fig_id}: {type(exc).__name__}: {exc}")
            written[fig_id] = []
            errored.append(fig_id)  # ...but it must not be reported as a clean run either

    print("\n== summary ==")
    total = 0
    for fig_id in requested:
        if fig_id in errored:
            print(f"  {fig_id}: ERRORED (see [error] message above)")
            continue
        paths = written[fig_id]
        if not paths:
            print(f"  {fig_id}: (nothing written -- skipped, see [skip] message(s) above)")
            continue
        for p in paths:
            size = p.stat().st_size if p.exists() else 0
            print(f"  {fig_id}: {p}  ({size:,} bytes)")
            total += 1
    print(f"total files written: {total}")

    if errored:
        print(f"\n{len(errored)} figure(s) errored: {errored}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
