"""f11: seasonal skill figures, from `scores_{stream}.nc` -- REQUIRES the Task 20
`region` dimension (global/nh/sh); a pre-Task-20 cache `[skip]`s cleanly (see
`_open_region_required`).

Season definitions (paper protocol, `configs/default.yaml`'s `seasons:` section,
binding): a sample's season is the Northern-Hemisphere meteorological convention
of its INITIALISATION month (`DJF`/`MAM`/`JJA`/`SON`); December initialisations
are excluded from every seasonal figure (`seasons.exclude_init_months`). Season
*membership* depends only on `init_time`, never on `region` -- the same set of
samples makes up "DJF" whether the panel shows NH or SH, only the SH panel's
*local* season is reversed (e.g. a DJF-labelled SH sample sits in local summer);
that reversal is a display-label fact (column titles / row labels / footnote),
not a different sample selection.

Aggregation identical to D5 (see `curves.py`): `rmse -> sqrt(mean of squares over
samples)`, `acc -> mean`, now computed over a per-season sample subset instead of
every sample. `curves._draw_scorecard_grid`/`curves._channel_groups` (and
`curves.geometry_streams`, for FESOM/EERIE stream+channel dispatch) are reused
as-is (imported, not duplicated) for the summary heatmap's channel
grouping/cell-drawing -- the same conventions `build_f03` already established.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper_eval.config import fig_dir
from paper_eval.plotting import style
from paper_eval.plotting.curves import (
    _channel_groups,
    _draw_scorecard_grid,
    geometry_streams,
)
from paper_eval.plotting.fesom_maps import get_mesh

#: Fixed season order/palette (brief: "DJF blue, MAM green, JJA orange, SON
#: brown") -- blue/green/orange reuse the validated categorical hues also used
#: elsewhere in this package; brown has no categorical slot, so it is a
#: seasonal-figure-local addition, not a change to `style.py`.
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
SEASON_COLORS = {
    "DJF": "#2a78d6",  # blue
    "MAM": "#008300",  # green
    "JJA": "#eb6834",  # orange
    "SON": "#8c5a2b",  # brown
}
#: All-season reference line: same muted grey as style.MODEL_COLORS' baselines.
_ALL_SEASON_COLOR = "#898781"

_REGIONS = ("nh", "sh")
_COLUMN_TITLES = {
    "nh": "Northern Hemisphere",
    "sh": "Southern Hemisphere (local season reversed)",
}


def _agg_rmse(da: xr.DataArray) -> xr.DataArray:
    return np.sqrt((da**2).mean("sample"))


def _agg_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean("sample")


def _month_to_season(seasons_cfg: dict) -> dict[int, str]:
    """int month (1-12) -> season name, from `seasons_cfg[season]`'s month lists.

    A month listed in `seasons_cfg["exclude_init_months"]` (December, per the
    paper protocol) is left OUT of the map even though it also appears in a
    season's calendar month list (DJF's `[12, 1, 2]`) -- excluded samples get no
    season at all, not a season with December silently included.
    """
    excluded = set(seasons_cfg.get("exclude_init_months", []))
    out: dict[int, str] = {}
    for season in SEASON_ORDER:
        for m in seasons_cfg[season]:
            if m not in excluded:
                out[m] = season
    return out


def assign_seasons(init_time, seasons_cfg: dict) -> np.ndarray:
    """Season label (`"DJF"`/`"MAM"`/`"JJA"`/`"SON"`) per entry of `init_time` (a
    `datetime64` array/DataArray), or `None` where the sample's init month is
    excluded (December) -- calendar month is read via `datetime64[M]`, which is
    exact for any proleptic-Gregorian year numpy supports, no leap-year handling
    needed since only the month component is used.
    """
    months = (np.asarray(init_time).astype("datetime64[M]").astype(int) % 12) + 1
    month_map = _month_to_season(seasons_cfg)
    return np.array([month_map.get(int(m)) for m in months], dtype=object)


def _season_masks(ds: xr.Dataset, seasons_cfg: dict) -> dict[str, np.ndarray]:
    """`{season: bool mask over ds's sample dim}`, from `ds["init_time"]`'s
    calendar month; an excluded-month (December) sample is False in every mask."""
    labels = assign_seasons(ds["init_time"].values, seasons_cfg)
    return {season: (labels == season) for season in SEASON_ORDER}


def _open_region_required(cache_paths: dict, fig_id: str, *keys: str) -> list[xr.Dataset] | None:
    """Open every `keys` cache file, requiring a Task-20 `region` dim on each.

    Unlike `curves._open_required` (which narrows a with-region cache down to
    `region="global"` for figures that predate regions), every f11 figure needs
    the `region` dim itself -- so a cache missing it entirely (old, pre-Task-20
    format) is a `[skip]`, not something to silently work around. Prints one
    `[skip]` message per problem (missing file(s), or missing `region` dim(s))
    and returns None; returns the opened datasets, in `keys` order, otherwise.
    """
    paths = []
    for key in keys:
        p = cache_paths.get(key)
        if p is None or not Path(p).exists():
            print(f"[skip] {fig_id}: missing cache input {key!r}")
            return None
        paths.append(Path(p))
    opened = [xr.open_dataset(p) for p in paths]
    no_region = [key for key, ds in zip(keys, opened) if "region" not in ds.dims]
    if no_region:
        print(
            f"[skip] {fig_id}: {no_region} lack a 'region' dim -- a region-format "
            "cache (Task 20+) is required for seasonal figures"
        )
        return None
    return opened


def _open_region_by_stream(
    cache_paths: dict, fig_id: str, streams: list[str]
) -> dict[str, xr.Dataset] | None:
    """`_open_region_required` for `scores_{stream}.nc` of each of `streams`,
    keyed by stream name -- the region-preserving counterpart of
    `curves._open_by_stream` (which narrows to `region="global"` and must NOT be
    used here: every f11 figure needs the full `region` dim intact)."""
    opened = _open_region_required(cache_paths, fig_id, *[f"scores_{s}" for s in streams])
    if opened is None:
        return None
    return dict(zip(streams, opened, strict=True))


# ── f11 curves: per-channel RMSE/ACC vs. lead, 4 seasons + all-season reference ───────
def _plot_channel_seasonal(
    cfg: dict, ch: str, score_ch: xr.DataArray, leads, masks: dict[str, np.ndarray], all_mask
) -> list[Path]:
    """`score_ch`: `ds["score"]` already selected to `model="emulator",
    channel=ch` (dims `region, metric, sample, lead`). One figure: 2 rows (RMSE,
    ACC) x 2 cols (nh, sh)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True, constrained_layout=True)

    legend_handles: dict[str, Line2D] = {}
    for j, region in enumerate(_REGIONS):
        region_da = score_ch.sel(region=region)
        ax_rmse, ax_acc = axes[0, j], axes[1, j]

        for season in SEASON_ORDER:
            mask = masks[season]
            n = int(mask.sum())
            if n == 0:  # no sample fell in this season (tiny dev sample, or a run whose
                # data covers only part of the year, e.g. all-November) -- skip the line
                # entirely rather than isel-ing with an all-False mask (see the matching
                # empty-season guard + comment in `_build_summary` for why that matters).
                continue
            da = region_da.isel(sample=mask)
            rmse = _agg_rmse(da.sel(metric="rmse"))
            (line,) = ax_rmse.plot(
                leads, rmse.values, color=SEASON_COLORS[season], linestyle="-",
                label=f"{season} (N={n})",
            )
            legend_handles[season] = line
            acc = _agg_mean(da.sel(metric="acc"))
            ax_acc.plot(leads, acc.values, color=SEASON_COLORS[season], linestyle="-")

        n_all = int(all_mask.sum())
        if n_all > 0:
            da_all = region_da.isel(sample=all_mask)
            rmse_all = _agg_rmse(da_all.sel(metric="rmse"))
            (line_all,) = ax_rmse.plot(
                leads, rmse_all.values, color=_ALL_SEASON_COLOR, linestyle="-", linewidth=1.0,
                label=f"All (Dec excluded, N={n_all})",
            )
            legend_handles["all"] = line_all
            acc_all = _agg_mean(da_all.sel(metric="acc"))
            ax_acc.plot(leads, acc_all.values, color=_ALL_SEASON_COLOR, linestyle="-", linewidth=1.0)

        for lead_mark in (15, 30):
            if lead_mark in leads:
                for ax in (ax_rmse, ax_acc):
                    ax.axvline(lead_mark, color="#c3c2b7", linewidth=0.8, linestyle=":", zorder=0)

        ax_rmse.set_title(_COLUMN_TITLES[region])
        ax_rmse.set_ylim(bottom=0)
        ax_rmse.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        style.no_sci_offset(ax_rmse)
        ax_acc.set_xlabel("lead [d]")
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    unit = style.channel_unit(ch)
    axes[0, 0].set_ylabel(f"RMSE [{unit}]" if unit else "RMSE")
    axes[1, 0].set_ylabel("ACC")

    fig.legend(
        legend_handles.values(), [h.get_label() for h in legend_handles.values()],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09),
    )
    fig.suptitle(f"{style.channel_label(ch)} seasonal skill vs. lead time (emulator)")
    return style.savefig(fig, f"f11_{ch}_seasonal", fig_dir(cfg))


def _build_curves(cfg: dict, cache_paths: dict, seasons_cfg: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    opened = _open_region_required(cache_paths, "f11", f"scores_{gs['surface_stream']}")
    if opened is None:
        return []
    (ds,) = opened

    channels = [c for c in seasons_cfg["channels"] if c in ds.channel.values]
    if not channels:
        print(f"[skip] f11: none of seasons.channels present in scores_{gs['surface_stream']}")
        return []

    masks = _season_masks(ds, seasons_cfg)
    all_mask = np.logical_or.reduce([masks[s] for s in SEASON_ORDER])
    leads = ds.lead.values
    score_emulator = ds["score"].sel(model="emulator")

    written: list[Path] = []
    for ch in channels:
        written += _plot_channel_seasonal(
            cfg, ch, score_emulator.sel(channel=ch), leads, masks, all_mask
        )
    return written


# ── f11 summary: skill-score heatmap, 8 rows (season x region) x all channels ─────────
def _build_summary(cfg: dict, cache_paths: dict, seasons_cfg: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    geometry = cfg.get("geometry", "fesom")
    streams = sorted({gs["surface_stream"]} | {s for _, _, s in gs["depth_groups"]})
    ds_by_stream = _open_region_by_stream(cache_paths, "f11", streams)
    if ds_by_stream is None:
        return []
    mesh = get_mesh(cfg["mesh_path"]) if geometry == "fesom" else None
    nz1 = mesh.nz1 if mesh is not None else None

    # Defensive lead filtering, same spirit as build_f03: a configured map_lead
    # absent from any needed scores cache would otherwise raise out of .sel() below.
    map_leads_cfg = list(cfg["leads"]["map_leads"])
    avail = set.intersection(*(set(ds.lead.values.tolist()) for ds in ds_by_stream.values()))
    summary_leads = [lead for lead in map_leads_cfg if lead in avail]
    missing = [lead for lead in map_leads_cfg if lead not in summary_leads]
    if missing:
        print(f"[skip] f11: lead(s) {missing} not present in every scores cache -- dropping")
    if not summary_leads:
        print("[skip] f11: none of leads.map_leads present in every needed scores cache")
        return []

    masks_by_stream = {
        stream: _season_masks(ds, seasons_cfg) for stream, ds in ds_by_stream.items()
    }
    # Same 5 groups, same order, as build_f03's scorecard -- "ALL channels
    # (grouped/ordered as the f03 scorecard)" per the brief.
    groups = [
        (gname, chans, ds, masks_by_stream[stream])
        for gname, chans, ds, stream in _channel_groups(gs, ds_by_stream)
    ]

    # 8 rows, season outer / region inner: DJF-NH, DJF-SH, MAM-NH, ..., SON-SH.
    # "NH"/"SH" (display abbreviations, not the raw lowercase `region` coord
    # values) per the brief's own row-order example and the "no internal
    # identifiers" style rule.
    row_labels = [f"{season}-{region.upper()}" for season in SEASON_ORDER for region in _REGIONS]

    written: list[Path] = []
    for lead in summary_leads:
        col_labels: list[str] = []
        group_boundaries: list[float] = []
        data_cols: list[list[float]] = []
        for gname, chans, ds, masks in groups:
            if not chans:
                continue
            if data_cols:
                group_boundaries.append(len(data_cols) - 0.5)
            score = ds["score"]
            for ch in chans:
                col = []
                for season in SEASON_ORDER:
                    mask = masks[season]
                    n = int(mask.sum())
                    for region in _REGIONS:
                        if n == 0:
                            # No sample falls in this season (e.g. a run whose
                            # results only cover part of the year, all one
                            # season) -- render a blank (NaN) cell instead of
                            # calling .isel with an all-False mask. On a
                            # not-yet-loaded (lazily backed) score array, an
                            # empty boolean .isel followed by arithmetic can
                            # raise inside xarray ("dimensions ('sample',) must
                            # have the same length as the number of data
                            # dimensions, ndim=6"): `.dims`/`.shape` correctly
                            # report a 1-D, 0-length result, but the backend's
                            # lazily materialized `.values` retains all 6
                            # original axes (collapsed to size-1, not dropped)
                            # -- confirmed by a minimal repro against this same
                            # cache. Skipping the empty case entirely avoids
                            # that backend/lazy-indexing edge case AND is the
                            # right rendering regardless: a season with zero
                            # samples has no skill score to show.
                            col.append(np.nan)
                            continue
                        sel_e = score.sel(
                            model="emulator", metric="rmse", lead=lead, channel=ch, region=region
                        ).isel(sample=mask)
                        sel_p = score.sel(
                            model="persistence", metric="rmse", lead=lead, channel=ch, region=region
                        ).isel(sample=mask)
                        rmse_e = float(_agg_rmse(sel_e))
                        rmse_p = float(_agg_rmse(sel_p))
                        with np.errstate(invalid="ignore", divide="ignore"):
                            skill = 1.0 - rmse_e / rmse_p
                        col.append(skill if np.isfinite(skill) else np.nan)
                data_cols.append(col)
                col_labels.append(style.channel_label(ch, nz1))

        data = np.array(data_cols, dtype=np.float64).T  # (8 rows, n_cols)
        n_cols = data.shape[1]

        vmin, vmax = style.symmetric_limits(data, pct=99.0)
        vmin = -vmax
        cmap = plt.get_cmap(style.DIVERGING_CMAP).copy()
        cmap.set_bad("white")

        fig_w = max(10.0, 0.16 * n_cols + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 7.0))
        im = _draw_scorecard_grid(
            ax, data, row_labels, col_labels, group_boundaries, vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(
            "Seasonal skill score  1 - RMSE_emulator / RMSE_persistence  by season x "
            f"hemisphere  --  lead {lead} d  (positive = emulator better)",
            fontsize=9,
        )
        # Reversal note as a footnote (brief: "in the y-label or a footnote line
        # under the title") rather than repeated on every SH row label.
        fig.text(
            0.5, -0.02,
            "SH rows: local season reversed (e.g. DJF = local summer in the Southern Hemisphere)",
            ha="center", fontsize=7, style="italic",
        )
        # Separator between each season's NH/SH row pair (mirrors build_f03's
        # separator between its persistence-baseline and climatology-baseline rows).
        for r in range(2, len(row_labels), 2):
            ax.axhline(r - 0.5, color="white", linewidth=1.0)

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        cbar.set_label("skill score")
        fig.tight_layout()

        written += style.savefig(fig, f"f11_summary_skill_{lead}d", fig_dir(cfg))

    return written


def build_f11(cfg: dict, cache_paths: dict) -> list[Path]:
    seasons_cfg = cfg.get("seasons")
    if not seasons_cfg:
        print("[skip] f11: config has no 'seasons' section")
        return []
    return _build_curves(cfg, cache_paths, seasons_cfg) + _build_summary(cfg, cache_paths, seasons_cfg)
