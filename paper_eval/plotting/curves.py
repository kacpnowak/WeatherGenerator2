"""f01, f02, f03: score curves and the scorecard heatmap, from `scores_{stream}.nc`.

Aggregation (plan D5, applied here at plot time): RMSE over samples is
`sqrt((rmse**2).mean("sample"))`; bias and ACC are a plain `.mean("sample")`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper_eval.config import fig_dir
from paper_eval.meshtools import parse_channel
from paper_eval.plotting import style
from paper_eval.plotting.fesom_maps import get_mesh

#: Per-geometry stream/channel-group table for the score-based figures (f01, f02,
#: f03, f06, f07, f11): the same figure-building code walks this table regardless
#: of geometry -- only this table (plus the channel-label/-unit lookups in
#: style.py) differs between FESOM's native mesh and EERIE's gridded streams, so a
#: new geometry only needs a new entry here, never a forked per-figure branch.
#: `depth_axis_prefix` names the `depth_groups` entry whose own channel set is a
#: superset of every other depth group's (FESOM: "temp" -- salt/u share its 20
#: levels, v is missing one; EERIE: "avg_uoe" -- thetao/so's 3 levels plus "surf"),
#: used by f06's shared lead-depth axis and f02's EERIE curve-depth selection (see
#: their docstrings).
GEOMETRY_STREAMS = {
    "fesom": {
        "surface_stream": "FESOM_NODE",
        "surface_channels": ("sst", "sss", "ssh"),
        "depth_groups": [
            ("Temperature", "temp", "FESOM_NODE"),
            ("Salinity", "salt", "FESOM_NODE"),
            ("u velocity", "u", "FESOM_ELEM"),
            ("v velocity", "v", "FESOM_ELEM"),
        ],
        "depth_axis_prefix": "temp",
    },
    "eerie": {
        "surface_stream": "OCEAN_2D",
        "surface_channels": ("avg_tos", "avg_sos", "avg_zos"),
        "depth_groups": [
            ("Temperature", "avg_thetao", "OCEAN_3D"),
            ("Salinity", "avg_so", "OCEAN_3D"),
            ("u velocity", "avg_uoe", "OCEAN_3D"),
            ("v velocity", "avg_von", "OCEAN_3D"),
        ],
        "depth_axis_prefix": "avg_uoe",
    },
}


def geometry_streams(cfg: dict) -> dict:
    """`GEOMETRY_STREAMS[cfg.get("geometry", "fesom")]` -- the one place every
    score-based figure resolves which streams/channel names it needs."""
    return GEOMETRY_STREAMS[cfg.get("geometry", "fesom")]


def _agg_rmse(da: xr.DataArray) -> xr.DataArray:
    return np.sqrt((da**2).mean("sample"))


def _agg_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean("sample")


def _open_by_stream(
    cache_paths: dict, fig_id: str, kind: str, streams: list[str]
) -> dict[str, xr.Dataset] | None:
    """Open `{kind}_{stream}.nc` (e.g. `"scores"`/`"maps"`) for each of `streams`,
    keyed by stream name -- shared by every figure that needs more than one
    stream's cache file (f02/f03/f06/f07/f11), so the open+dedup+zip pattern lives
    in one place rather than being re-derived per figure. `[skip]`s (returns
    `None`, message already printed by `_open_required`) if any is missing.
    """
    opened = _open_required(cache_paths, fig_id, *[f"{kind}_{s}" for s in streams])
    if opened is None:
        return None
    return dict(zip(streams, opened, strict=True))


_ChannelGroup = tuple[str, list[str], xr.Dataset, str]


def _channel_groups(gs: dict, ds_by_stream: dict[str, xr.Dataset]) -> list[_ChannelGroup]:
    """`[(group_name, channel_names, ds, stream), ...]` -- "Surface" plus each of
    `gs["depth_groups"]`, channel lists filtered to what the corresponding
    dataset actually has and depth-sorted. Shared by `build_f03`'s scorecard
    columns and `seasonal.py`'s f11 summary heatmap columns -- both group/order
    channels identically (Surface | Temperature | Salinity | u velocity |
    v velocity), for either geometry.
    """
    surface_ds = ds_by_stream[gs["surface_stream"]]
    groups = [
        (
            "Surface",
            [c for c in gs["surface_channels"] if c in surface_ds.channel.values],
            surface_ds,
            gs["surface_stream"],
        )
    ]
    for gname, prefix, stream in gs["depth_groups"]:
        ds = ds_by_stream[stream]
        chans = [c for _, c in _sorted_depth_channels(ds.channel.values, prefix)]
        groups.append((gname, chans, ds, stream))
    return groups


def _depth_suffix_value(suffix: str) -> float | None:
    """Parse a channel depth-suffix string into a sortable/display depth in
    meters, or `None` if unparseable. FESOM: `"107"` -> `107.0` (an approximate
    integer label; `depth_value` resolves the exact depth via
    `meshtools.parse_channel`+`nz1` for real display use). EERIE: the suffix
    already IS the depth in meters, exactly -- `"2.5m"` -> `2.5`, `"97.5m"` ->
    `97.5`, `"surf"` -> `0.0` (the shallowest level). One parser for both naming
    schemes, so a new scheme only needs a new case here, never a forked
    per-figure loop.
    """
    if suffix == "surf":
        return 0.0
    text = suffix[:-1] if suffix.endswith("m") else suffix
    try:
        return float(text)
    except ValueError:
        return None


def depth_suffixes(channels, prefix: str) -> list[str]:
    """Depth-ascending suffix strings (e.g. `"2"`, `"107"`, `"2.5m"`, `"surf"`)
    for every `{prefix}_<suffix>` channel present in `channels`."""
    pairs = [
        (_depth_suffix_value(c[len(prefix) + 1 :]), c[len(prefix) + 1 :])
        for c in channels
        if c.startswith(prefix + "_") and _depth_suffix_value(c[len(prefix) + 1 :]) is not None
    ]
    return [suffix for _, suffix in sorted(pairs)]


def depth_value(prefix: str, suffix: str, nz1=None) -> float:
    """Real display depth (m) for one `{prefix}_{suffix}` channel. FESOM:
    resolved via the mesh's own nz1 lookup (`meshtools.parse_channel`) when
    `nz1` is given. EERIE (`nz1=None`, or any suffix `parse_channel` itself
    doesn't recognize): the suffix already IS the depth (see
    `_depth_suffix_value`)."""
    if nz1 is not None:
        try:
            _, real_depth, _ = parse_channel(f"{prefix}_{suffix}", nz1)
            return real_depth
        except ValueError:
            pass
    depth = _depth_suffix_value(suffix)
    if depth is None:
        raise ValueError(f"cannot determine depth for channel '{prefix}_{suffix}'")
    return depth


def _open_required(cache_paths: dict, fig_id: str, *keys: str) -> list[xr.Dataset] | None:
    """Open every `keys` cache file, or print a warning and return None if any is missing.

    A `scores_{stream}.nc` built after Task 20 carries a `region` dimension on
    `score` (global/nh/sh per-sample aggregation); every figure here predates
    regions and must keep aggregating over the whole mesh, so datasets with that
    dimension are immediately narrowed to `region="global"` (which also drops the
    now-scalar `region` coordinate) -- the rest of this module never has to know
    the dimension exists. A pre-Task-20 cache has no `region` dim and passes
    through unchanged.
    """
    paths = []
    for key in keys:
        p = cache_paths.get(key)
        if p is None or not Path(p).exists():
            print(f"[skip] {fig_id}: missing cache input {key!r}")
            return None
        paths.append(Path(p))
    opened = [xr.open_dataset(p) for p in paths]
    return [ds.sel(region="global") if "region" in ds.dims else ds for ds in opened]


def _sorted_depth_channels(channels, prefix: str) -> list[tuple[float, str]]:
    """`[(depth, channel_name), ...]` for `{prefix}_<suffix>` channels,
    depth-ascending -- thin wrapper around `depth_suffixes`/`_depth_suffix_value`
    for callers (f03's scorecard columns, f11's summary heatmap columns) that
    just need channel names in depth order, not the raw suffix string."""
    return [(_depth_suffix_value(s), f"{prefix}_{s}") for s in depth_suffixes(channels, prefix)]


# ── f01: surface (sst/sss/ssh or avg_tos/avg_sos/avg_zos) scores vs lead ──────────────
def _damped_rmse(cache_paths: dict, stream: str) -> xr.DataArray | None:
    """Task 28 (oe3i7whz only, for now): the damped-persistence RMSE curve, `(lead,
    channel)`, fit post hoc from `damped_{stream}.nc`'s cached sufficient statistics
    -- `None` if that cache file does not exist (every run except oe3i7whz, and
    oe3i7whz before `compute_damped.py` is run), which is what keeps `build_f01`'s
    output byte-identical to before this Task for every other run/config.

    `alpha*(L) = sum_sample(S0L) / sum_sample(S00)`, region "global", matching
    `compute_damped.py`'s module docstring; RMSE(L) is then
    `sqrt(sum_sample(alpha*^2*S00 - 2*alpha*S0L + SLL) / sum_sample(W))`.
    """
    damped_path = cache_paths[f"scores_{stream}"].parent / "damped_FESOM_NODE.nc"
    if not damped_path.exists():
        return None
    with xr.open_dataset(damped_path) as damped:
        d = damped.sel(region="global").sum("sample")
        alpha_star = d["S0L"] / d["S00"]
        rmse = np.sqrt((alpha_star**2 * d["S00"] - 2 * alpha_star * d["S0L"] + d["SLL"]) / d["W"])
        return rmse.load()


def build_f01(cfg: dict, cache_paths: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    stream = gs["surface_stream"]
    opened = _open_required(cache_paths, "f01", f"scores_{stream}")
    if opened is None:
        return []
    (ds,) = opened
    score = ds["score"]

    cols = [c for c in gs["surface_channels"] if c in ds.channel.values]
    if not cols:
        print(f"[skip] f01: none of {gs['surface_channels']} present in scores_{stream}")
        return []

    leads = ds.lead.values
    damped_rmse = _damped_rmse(cache_paths, stream)
    fig, axes = plt.subplots(
        2, len(cols), figsize=(3.4 * len(cols), 5.8), sharex=True, constrained_layout=True
    )
    axes = np.asarray(axes).reshape(2, len(cols))

    legend_handles: dict[str, Line2D] = {}
    for j, ch in enumerate(cols):
        ax_rmse, ax_acc = axes[0, j], axes[1, j]

        # Task 28: drawn first (beneath, per the brief) so the emulator's line -- and
        # anything else this axes still plots -- renders on top of it, grey/dash-dot/
        # thinner than the default (rcParams) line width. ACC row: nothing new (an
        # alpha-damped forecast has no separate ACC statistic cached here).
        if damped_rmse is not None and ch in damped_rmse.channel.values:
            (line,) = ax_rmse.plot(
                damped_rmse.lead.values, damped_rmse.sel(channel=ch).values,
                color=style.MODEL_COLORS["persistence"], linestyle="-.", linewidth=1.0,
                label="Damped persistence",
            )
            legend_handles["damped"] = line

        for model in ("emulator", "persistence", "climatology"):
            rmse = _agg_rmse(score.sel(model=model, metric="rmse", channel=ch))
            (line,) = ax_rmse.plot(
                leads, rmse.values, color=style.MODEL_COLORS[model],
                linestyle=style.MODEL_LINESTYLES[model], label=style.MODEL_LABELS[model],
            )
            legend_handles[model] = line
        # ACC row: climatology's ACC is undefined (f'=clim-clim=0 -> 0/0 NaN), so it is
        # deliberately excluded here, not just left to fail silently.
        for model in ("emulator", "persistence"):
            acc = _agg_mean(score.sel(model=model, metric="acc", channel=ch))
            (line,) = ax_acc.plot(
                leads, acc.values, color=style.MODEL_COLORS[model],
                linestyle=style.MODEL_LINESTYLES[model], label=style.MODEL_LABELS[model],
            )
            legend_handles[model] = line

        for lead_mark in (15, 30):
            if lead_mark in leads:
                for ax in (ax_rmse, ax_acc):
                    ax.axvline(lead_mark, color="#c3c2b7", linewidth=0.8, linestyle=":", zorder=0)

        unit = style.channel_unit(ch)
        ax_rmse.set_title(style.channel_label(ch))
        ax_rmse.set_ylabel(f"RMSE [{unit}]" if unit else "RMSE")
        ax_rmse.set_ylim(bottom=0)
        ax_rmse.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        style.no_sci_offset(ax_rmse)
        ax_acc.set_ylabel("ACC")
        ax_acc.set_xlabel("lead [d]")
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    fig.legend(
        legend_handles.values(), [h.get_label() for h in legend_handles.values()],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle("Surface scores vs. lead time")
    return style.savefig(fig, "f01_surface_scores_vs_lead", fig_dir(cfg))


# ── f02: depth-resolved (temp/salt/u/v or their EERIE equivalents) scores vs lead,
#         one line per highlighted depth ───────────────────────────────────────────────
def build_f02(cfg: dict, cache_paths: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    geometry = cfg.get("geometry", "fesom")
    streams = sorted({stream for _, _, stream in gs["depth_groups"]})
    ds_by_stream = _open_by_stream(cache_paths, "f02", "scores", streams)
    if ds_by_stream is None:
        return []
    mesh = get_mesh(cfg["mesh_path"]) if geometry == "fesom" else None
    nz1 = mesh.nz1 if mesh is not None else None

    if geometry == "eerie":
        # EERIE has only 3-4 levels total per variable (no `figures.curve_depths`
        # hand-picked subset needed/applicable -- that key is FESOM-only, see
        # configs/dbuzcpgf.yaml's comment): highlight every depth of the
        # superset variable (`depth_axis_prefix`), same set f06's heatmap uses.
        canonical_prefix = gs["depth_axis_prefix"]
        canonical_stream = next(s for _, p, s in gs["depth_groups"] if p == canonical_prefix)
        depths = depth_suffixes(ds_by_stream[canonical_stream].channel.values, canonical_prefix)
    else:
        depths = [str(d) for d in cfg["figures"]["curve_depths"]]
    depth_hues = style.depth_colors(len(depths))

    var_specs = [(prefix, ds_by_stream[stream]) for _, prefix, stream in gs["depth_groups"]]
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.5), sharex=True, constrained_layout=True)

    depth_handles: dict[float, Line2D] = {}
    for j, (prefix, ds) in enumerate(var_specs):
        score = ds["score"]
        leads = ds.lead.values
        ax_rmse, ax_acc = axes[0, j], axes[1, j]

        for d_idx, depth_suffix in enumerate(depths):
            ch = f"{prefix}_{depth_suffix}"
            if ch not in ds.channel.values:
                continue  # skip missing channel (e.g. FESOM's v_210 is absent by design)
            color = depth_hues[d_idx]
            real_depth = depth_value(prefix, depth_suffix, nz1)

            rmse_e = _agg_rmse(score.sel(model="emulator", metric="rmse", channel=ch))
            (line,) = ax_rmse.plot(
                leads, rmse_e.values, color=color, linestyle="-", label=f"{real_depth:g} m"
            )
            rmse_p = _agg_rmse(score.sel(model="persistence", metric="rmse", channel=ch))
            ax_rmse.plot(
                leads, rmse_p.values, color=color, linestyle="--", alpha=0.45, linewidth=1.2
            )

            acc_e = _agg_mean(score.sel(model="emulator", metric="acc", channel=ch))
            ax_acc.plot(leads, acc_e.values, color=color, linestyle="-")
            acc_p = _agg_mean(score.sel(model="persistence", metric="acc", channel=ch))
            ax_acc.plot(leads, acc_p.values, color=color, linestyle="--", alpha=0.45, linewidth=1.2)

            depth_handles[real_depth] = line

        unit = style.channel_unit(prefix)
        ax_rmse.set_title(style.PREFIX_DISPLAY.get(prefix, prefix))
        ax_rmse.set_ylabel(f"RMSE [{unit}]" if unit else "RMSE")
        ax_rmse.set_ylim(bottom=0)
        ax_rmse.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        style.no_sci_offset(ax_rmse)
        ax_acc.set_ylabel("ACC")
        ax_acc.set_xlabel("lead [d]")
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.xaxis.set_major_locator(MaxNLocator(6, integer=True))

    handles = list(depth_handles.values())
    labels = [h.get_label() for h in handles]
    handles.append(Line2D([0], [0], color="0.4", linestyle="--", alpha=0.6))
    labels.append("persistence (faint, dashed)")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Depth-resolved scores vs. lead time (emulator solid, persistence faint dashed)")
    return style.savefig(fig, "f02_depth_scores_vs_lead", fig_dir(cfg))


#: Rotated-annotation sizing (Task 23). Unrotated, a "+0.12"-style annotation's
#: on-screen width scales with its *string length* (~5 characters); at the channel
#: counts these grids run at (~80+ columns, ~10 pt/cell) that collides with the
#: neighboring cell regardless of font size. Rotated 90 deg, the string length
#: becomes the annotation's *vertical* extent instead, and its horizontal footprint
#: drops to roughly one glyph's width -- empirically ~1.0x the font size in points,
#: independent of string length (checked directly against rendered text bboxes).
#: `_ANNOT_WIDTH_FRACTION` budgets that footprint to 60% of the measured cell
#: width, leaving a visual gap to the next column; `_ANNOT_FONT_FLOOR` is the
#: legibility floor below which annotations are dropped entirely rather than
#: shrunk further, and `_ANNOT_FONT_CEIL` keeps wide cells (few-channel grids)
#: from getting oversized annotations.
_ANNOT_WIDTH_FRACTION = 0.6
_ANNOT_FONT_FLOOR = 5.0
_ANNOT_FONT_CEIL = 7.0


def _draw_scorecard_grid(
    ax, data, row_labels, col_labels, group_boundaries, *, vmin, vmax, cmap
):
    """Draw one scorecard cell grid (imshow + value annotations + row/column labels
    + vertical group-separator lines + solid frame) onto `ax`. `data` is `(n_rows,
    n_cols)`: channels sit along the (rotated-label) x axis, (lead, baseline)
    combinations along the y axis -- the full multi-row grid for the combined
    heatmap, or a single `(1, n_cols)` row slice for one of the individual scorecard
    strips; either way this is the one place that assembles a scorecard cell grid,
    so `build_f03`'s combined and per-row paths (and `seasonal.py`'s f11 summary)
    never duplicate the imshow/annotate/label logic. `group_boundaries` are x
    positions, already offset by -0.5 (i.e. ready for `axvline`), between
    contiguous channel groups (e.g. Temperature | Salinity). Returns the `im`
    (AxesImage) for the caller to attach its own colorbar to.

    Per-cell value annotations are rotated 90 deg and sized from the axes' own
    rendered width (via the canvas renderer, not guessed from figure inches) --
    see `_ANNOT_WIDTH_FRACTION` et al. above. If even the smallest usable font
    would not fit, annotations are suppressed for the whole grid (all-or-nothing --
    a mix of annotated/blank cells would read as a data gap, not a legibility
    choice) and a caption pointing at the colorbar is drawn in their place.
    """
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    n_rows, n_cols = data.shape

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=6, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(axis="both", length=0)
    style.full_frame(ax)

    for b in group_boundaries:
        ax.axvline(b, color="white", linewidth=1.5)

    fig = ax.figure
    fig.canvas.draw()  # realize the axes' current layout so its extent is real
    cell_width_pt = (
        ax.get_window_extent(renderer=fig.canvas.get_renderer()).width
        / max(n_cols, 1)
        / fig.dpi
        * 72.0
    )
    fontsize_raw = cell_width_pt * _ANNOT_WIDTH_FRACTION

    if fontsize_raw < _ANNOT_FONT_FLOOR:
        ax.set_xlabel(
            "values omitted for readability; see colorbar", fontsize=7, style="italic"
        )
    else:
        fontsize_annot = min(_ANNOT_FONT_CEIL, fontsize_raw)
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if np.isfinite(v):
                    txt_color = "white" if abs(v) > 0.6 * vmax else "#0b0b0b"
                    ax.text(
                        j, i, f"{v:+.2f}", ha="center", va="center", rotation=90,
                        fontsize=fontsize_annot, color=txt_color,
                    )
    return im


# ── f03: scorecard heatmap (skill vs. persistence / climatology, all channels) ────────
def build_f03(cfg: dict, cache_paths: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    geometry = cfg.get("geometry", "fesom")
    streams = sorted({gs["surface_stream"]} | {s for _, _, s in gs["depth_groups"]})
    ds_by_stream = _open_by_stream(cache_paths, "f03", "scores", streams)
    if ds_by_stream is None:
        return []
    mesh = get_mesh(cfg["mesh_path"]) if geometry == "fesom" else None
    nz1 = mesh.nz1 if mesh is not None else None

    # Defensive lead filtering (same spirit as f04/f05/f06/f07): a configured
    # map_lead absent from any needed scores cache would otherwise raise a
    # KeyError out of `.sel()` below, into the CLI's outer catch as a hard
    # [error] instead of a graceful [skip] -- so intersect against what every
    # dataset actually has, keeping one consistent row set across every
    # baseline/lead combination.
    map_leads_cfg = list(cfg["leads"]["map_leads"])
    avail = set.intersection(*(set(ds.lead.values.tolist()) for ds in ds_by_stream.values()))
    map_leads = [lead for lead in map_leads_cfg if lead in avail]
    missing = [lead for lead in map_leads_cfg if lead not in map_leads]
    if missing:
        print(f"[skip] f03: lead(s) {missing} not present in every scores cache -- dropping row")
    if not map_leads:
        print("[skip] f03: none of leads.map_leads present in every needed scores cache")
        return []

    groups = [(gname, chans, ds) for gname, chans, ds, _stream in _channel_groups(gs, ds_by_stream)]

    # (lead, baseline) combinations, baseline outer / lead inner, as the four Y rows
    # of the combined heatmap (and each individual scorecard's single row): "15 d vs
    # persistence", "30 d vs persistence", "15 d vs climatology", "30 d vs climatology".
    row_combos = [(lead, baseline) for baseline in ("persistence", "climatology") for lead in map_leads]
    row_labels = [f"{lead} d vs {baseline}" for lead, baseline in row_combos]

    # Channels now run along the X axis, grouped (Surface | Temperature | Salinity |
    # u velocity | v velocity) with a vertical separator line -- not a spacer row/
    # column -- between contiguous groups; `group_boundaries` collects the x
    # position (already offset -0.5, ready for axvline) before each non-first group.
    col_labels: list[str] = []
    group_boundaries: list[float] = []
    data_cols: list[list[float]] = []
    for gname, chans, ds in groups:
        if not chans:
            continue
        if data_cols:
            group_boundaries.append(len(data_cols) - 0.5)
        score = ds["score"]
        for ch in chans:
            col = []
            for lead, baseline in row_combos:
                sel_e = score.sel(model="emulator", metric="rmse", lead=lead, channel=ch)
                sel_b = score.sel(model=baseline, metric="rmse", lead=lead, channel=ch)
                rmse_e = float(_agg_rmse(sel_e))
                rmse_b = float(_agg_rmse(sel_b))
                with np.errstate(invalid="ignore", divide="ignore"):
                    skill = 1.0 - rmse_e / rmse_b
                col.append(skill if np.isfinite(skill) else np.nan)
            data_cols.append(col)
            col_labels.append(style.channel_label(ch, nz1))

    data = np.array(data_cols, dtype=np.float64).T  # (n_rows=len(row_combos), n_cols=nchan)
    n_cols = data.shape[1]

    vmin, vmax = style.symmetric_limits(data, pct=99.0)
    vmin = -vmax

    cmap = plt.get_cmap(style.DIVERGING_CMAP).copy()
    cmap.set_bad("white")

    fig_w = max(10.0, 0.16 * n_cols + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    im = _draw_scorecard_grid(
        ax, data, row_labels, col_labels, group_boundaries, vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_title(
        "Skill score  1 - RMSE_emulator / RMSE_baseline   (positive = emulator better)", fontsize=9
    )
    if len(map_leads) > 1:
        # Separates the persistence-baseline rows from the climatology-baseline rows.
        ax.axhline(len(map_leads) - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("skill score")
    fig.tight_layout()

    out_dir = fig_dir(cfg)
    written = style.savefig(fig, "f03_scorecard", out_dir)

    # Individual per-row scorecards (Task 18/19): one (lead, baseline) row -- all
    # channel columns -- as its own single horizontal strip, reusing the exact same
    # `_draw_scorecard_grid` cell-drawing code, the same `vmin`/`vmax`/`cmap`, and the
    # same x-labels as the combined heatmap above -- written in this same pass over
    # `data`, no recomputation of the skill values.
    baselines = (("pers", "persistence"), ("clim", "climatology"))
    for lead in map_leads:
        for baseline_token, baseline_name in baselines:
            row = row_combos.index((lead, baseline_name))
            row_data = data[row : row + 1, :]
            fig_c, ax_c = plt.subplots(figsize=(fig_w, 2.2))
            im_c = _draw_scorecard_grid(
                ax_c, row_data, [row_labels[row]], col_labels, group_boundaries,
                vmin=vmin, vmax=vmax, cmap=cmap,
            )
            ax_c.set_title(f"Skill vs {baseline_name}  --  lead {lead} d", fontsize=9)
            cbar_c = fig_c.colorbar(im_c, ax=ax_c, shrink=0.8, pad=0.01)
            cbar_c.set_label("skill score")
            fig_c.tight_layout()
            written += style.savefig(fig_c, f"f03_scorecard_{lead}d_{baseline_token}", out_dir)

    return written
