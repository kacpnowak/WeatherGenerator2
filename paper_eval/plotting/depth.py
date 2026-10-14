"""f06, f07: lead-depth heatmaps and depth profiles, from `scores_{stream}.nc`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from paper_eval.config import fig_dir
from paper_eval.meshtools import masks_and_weights
from paper_eval.plotting import style
from paper_eval.plotting.curves import (
    _open_by_stream,
    _sorted_depth_channels,
    depth_suffixes,
    depth_value,
    geometry_streams,
)
from paper_eval.plotting.fesom_maps import get_mesh

#: Preferred lead tick values for the f06 heatmap x-axis (imshow's x-axis is
#: categorical -- one column per lead index -- so ticks must be picked explicitly;
#: one label per lead (leads.max=30) would overlap into an unreadable smear).
_LEAD_TICK_TARGETS = (1, 5, 10, 15, 20, 25, 30)


def _thin_categorical_ticks(values, targets=None, max_ticks: int = 6) -> tuple[list[int], list]:
    """`(positions, labels)` for a categorical (imshow-style) axis with one slot per
    entry of `values`, thinned to stay legible.

    If `targets` is given, keep the entries of `values` that match one of `targets`
    (e.g. lead 1/5/10/.../30) -- meaningful, human-picked ticks rather than an
    arbitrary stride. Falls back to an evenly spaced subsample (always including
    both ends) if none of `targets` are present (e.g. a short debug run) or if
    `targets` is None (e.g. the depth axis, whose values aren't round numbers) --
    never one tick per entry.
    """
    values = list(values)
    n = len(values)
    if targets is not None:
        picks = [(i, v) for i, v in enumerate(values) if v in targets]
        if picks:
            positions, labels = zip(*picks)
            return list(positions), list(labels)
    if n <= max_ticks:
        return list(range(n)), values
    idx = sorted({int(round(i)) for i in np.linspace(0, n - 1, max_ticks)})
    return idx, [values[i] for i in idx]


def _agg_rmse(da: xr.DataArray) -> xr.DataArray:
    return np.sqrt((da**2).mean("sample"))


def _agg_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean("sample")


def _depth_profile(
    score: xr.DataArray, agg, model: str, metric: str, lead: int, depth_channels: list[str],
) -> list[float]:
    """Aggregated `metric` for `model`/`lead`, one value per channel in `depth_channels`."""
    return [
        float(agg(score.sel(model=model, metric=metric, lead=lead, channel=ch)))
        for ch in depth_channels
    ]


def _area_weighted_mean_fesom(values: np.ndarray, mask: np.ndarray, w: np.ndarray) -> float:
    """Area-weighted mean of `values` over the wet (`mask`) points of one channel.

    `maps_{stream}.nc`'s `pred_mean`/`target_mean` carry NaN or other garbage at dry
    cells (see meshtools.masks_and_weights docstring: the mesh-derived `mask` is the
    sole wet/dry authority, never the data). `weight` is zero at every dry point, so
    the numerator's `values * weight` term there is `garbage * 0` -- possibly itself
    NaN/inf (e.g. `0 * nan == nan`), which `np.nansum` then correctly drops (treats as
    0) rather than poisoning the whole sum; the denominator (`weight` alone, no data
    values involved) is always finite. Returns NaN if a channel has zero wet weight
    (should not happen for a real channel, but avoids a 0/0 division).
    """
    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(mask, dtype=bool).astype(np.float64) * np.asarray(w, dtype=np.float64)
    denom = float(np.sum(weight))
    if denom <= 0:
        return float("nan")
    return float(np.nansum(values * weight)) / denom


def _area_weighted_profile_fesom(
    maps_ds: xr.Dataset, var: str, lead: int, depth_channels: list[str], mesh, stream: str,
) -> list[float]:
    """Area-weighted mean of `maps_ds[var]` (e.g. `"pred_mean"`/`"target_mean"`) at
    `lead`, one value per channel in `depth_channels` -- the spatial reduction of the
    already time-averaged, per-point `maps_{stream}.nc` field down to one scalar per
    depth channel, weighted by the FESOM mesh's own per-point area (`nod_area`/
    `elem_area`) and restricted to wet points (`meshtools.masks_and_weights`).
    """
    mask, w = masks_and_weights(mesh, stream, depth_channels)
    return [
        _area_weighted_mean_fesom(maps_ds[var].sel(lead=lead, channel=ch).values, mask[i], w[i])
        for i, ch in enumerate(depth_channels)
    ]


def _area_weighted_mean_grid(values: np.ndarray, count: np.ndarray, lat: np.ndarray) -> float:
    """EERIE geometry counterpart of `_area_weighted_mean_fesom`: weight by
    `cos(lat)` (the regular 0.25 deg grid's own cell-area proxy) restricted to
    valid cells (`count > 0` -- the per-cell accumulated-sample count
    `scores._reduce_blocks_eerie` writes, this geometry's wet/valid authority in
    place of FESOM's fixed mesh mask; see `eerie_grid.py`'s module docstring).
    Returns NaN if no cell is valid.
    """
    valid = np.asarray(count) > 0
    weight = np.cos(np.radians(np.asarray(lat, dtype=np.float64))) * valid.astype(np.float64)
    denom = float(np.sum(weight))
    if denom <= 0:
        return float("nan")
    return float(np.nansum(np.asarray(values, dtype=np.float64) * weight)) / denom


def _area_weighted_profile_grid(
    maps_ds: xr.Dataset, var: str, lead: int, depth_channels: list[str],
) -> list[float]:
    """EERIE geometry counterpart of `_area_weighted_profile_fesom`, one value per
    channel in `depth_channels` -- reads `lat`/`count` straight off `maps_ds`
    itself (the full-grid `maps_{stream}.nc` already carries both; no mesh
    needed)."""
    lat = maps_ds["lat"].values
    return [
        _area_weighted_mean_grid(
            maps_ds[var].sel(lead=lead, channel=ch).values,
            maps_ds["count"].sel(lead=lead, channel=ch).values,
            lat,
        )
        for ch in depth_channels
    ]


# ── f06: lead-depth score heatmaps ─────────────────────────────────────────────────────
def build_f06(cfg: dict, cache_paths: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    geometry = cfg.get("geometry", "fesom")
    streams = sorted({s for _, _, s in gs["depth_groups"]})
    ds_by_stream = _open_by_stream(cache_paths, "f06", "scores", streams)
    if ds_by_stream is None:
        return []
    mesh = get_mesh(cfg["mesh_path"]) if geometry == "fesom" else None
    nz1 = mesh.nz1 if mesh is not None else None

    # Canonical depth axis: `depth_axis_prefix`'s own channel set is a superset of
    # every other depth group's (FESOM: temp/salt/u all carry the full 20-level
    # set, v is missing 210 m; EERIE: avg_uoe/avg_von carry "surf" plus the same 3
    # levels avg_thetao/avg_so do) -- a channel absent at a given depth for one
    # variable renders as a NaN gap row at that index so all four columns' depth
    # axes stay aligned (e.g. FESOM's v @ 210 m, or EERIE's avg_thetao/avg_so
    # missing a "surf" level).
    canonical_prefix = gs["depth_axis_prefix"]
    canonical_stream = next(s for _, p, s in gs["depth_groups"] if p == canonical_prefix)
    depth_suffixes_ = depth_suffixes(
        ds_by_stream[canonical_stream].channel.values, canonical_prefix
    )
    depth_labels = [f"{depth_value(canonical_prefix, s, nz1):g}" for s in depth_suffixes_]
    leads = ds_by_stream[canonical_stream].lead.values

    # Thinned tick sets, computed once and shared by every panel/row: the lead axis
    # picks round lead values (1/5/10/.../30), the depth axis an evenly spaced
    # subsample -- one tick per lead/level would overlap unreadably.
    lead_ticks, lead_tick_labels = _thin_categorical_ticks(
        leads.tolist(), targets=_LEAD_TICK_TARGETS
    )
    depth_ticks, depth_tick_labels = _thin_categorical_ticks(depth_labels)

    var_specs = [(prefix, ds_by_stream[stream]) for _, prefix, stream in gs["depth_groups"]]
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.5), sharey="row", constrained_layout=True)

    gap_cmap = plt.get_cmap(style.SEQUENTIAL_CMAP).copy()
    gap_cmap.set_bad("#e1e0d9")

    # Per-run opt-in: top row plots skill against climatology instead of absolute RMSE
    # instead of absolute RMSE, so the panel shows relative skill rather than raw
    # amplitude (which falls with depth regardless of skill). Absolute RMSE remains
    # available elsewhere (the scores table, f07's middle row); gated so every other
    # run's f06 renders byte-identical to today.
    normalise_rmse = bool(cfg.get("figures", {}).get("depth_rmse_normalised", False))
    if normalise_rmse:
        nrmse_cmap = plt.get_cmap(style.DIVERGING_CMAP).copy()
        nrmse_cmap.set_bad("#e1e0d9")

    for j, (prefix, ds) in enumerate(var_specs):
        score = ds["score"]
        rmse_grid = np.full((len(depth_suffixes_), len(leads)), np.nan)
        acc_grid = np.full((len(depth_suffixes_), len(leads)), np.nan)
        clim_rmse_grid = (
            np.full((len(depth_suffixes_), len(leads)), np.nan) if normalise_rmse else None
        )
        for i, suffix in enumerate(depth_suffixes_):
            ch = f"{prefix}_{suffix}"
            if ch not in ds.channel.values:
                continue  # gap row (e.g. FESOM's v_210, or EERIE's avg_thetao_surf)
            sel = {"model": "emulator", "channel": ch}
            rmse_grid[i, :] = _agg_rmse(score.sel(metric="rmse", **sel)).values
            acc_grid[i, :] = _agg_mean(score.sel(metric="acc", **sel)).values
            if normalise_rmse:
                clim_sel = {"model": "climatology", "channel": ch}
                clim_rmse_grid[i, :] = _agg_rmse(score.sel(metric="rmse", **clim_sel)).values

        ax_rmse, ax_acc = axes[0, j], axes[1, j]
        if normalise_rmse:
            # Diverging and centred on zero, clipped to [-1, 1]: the eye reads a
            # balanced colour scale as signed about zero, which the raw ratio
            # (neutral at 1) does not support. Same palette and limits as the skill
            # maps, so "more accurate than climatology" carries one colour throughout
            # the paper. Gap rows (NaN / NaN) stay NaN, same grey as the other panels.
            with np.errstate(invalid="ignore", divide="ignore"):
                # Skill score rather than the raw ratio: a diverging colour scale is
                # read as centred on zero, and this is the same 1 - RMSE/RMSE_ref
                # definition used by the scorecard and the skill maps. Positive means
                # more accurate than climatology, zero means equally accurate.
                top_grid = 1.0 - rmse_grid / clim_rmse_grid
            im_r = ax_rmse.imshow(
                top_grid, aspect="auto", cmap=nrmse_cmap, vmin=-1.0, vmax=1.0
            )
            fig.colorbar(im_r, ax=ax_rmse, shrink=0.85).set_label(
                "skill vs climatology (clipped at -1)"
            )
        else:
            vmin_r, vmax_r = style.sequential_limits(rmse_grid, pct=99.0)
            im_r = ax_rmse.imshow(rmse_grid, aspect="auto", cmap=gap_cmap, vmin=vmin_r, vmax=vmax_r)
            unit = style.channel_unit(prefix)
            fig.colorbar(im_r, ax=ax_rmse, shrink=0.85).set_label(f"RMSE [{unit}]" if unit else "RMSE")
        ax_rmse.set_title(style.PREFIX_DISPLAY.get(prefix, prefix))

        im_a = ax_acc.imshow(acc_grid, aspect="auto", cmap=gap_cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(im_a, ax=ax_acc, shrink=0.85).set_label("ACC")

        for ax in (ax_rmse, ax_acc):
            ax.set_xticks(lead_ticks)
            ax.set_xticklabels(lead_tick_labels)
            ax.grid(False)
            style.full_frame(ax)
        # `sharey="row"` makes every axes in a row share ONE Locator/Formatter
        # object (verified: id(ax.yaxis.get_major_locator()) is identical across
        # the row) -- calling set_yticks/set_yticklabels on column j>0 does not
        # just affect that axes, it overwrites the *shared* formatter and would
        # blank out column 0's labels too (this was the actual cause of "y axes
        # show no numbers": the old code's `else: ax.set_yticklabels([])` on the
        # later columns silently wiped out column 0's labels set earlier in the
        # loop). So the y ticks/labels are set exactly once per row, from column 0
        # only; the non-first columns rely on `plt.subplots(sharey=...)`'s own
        # default (label1On=False there already) to stay unlabeled -- they must
        # not call set_yticks/set_yticklabels at all.
        if j == 0:
            for ax in (ax_rmse, ax_acc):
                ax.set_yticks(depth_ticks)
                ax.set_yticklabels(depth_tick_labels)
                ax.set_ylabel("depth [m]")
        ax_acc.set_xlabel("lead [d]")

    if normalise_rmse:
        fig.suptitle(
            "Lead-depth score heatmaps (top row: skill against climatology, "
            "1 - RMSE/RMSE_clim; grey gap = channel absent at that depth)"
        )
    else:
        fig.suptitle("Lead-depth score heatmaps (emulator; grey gap = channel absent at that depth)")
    return style.savefig(fig, "f06_lead_depth_heatmaps", fig_dir(cfg))


# ── f07: depth profiles of ground truth / emulator mean, RMSE and bias ────────────────
def build_f07(cfg: dict, cache_paths: dict) -> list[Path]:
    gs = geometry_streams(cfg)
    geometry = cfg.get("geometry", "fesom")
    streams = sorted({s for _, _, s in gs["depth_groups"]})
    scores_by_stream = _open_by_stream(cache_paths, "f07", "scores", streams)
    if scores_by_stream is None:
        return []
    maps_by_stream = _open_by_stream(cache_paths, "f07", "maps", streams)
    if maps_by_stream is None:
        return []
    mesh = get_mesh(cfg["mesh_path"]) if geometry == "fesom" else None

    # Defensive lead filtering (same spirit as f03/f04/f05): a configured map_lead
    # absent from any needed cache would otherwise raise a KeyError out of
    # `.sel()` below, into the CLI's outer catch as a hard [error] instead of a
    # graceful [skip] -- so intersect against what every dataset actually has.
    map_leads_cfg = list(cfg["leads"]["map_leads"])
    avail = set.intersection(*(
        set(ds.lead.values.tolist())
        for by_stream in (scores_by_stream, maps_by_stream)
        for ds in by_stream.values()
    ))
    map_leads = [lead for lead in map_leads_cfg if lead in avail]
    missing = [lead for lead in map_leads_cfg if lead not in map_leads]
    if missing:
        print(f"[skip] f07: lead(s) {missing} not present in every scores/maps cache -- dropping")
    if not map_leads:
        print("[skip] f07: none of leads.map_leads present in every scores_/maps_ cache")
        return []
    lead_solid, lead_dashed = map_leads[0], map_leads[-1]

    var_specs = [
        (prefix, scores_by_stream[stream], maps_by_stream[stream], stream)
        for _, prefix, stream in gs["depth_groups"]
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15.0, 10.0), sharey="row", constrained_layout=True)

    for j, (prefix, ds, maps_ds, stream) in enumerate(var_specs):
        score = ds["score"]
        depth_channels = [c for _, c in _sorted_depth_channels(ds.channel.values, prefix)]
        real_depths = [depth_value(prefix, c[len(prefix) + 1 :], mesh.nz1 if mesh else None)
                        for c in depth_channels]

        unit = style.channel_unit(prefix)

        # Row 0 (new): area-weighted mean profile -- ground truth (target_mean at the
        # first/primary map lead, lead-independent in expectation) vs. emulator
        # (pred_mean) at lead_solid (solid) and lead_dashed (dashed). Wet-point-only,
        # area-weighted -- FESOM via the mesh's own area weights
        # (`_area_weighted_profile_fesom`), EERIE via cos(lat) + `count > 0`
        # (`_area_weighted_profile_grid`); see either docstring for why the
        # geometry's own wet/valid authority, not the data, is the exclusion rule.
        ax_mean = axes[0, j]
        if geometry == "eerie":
            truth_vals = _area_weighted_profile_grid(
                maps_ds, "target_mean", lead_solid, depth_channels
            )
        else:
            truth_vals = _area_weighted_profile_fesom(
                maps_ds, "target_mean", lead_solid, depth_channels, mesh, stream
            )
        ax_mean.plot(truth_vals, real_depths, color="black", linestyle="-", label="ground truth")
        for lead, ls in ((lead_solid, "-"), (lead_dashed, "--")):
            if geometry == "eerie":
                pred_vals = _area_weighted_profile_grid(maps_ds, "pred_mean", lead, depth_channels)
            else:
                pred_vals = _area_weighted_profile_fesom(
                    maps_ds, "pred_mean", lead, depth_channels, mesh, stream
                )
            ax_mean.plot(
                pred_vals, real_depths, color=style.lead_color(lead), linestyle=ls,
                label=f"emulator (lead {lead})",
            )
        ax_mean.set_xlabel(f"mean [{unit}]" if unit else "mean")
        ax_mean.xaxis.set_major_locator(MaxNLocator(5))
        style.no_sci_offset(ax_mean, axis="x")
        ax_mean.set_title(style.PREFIX_DISPLAY.get(prefix, prefix))
        if j == 0:
            ax_mean.set_ylabel("depth [m]")
            # Axes share the y-axis within a row (sharey="row"): inverting it from
            # every column would toggle the shared state back and forth (an even
            # number of calls cancels out entirely, e.g. 4 columns), so invert
            # exactly once per row, from the first column only.
            ax_mean.invert_yaxis()

        # Rows 1-2 (unchanged): RMSE and bias profiles.
        for metric_row, metric in enumerate(("rmse", "bias"), start=1):
            ax = axes[metric_row, j]
            agg = _agg_rmse if metric == "rmse" else _agg_mean

            for lead, ls in ((lead_solid, "-"), (lead_dashed, "--")):
                vals = _depth_profile(score, agg, "emulator", metric, lead, depth_channels)
                ax.plot(
                    vals, real_depths, color=style.lead_color(lead), linestyle=ls,
                    label=f"emulator (lead {lead})",
                )
            vals_p = _depth_profile(score, agg, "persistence", metric, lead_solid, depth_channels)
            ax.plot(
                vals_p, real_depths, color=style.MODEL_COLORS["persistence"], linestyle="-",
                alpha=0.45, linewidth=1.2, label=f"persistence (lead {lead_solid})",
            )

            if metric == "bias":
                ax.axvline(0, color="#898781", linewidth=0.8, linestyle=":")

            ax.set_xlabel(f"{metric.upper()} [{unit}]" if unit else metric.upper())
            ax.xaxis.set_major_locator(MaxNLocator(5))
            style.no_sci_offset(ax, axis="x")
            if j == 0:
                ax.set_ylabel("depth [m]")
                ax.invert_yaxis()

    # Legend handles come from two rows (mean profile + RMSE), deduplicated by label:
    # the mean-profile row alone contributes "ground truth" (unique to row 0), the
    # RMSE row alone contributes "persistence (lead N)" (not plotted in row 0); the
    # "emulator (lead N)" pair is common to every row and kept once.
    handles, labels = [], []
    for ax_ in (axes[0, 0], axes[1, 0]):
        for h, label in zip(*ax_.get_legend_handles_labels()):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Depth profiles: area-weighted mean (ground truth vs. emulator), RMSE, and bias")
    return style.savefig(fig, "f07_depth_profiles", fig_dir(cfg))
