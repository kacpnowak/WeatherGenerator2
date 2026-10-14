"""EERIE gridded-geometry map figures (f04/f05/f08/f10) for run dbuzcpgf's
OCEAN_2D/OCEAN_3D streams -- the `cfg["geometry"] == "eerie"` counterpart of
`fesom_maps.py`'s native-mesh map figures.

Mechanics: `maps_{stream}.nc`'s spatial fields already sit on the FULL flat cell
axis (`n_cells = 721*1440`, row-major `(lat, lon)` -- `eerie_grid.EerieGrid`'s own
convention, see its module docstring), so rendering them is a plain reshape +
`imshow`, no triangulation. `snapshots_{stream}.nc`'s `target`/`prediction` are
genuinely SPARSE (re-drawn every forecast step, ~65k of the grid's ~1.04M cells) so
those are drawn as a point scatter instead; its `source` is a full wet-cell field
but stored as its own sparse `(channel, ipoint_source)` array with separate
`source_lat`/`source_lon` coordinates (not yet reshaped onto the grid) and is
scattered onto the flat cell axis via `eerie_grid.EerieGrid.cell_index` before
reuse of the same grid renderer.

Coastline overlay, axis/frame formatting, per-channel maps-field selection and the
skill-map formula are reused from `fesom_maps.py` (imported, not copied) -- none of
that logic is mesh-specific; only mesh-based rendering (`draw_node_field`/
`draw_elem_field`/`get_mesh`/`get_triangulation`) is FESOM-only and is never
imported here, so this module never triggers a FESOM mesh load or triangulation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm

from paper_eval import eerie_grid
from paper_eval.config import fig_dir
from paper_eval.plotting import fesom_maps, style
from paper_eval.plotting.curves import GEOMETRY_STREAMS, depth_suffixes

_EERIE = GEOMETRY_STREAMS["eerie"]

#: Leads mirroring FESOM f08's own choice of a shallow/near-surface diagnostic lead
#: (which now render every lead in `leads.map_leads`); the EERIE variant still fixes one.
_VARIANCE_LEAD = 30
_EKE_LEAD = 15
_RATIO_LIMIT = 2.0


def _stream_for_channel(channel: str) -> str:
    """EERIE counterpart of `fesom_maps._stream_for_channel`, reusing the same
    `GEOMETRY_STREAMS["eerie"]` table `curves.py`'s score-based figures use --
    channel-name-to-stream knowledge lives in exactly one place."""
    if channel in _EERIE["surface_channels"]:
        return _EERIE["surface_stream"]
    for _, prefix, stream in _EERIE["depth_groups"]:
        if channel.startswith(prefix + "_"):
            return stream
    raise ValueError(f"cannot determine stream for channel {channel!r}")


# ── Grid renderer ───────────────────────────────────────────────────────────────────────
def draw_grid_field(
    ax,
    grid: eerie_grid.EerieGrid,
    values,
    *,
    dry=None,
    cmap=style.SEQUENTIAL_CMAP,
    vmin=None,
    vmax=None,
    norm=None,
    add_colorbar: bool = True,
    cbar_label: str = "",
    shrink: float = 0.75,
    coastline: bool = True,
):
    """Render a full-grid EERIE field (flat, row-major `(lat, lon)` cell axis --
    `eerie_grid.EerieGrid`'s convention, same as `maps_{stream}.nc`'s `ipoint`
    axis) as a raster map via `imshow`.

    `dry`, if given, is a boolean array over cells (True = exclude: `count == 0`
    or land -- there is no fixed wet/dry mesh mask here, see `eerie_grid.py`'s
    module docstring). NaN values (including anything `dry` masks out) render
    transparent by matplotlib's own default colormap `bad`-color behaviour (alpha
    0), so no extra masking call is needed for that -- same convention
    `fesom_maps.draw_node_field`/`draw_elem_field` rely on for their own masked
    triangles.
    """
    n_lat, n_lon = grid.lat.size, grid.lon.size
    field = np.asarray(values, dtype=np.float64).reshape(n_lat, n_lon)
    if dry is not None:
        field = np.where(np.asarray(dry, dtype=bool).reshape(n_lat, n_lon), np.nan, field)

    half = eerie_grid.GRID_RES / 2.0
    extent = (
        grid.lon[0] - half, grid.lon[-1] + half, grid.lat[0] - half, grid.lat[-1] + half,
    )
    kwargs = {
        "cmap": cmap, "interpolation": "nearest", "origin": "lower", "extent": extent,
    }
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"], kwargs["vmax"] = vmin, vmax
    im = ax.imshow(field, **kwargs)
    # An AxesImage is already a raster in vector (PDF) output; set_rasterized is a
    # harmless no-op here, kept only for parity with fesom_maps' explicit call on
    # its (genuinely vector-by-default) tripcolor field layer.
    im.set_rasterized(True)

    fesom_maps._finalize_map_axes(ax)
    if coastline:
        fesom_maps.add_coastline(ax)
    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=shrink, pad=0.02)
        cbar.set_label(cbar_label)
    return im


def draw_scatter_field(
    ax,
    lat,
    lon,
    values,
    *,
    cmap=style.SEQUENTIAL_CMAP,
    vmin=None,
    vmax=None,
    norm=None,
    add_colorbar: bool = True,
    cbar_label: str = "",
    shrink: float = 0.75,
    coastline: bool = True,
    s: float = 0.4,
):
    """Render a SPARSE EERIE point field (e.g. a snapshot's per-lead
    target/prediction/error, re-drawn every forecast step -- see this module's
    docstring) as a small-marker scatter. Same axis/frame/coastline conventions as
    `draw_grid_field`, just a different mark for genuinely sparse data."""
    kwargs = {"cmap": cmap, "s": s, "linewidths": 0}
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"], kwargs["vmax"] = vmin, vmax
    coll = ax.scatter(lon, lat, c=np.asarray(values, dtype=np.float64), **kwargs)
    coll.set_rasterized(True)

    fesom_maps._finalize_map_axes(ax)
    if coastline:
        fesom_maps.add_coastline(ax)
    if add_colorbar:
        cbar = ax.figure.colorbar(coll, ax=ax, shrink=shrink, pad=0.02)
        cbar.set_label(cbar_label)
    return coll


def _save_single_panel(draw_fn, name: str, out_dir, title: str, figsize=(6.0, 5.0)) -> list[Path]:
    """`draw_fn(ax)` draws one field with its own colorbar; wraps it in a
    standalone figure + title + `style.savefig`. The one place that assembles a
    standalone EERIE map/scatter figure (f04/f10's individual-panel outputs) --
    the EERIE counterpart of `fesom_maps.save_single_panel_map`, not copied from
    it since the underlying draw call's signature (grid+dry vs. mesh_path+dry, or
    a scatter's lat/lon) genuinely differs.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, subplot_kw={"aspect": "equal"})
    draw_fn(ax)
    ax.set_title(title)
    return style.savefig(fig, name, out_dir)


# ── f04: snapshot triptych (source / target / prediction / error) ─────────────────────
def build_f04(cfg: dict, cache_paths: dict) -> list[Path]:
    grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])
    out_dir = fig_dir(cfg)
    map_leads = list(cfg["leads"]["map_leads"])
    written: list[Path] = []

    for channel in cfg["figures"]["map_channels"]:
        stream = _stream_for_channel(channel)
        key = f"snapshots_{stream}"
        path = cache_paths.get(key)
        if path is None or not Path(path).exists():
            print(f"[skip] f04 channel={channel!r}: missing cache input {key!r}")
            continue

        ds = xr.open_dataset(path)
        if channel not in ds.channel.values:
            print(f"[skip] f04 channel={channel!r}: not present in {key}")
            continue
        leads = [lead for lead in map_leads if f"target_{lead}" in ds.data_vars]
        if not leads:
            print(f"[skip] f04 channel={channel!r}: none of map_leads present in {key}")
            continue

        # Source (fstep 0): full wet-cell field, denormalized, but stored sparse
        # (its own ipoint_source/source_lat/source_lon) -- scatter it onto the
        # flat grid via cell_index before reusing the grid renderer. Own
        # percentile-based norm (not the target+prediction shared one below): a
        # fstep-0 initial condition is a different field/lead than the forecast
        # snapshots, not part of that pair.
        src_coords = np.stack([ds["source_lat"].values, ds["source_lon"].values], axis=1)
        src_cell_idx = grid.cell_index(src_coords)
        src_vals = ds["source"].sel(channel=channel).values.astype(np.float64)
        source_grid = np.full(grid.n_cells, np.nan, dtype=np.float64)
        source_grid[src_cell_idx] = src_vals
        src_finite = source_grid[np.isfinite(source_grid)]
        vmin_src = float(np.percentile(src_finite, 1)) if src_finite.size else 0.0
        vmax_src = float(np.percentile(src_finite, 99)) if src_finite.size else 1.0

        targets, preds, errors, lat_pts, lon_pts = {}, {}, {}, {}, {}
        for lead in leads:
            t = ds[f"target_{lead}"].sel(channel=channel).values.astype(np.float64)
            p = ds[f"prediction_{lead}"].sel(channel=channel).values.astype(np.float64)
            targets[lead], preds[lead], errors[lead] = t, p, p - t
            lat_pts[lead] = ds[f"lat_{lead}"].values
            lon_pts[lead] = ds[f"lon_{lead}"].values

        # Shared norm rule (same as FESOM f04): target+prediction (sparse, all
        # leads) share one norm; error keeps its own symmetric norm.
        tp_values = np.concatenate(
            [targets[lead] for lead in leads] + [preds[lead] for lead in leads]
        )
        tp_values = tp_values[np.isfinite(tp_values)]
        vmin_tp = float(np.percentile(tp_values, 1)) if tp_values.size else 0.0
        vmax_tp = float(np.percentile(tp_values, 99)) if tp_values.size else 1.0

        err_values = np.concatenate([errors[lead] for lead in leads])
        err_vmin, err_vmax = style.symmetric_limits(err_values, pct=99.0)

        fig, axes = plt.subplots(
            4, len(leads), figsize=(5.2 * len(leads), 15.0), constrained_layout=True,
            subplot_kw={"aspect": "equal"},
        )
        axes = np.asarray(axes).reshape(4, len(leads))

        unit = style.channel_unit(channel)
        label = style.channel_label(channel)
        cbar_label_src = f"{label} [{unit}]" if unit else label
        cbar_label_tp = cbar_label_src
        cbar_label_e = f"error [{unit}]" if unit else "error"
        init_time = str(ds.attrs.get("init_time", ""))[:10]
        sample = ds.attrs.get("sample", "?")

        coll_src = coll_t = coll_e = None
        for j, lead in enumerate(leads):
            # Row 0 -- source is lead-independent (fstep 0): the same field/norm
            # is drawn again in every lead column so the row shares one colorbar
            # and stays visually aligned with the per-lead rows below, rather than
            # inventing a separate single-panel layout just for this one row.
            coll_src = draw_grid_field(
                axes[0, j], grid, source_grid, cmap=style.SEQUENTIAL_CMAP,
                vmin=vmin_src, vmax=vmax_src, add_colorbar=False,
            )
            axes[0, j].set_title(f"lead {lead} d")

            coll_t = draw_scatter_field(
                axes[1, j], lat_pts[lead], lon_pts[lead], targets[lead],
                cmap=style.SEQUENTIAL_CMAP, vmin=vmin_tp, vmax=vmax_tp, add_colorbar=False,
            )
            draw_scatter_field(
                axes[2, j], lat_pts[lead], lon_pts[lead], preds[lead],
                cmap=style.SEQUENTIAL_CMAP, vmin=vmin_tp, vmax=vmax_tp, add_colorbar=False,
            )
            coll_e = draw_scatter_field(
                axes[3, j], lat_pts[lead], lon_pts[lead], errors[lead],
                cmap=style.DIVERGING_CMAP, vmin=err_vmin, vmax=err_vmax, add_colorbar=False,
            )

            # Individual panels (Task 18 convention, extended here): the exact
            # same field/norm as the combined panel above, one file per channel x
            # kind x lead.
            for kind, values, cmap, vmin, vmax, cbar_label in (
                ("target", targets[lead], style.SEQUENTIAL_CMAP, vmin_tp, vmax_tp, cbar_label_tp),
                ("prediction", preds[lead], style.SEQUENTIAL_CMAP, vmin_tp, vmax_tp, cbar_label_tp),
                ("error", errors[lead], style.DIVERGING_CMAP, err_vmin, err_vmax, cbar_label_e),
            ):
                kind_title = "error (pred - target)" if kind == "error" else kind
                written += _save_single_panel(
                    lambda ax, lat_l=lat_pts[lead], lon_l=lon_pts[lead], values=values,
                    cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label: draw_scatter_field(
                        ax, lat_l, lon_l, values, cmap=cmap, vmin=vmin, vmax=vmax,
                        add_colorbar=True, cbar_label=cbar_label,
                    ),
                    f"f04_{channel}_{kind}_{lead}d", out_dir,
                    f"{label} -- {kind_title}, lead {lead} d, init {init_time}",
                )

        # Source has no lead axis of its own -- written once per channel, not per lead.
        written += _save_single_panel(
            lambda ax, source_grid=source_grid, vmin_src=vmin_src, vmax_src=vmax_src,
            cbar_label_src=cbar_label_src: draw_grid_field(
                ax, grid, source_grid, cmap=style.SEQUENTIAL_CMAP, vmin=vmin_src, vmax=vmax_src,
                add_colorbar=True, cbar_label=cbar_label_src,
            ),
            f"f04_{channel}_source", out_dir, f"{label} -- source (init), init {init_time}",
        )

        axes[0, 0].set_ylabel("source (init)")
        axes[1, 0].set_ylabel("target")
        axes[2, 0].set_ylabel("prediction")
        axes[3, 0].set_ylabel("error (pred - target)")

        cbar_src = fig.colorbar(coll_src, ax=axes[0, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_src.set_label(cbar_label_src)
        cbar_tp = fig.colorbar(coll_t, ax=axes[1:3, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_tp.set_label(cbar_label_tp)
        cbar_e = fig.colorbar(coll_e, ax=axes[3, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_e.set_label(cbar_label_e)

        fig.suptitle(f"{label} snapshot  --  sample {sample}, init {init_time}")

        written += style.savefig(fig, f"f04_{channel}_triptych", out_dir)

    return written


# ── f05: bias / RMSE grid maps ─────────────────────────────────────────────────────────
def build_f05(cfg: dict, cache_paths: dict) -> list[Path]:
    grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])
    out_dir = fig_dir(cfg)
    map_leads = list(cfg["leads"]["map_leads"])
    written: list[Path] = []

    for channel in cfg["figures"]["map_channels"]:
        stream = _stream_for_channel(channel)
        key = f"maps_{stream}"
        path = cache_paths.get(key)
        if path is None or not Path(path).exists():
            print(f"[skip] f05 channel={channel!r}: missing cache input {key!r}")
            continue

        ds = xr.open_dataset(path)
        if channel not in ds.channel.values:
            print(f"[skip] f05 channel={channel!r}: not present in {key}")
            continue
        leads = [lead for lead in map_leads if lead in ds.lead.values]
        if not leads:
            print(f"[skip] f05 channel={channel!r}: none of map_leads present in {key}")
            continue

        dry_by_lead = {
            lead: ds["count"].sel(lead=lead, channel=channel).values == 0 for lead in leads
        }
        bias_by_lead = {lead: fesom_maps._map_field(ds, "bias", lead, channel) for lead in leads}
        rmse_by_lead = {lead: fesom_maps._map_field(ds, "rmse", lead, channel) for lead in leads}

        bias_wet = np.concatenate([bias_by_lead[lead][~dry_by_lead[lead]] for lead in leads])
        bias_vmin, bias_vmax = style.symmetric_limits(bias_wet, pct=99.0)
        rmse_wet = np.concatenate([rmse_by_lead[lead][~dry_by_lead[lead]] for lead in leads])
        rmse_vmin, rmse_vmax = style.sequential_limits(rmse_wet, pct=99.0)

        fig, axes = plt.subplots(
            2, len(leads), figsize=(5.2 * len(leads), 8.0), constrained_layout=True,
            subplot_kw={"aspect": "equal"},
        )
        axes = np.asarray(axes).reshape(2, len(leads))

        unit = style.channel_unit(channel)
        label = style.channel_label(channel)

        coll_b = coll_r = None
        for j, lead in enumerate(leads):
            dry = dry_by_lead[lead]
            coll_b = draw_grid_field(
                axes[0, j], grid, bias_by_lead[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=bias_vmin, vmax=bias_vmax, add_colorbar=False,
            )
            axes[0, j].set_title(f"lead {lead} d")
            coll_r = draw_grid_field(
                axes[1, j], grid, rmse_by_lead[lead], dry=dry, cmap=style.SEQUENTIAL_CMAP,
                vmin=rmse_vmin, vmax=rmse_vmax, add_colorbar=False,
            )

        n_samples = ds.attrs.get("n_samples", "?")
        axes[0, 0].set_ylabel(f"mean bias (N={n_samples})")
        axes[1, 0].set_ylabel(f"RMSE (N={n_samples})")

        cbar_b = fig.colorbar(coll_b, ax=axes[0, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_b.set_label(f"bias [{unit}]" if unit else "bias")
        cbar_r = fig.colorbar(coll_r, ax=axes[1, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_r.set_label(f"RMSE [{unit}]" if unit else "RMSE")

        leads_str = "/".join(str(lead) for lead in leads)
        fig.suptitle(
            f"Mean bias and RMSE over N={n_samples} forecasts — {label}, leads {leads_str} d"
        )

        written += style.savefig(fig, f"f05_{channel}_mean_bias_rmse", out_dir)

    return written


# ── f08: variance ratio + EKE (all avg_uoe_X/avg_von_X depth pairs) ───────────────────
def build_f08(cfg: dict, cache_paths: dict) -> list[Path]:
    ocean2d_path = cache_paths.get("maps_OCEAN_2D")
    ocean3d_path = cache_paths.get("maps_OCEAN_3D")
    if ocean2d_path is None or not Path(ocean2d_path).exists():
        print("[skip] f08: missing cache input 'maps_OCEAN_2D'")
        return []
    if ocean3d_path is None or not Path(ocean3d_path).exists():
        print("[skip] f08: missing cache input 'maps_OCEAN_3D'")
        return []

    ds2d = xr.open_dataset(ocean2d_path)
    ds3d = xr.open_dataset(ocean3d_path)
    grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])

    ratio_specs = []  # [(display_name, channel), ...]
    for ch, disp in (("avg_tos", "SST"), ("avg_zos", "SSH")):
        if ch in ds2d.channel.values and _VARIANCE_LEAD in ds2d.lead.values:
            ratio_specs.append((disp, ch))
        else:
            print(f"[skip] f08: {ch} @ lead {_VARIANCE_LEAD} not available in maps_OCEAN_2D")

    # EKE pairs: every depth where BOTH avg_uoe_X and avg_von_X exist (includes
    # the "surf" level -- it pairs like any other depth, no special exclusion).
    # `depth_suffixes` already returns its list depth-ascending (shallowest --
    # including "surf" at 0 m -- first), so filtering `u`'s list against `v`'s set
    # keeps that order rather than re-sorting suffix strings lexicographically
    # (which would misorder e.g. "12.5m" before "2.5m").
    if _EKE_LEAD in ds3d.lead.values:
        u_suffixes = depth_suffixes(ds3d.channel.values, "avg_uoe")
        v_suffixes = set(depth_suffixes(ds3d.channel.values, "avg_von"))
        eke_suffixes = [s for s in u_suffixes if s in v_suffixes]
    else:
        eke_suffixes = []
    if not eke_suffixes:
        print(
            f"[skip] f08: no avg_uoe_X/avg_von_X pair available at lead {_EKE_LEAD} "
            "in maps_OCEAN_3D"
        )

    if not ratio_specs and not eke_suffixes:
        print("[skip] f08: nothing to plot (no variance-ratio or EKE inputs available)")
        return []

    n_rows = (1 if ratio_specs else 0) + len(eke_suffixes)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(17.0, 5.0 * n_rows), constrained_layout=True,
        subplot_kw={"aspect": "equal"}, squeeze=False,
    )

    row = 0
    if ratio_specs:
        for col, (disp, ch) in enumerate(ratio_specs):
            count = ds2d["count"].sel(lead=_VARIANCE_LEAD, channel=ch).values
            dry = count == 0
            pred_var = ds2d["pred_var"].sel(
                lead=_VARIANCE_LEAD, channel=ch
            ).values.astype(np.float64)
            target_var = ds2d["target_var"].sel(
                lead=_VARIANCE_LEAD, channel=ch
            ).values.astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.log2(pred_var / target_var)
            draw_grid_field(
                axes[row, col], grid, ratio, dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=-_RATIO_LIMIT, vmax=_RATIO_LIMIT, cbar_label="log$_2$(pred_var / target_var)",
            )
            axes[row, col].set_title(f"{disp} variance ratio (lead {_VARIANCE_LEAD} d)")
        for col in range(len(ratio_specs), 3):
            axes[row, col].set_axis_off()
        row += 1

    for suffix in eke_suffixes:
        u_ch, v_ch = f"avg_uoe_{suffix}", f"avg_von_{suffix}"
        depth_disp = "surface" if suffix == "surf" else suffix

        count_u = ds3d["count"].sel(lead=_EKE_LEAD, channel=u_ch).values
        count_v = ds3d["count"].sel(lead=_EKE_LEAD, channel=v_ch).values
        dry = (count_u == 0) | (count_v == 0)

        pv_u = ds3d["pred_var"].sel(lead=_EKE_LEAD, channel=u_ch).values.astype(np.float64)
        pv_v = ds3d["pred_var"].sel(lead=_EKE_LEAD, channel=v_ch).values.astype(np.float64)
        tv_u = ds3d["target_var"].sel(lead=_EKE_LEAD, channel=u_ch).values.astype(np.float64)
        tv_v = ds3d["target_var"].sel(lead=_EKE_LEAD, channel=v_ch).values.astype(np.float64)
        eke_pred = 0.5 * (pv_u + pv_v)
        eke_target = 0.5 * (tv_u + tv_v)
        with np.errstate(invalid="ignore", divide="ignore"):
            eke_ratio = np.log2(eke_pred / eke_target)

        # LogNorm needs strictly positive values; anything <= 0 is excluded via the
        # wet mask, not clipped to a small positive number (same rule as FESOM f08).
        eke_wet_vals = np.concatenate([eke_target[~dry], eke_pred[~dry]])
        eke_wet_vals = eke_wet_vals[np.isfinite(eke_wet_vals) & (eke_wet_vals > 0)]
        if eke_wet_vals.size:
            eke_vmin = max(float(np.percentile(eke_wet_vals, 1)), 1e-10)
            eke_vmax = float(np.percentile(eke_wet_vals, 99))
            eke_vmax = max(eke_vmax, eke_vmin * 10)
        else:
            eke_vmin, eke_vmax = 1e-6, 1e-2
        eke_norm = LogNorm(vmin=eke_vmin, vmax=eke_vmax)

        draw_grid_field(
            axes[row, 0], grid, eke_target, dry=dry, cmap=style.SEQUENTIAL_CMAP, norm=eke_norm,
            cbar_label="EKE [m$^2$/s$^2$]",
        )
        axes[row, 0].set_title(f"Target EKE @ {depth_disp} (lead {_EKE_LEAD} d)")

        draw_grid_field(
            axes[row, 1], grid, eke_pred, dry=dry, cmap=style.SEQUENTIAL_CMAP, norm=eke_norm,
            cbar_label="EKE [m$^2$/s$^2$]",
        )
        axes[row, 1].set_title(f"Prediction EKE @ {depth_disp} (lead {_EKE_LEAD} d)")

        draw_grid_field(
            axes[row, 2], grid, eke_ratio, dry=dry, cmap=style.DIVERGING_CMAP,
            vmin=-_RATIO_LIMIT, vmax=_RATIO_LIMIT, cbar_label="log$_2$(EKE pred / EKE target)",
        )
        axes[row, 2].set_title(f"EKE ratio @ {depth_disp} (lead {_EKE_LEAD} d)")
        row += 1

    n_samples = ds3d.attrs.get("n_samples", ds2d.attrs.get("n_samples", "?"))
    fig.suptitle(f"Variability diagnostics  --  n_samples={n_samples}")

    return style.savefig(fig, "f08_variability", fig_dir(cfg))


# ── f10: skill (vs persistence / vs climatology) + ACC grid maps ──────────────────────
def build_f10(cfg: dict, cache_paths: dict) -> list[Path]:
    grid = eerie_grid.load_grid(cfg["truth"]["OCEAN_2D"])
    out_dir = fig_dir(cfg)
    map_leads = list(cfg["leads"]["map_leads"])
    written: list[Path] = []

    for channel in cfg["figures"]["map_channels"]:
        stream = _stream_for_channel(channel)
        key = f"maps_{stream}"
        path = cache_paths.get(key)
        if path is None or not Path(path).exists():
            print(f"[skip] f10 channel={channel!r}: missing cache input {key!r}")
            continue

        ds = xr.open_dataset(path)
        missing = [v for v in fesom_maps._SKILL_VARS if v not in ds.data_vars]
        if missing:
            print(f"[skip] f10 channel={channel!r}: {key} lacks {missing} (pre-Task-14 cache)")
            continue
        if channel not in ds.channel.values:
            print(f"[skip] f10 channel={channel!r}: not present in {key}")
            continue
        leads = [lead for lead in map_leads if lead in ds.lead.values]
        if not leads:
            print(f"[skip] f10 channel={channel!r}: none of map_leads present in {key}")
            continue

        dry_by_lead = {
            lead: ds["count"].sel(lead=lead, channel=channel).values == 0 for lead in leads
        }
        skill_p_by_lead, skill_c_by_lead, acc_by_lead = {}, {}, {}
        for lead in leads:
            rmse = fesom_maps._map_field(ds, "rmse", lead, channel)
            rmse_p = fesom_maps._map_field(ds, "rmse_persistence", lead, channel)
            rmse_c = fesom_maps._map_field(ds, "rmse_climatology", lead, channel)
            skill_p_by_lead[lead] = fesom_maps._skill(rmse, rmse_p)
            skill_c_by_lead[lead] = fesom_maps._skill(rmse, rmse_c)
            acc_by_lead[lead] = fesom_maps._map_field(ds, "acc", lead, channel)

        fig, axes = plt.subplots(
            3, len(leads), figsize=(5.2 * len(leads), 12.0), constrained_layout=True,
            subplot_kw={"aspect": "equal"},
        )
        axes = np.asarray(axes).reshape(3, len(leads))

        label = style.channel_label(channel)
        cbar_label_p = "skill vs persistence (clipped at -1)"
        cbar_label_c = "skill vs climatology (clipped at -1)"
        cbar_label_a = "ACC (clipped at 0)"
        n_samples = ds.attrs.get("n_samples", "?")

        coll_p = coll_c = coll_a = None
        for j, lead in enumerate(leads):
            dry = dry_by_lead[lead]
            coll_p = draw_grid_field(
                axes[0, j], grid, skill_p_by_lead[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=-1.0, vmax=1.0, add_colorbar=False,
            )
            axes[0, j].set_title(f"lead {lead} d")
            coll_c = draw_grid_field(
                axes[1, j], grid, skill_c_by_lead[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=-1.0, vmax=1.0, add_colorbar=False,
            )
            coll_a = draw_grid_field(
                axes[2, j], grid, acc_by_lead[lead], dry=dry, cmap=style.SEQUENTIAL_CMAP,
                vmin=0.0, vmax=1.0, add_colorbar=False,
            )

            # Individual panels (Task 18 convention): same field/norm/label as the
            # combined row above, one file per channel x kind x lead.
            for kind, values, cmap, vmin, vmax, cbar_label, kind_title in (
                (
                    "skill_pers", skill_p_by_lead[lead], style.DIVERGING_CMAP, -1.0, 1.0,
                    cbar_label_p, "skill vs persistence",
                ),
                (
                    "skill_clim", skill_c_by_lead[lead], style.DIVERGING_CMAP, -1.0, 1.0,
                    cbar_label_c, "skill vs climatology",
                ),
                ("acc", acc_by_lead[lead], style.SEQUENTIAL_CMAP, 0.0, 1.0, cbar_label_a, "ACC"),
            ):
                written += _save_single_panel(
                    lambda ax, values=values, dry=dry, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar_label=cbar_label: draw_grid_field(
                        ax, grid, values, dry=dry, cmap=cmap, vmin=vmin, vmax=vmax,
                        add_colorbar=True, cbar_label=cbar_label,
                    ),
                    f"f10_{channel}_{kind}_{lead}d", out_dir,
                    f"{label} -- {kind_title}, lead {lead} d (N={n_samples})",
                )

        axes[0, 0].set_ylabel("skill vs persistence")
        axes[1, 0].set_ylabel("skill vs climatology")
        axes[2, 0].set_ylabel("ACC")

        cbar_p = fig.colorbar(coll_p, ax=axes[0, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_p.set_label(cbar_label_p)
        cbar_c = fig.colorbar(coll_c, ax=axes[1, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_c.set_label(cbar_label_c)
        cbar_a = fig.colorbar(coll_a, ax=axes[2, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_a.set_label(cbar_label_a)

        leads_str = "/".join(str(lead) for lead in leads)
        fig.suptitle(
            f"Skill vs. baselines and ACC over N={n_samples} forecasts — "
            f"{label}, leads {leads_str} d"
        )

        written += style.savefig(fig, f"f10_{channel}_skill_acc", out_dir)

    return written
