"""Native-mesh map rendering (CORE2 triangulation) + the f04/f05/f10 map figures.

Mechanics adapted from `examples/plot_fesom_sample.py` (not imported): node fields
use `tripcolor(..., shading="gouraud")`, element fields use `tripcolor(...,
facecolors=...)`, dateline-spanning triangles are masked, coastlines come from the
offline cartopy/Natural-Earth assets shipped in the repo.

Mesh and triangulation are expensive to build (mesh load ~5s) and are reused by
every map figure and every channel/lead panel within them, so both are cached here
per `mesh_path`, module-wide.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.tri import Triangulation

from paper_eval.config import fig_dir
from paper_eval.meshtools import Mesh, load_mesh, masks_and_weights, triangulation
from paper_eval.plotting import style

#: Natural Earth shapefiles shipped with the repo (only 110m resolution present).
CARTOPY_ASSETS = "/work/ab0995/a270225/WeatherGenerator2/assets/cartopy"

# ── Module-wide mesh/triangulation cache, keyed by mesh_path ───────────────────────────
_MESH_CACHE: dict[str, Mesh] = {}
_TRI_CACHE: dict[str, Triangulation] = {}
_DATELINE_MASK_CACHE: dict[str, np.ndarray] = {}


def get_mesh(mesh_path) -> Mesh:
    """Module-cached `Mesh` for `mesh_path` (loaded once, reused across figures)."""
    key = str(mesh_path)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = load_mesh(mesh_path)
    return _MESH_CACHE[key]


def get_triangulation(mesh_path) -> Triangulation:
    """Module-cached, read-only reference `Triangulation` for `mesh_path`,
    dateline-masked.

    This exact object must never be mutated or handed to `tripcolor` directly:
    matplotlib's `TriMesh`/gouraud collection keeps a *live* reference to whatever
    `Triangulation` it was built from and re-reads its mask at render/savefig time
    (not at construction time), so if two panels drew from this one shared,
    repeatedly-`set_mask`-ed object, every panel in a figure would render with
    whichever mask was set *last*, not its own. Draw calls instead build their own
    fresh `Triangulation` from the same cached `lon`/`lat`/`tri_elem` arrays (see
    `_fresh_triangulation`) -- cheap (no recomputation), but never aliased. Use
    `_dateline_mask` for the cached dateline-only mask array.
    """
    key = str(mesh_path)
    if key not in _TRI_CACHE:
        mesh = get_mesh(mesh_path)
        tri = triangulation(mesh)
        _TRI_CACHE[key] = tri
        dateline = tri.mask
        if dateline is None:
            dateline = np.zeros(mesh.tri_elem.shape[0], dtype=bool)
        _DATELINE_MASK_CACHE[key] = dateline.copy()
    return _TRI_CACHE[key]


def _dateline_mask(mesh_path) -> np.ndarray:
    key = str(mesh_path)
    if key not in _DATELINE_MASK_CACHE:
        get_triangulation(mesh_path)
    return _DATELINE_MASK_CACHE[key]


def _fresh_triangulation(mesh_path, extra_mask: np.ndarray) -> Triangulation:
    """A brand-new `Triangulation` (own mask, safe to hand to `tripcolor`),
    built from the cached mesh's arrays (no recomputation) with
    `extra_mask | dateline_mask` set on it.

    Never returns/mutates the module-cached `get_triangulation` object -- see its
    docstring for why sharing one `Triangulation` instance across panels is unsafe
    for gouraud/tripcolor rendering.
    """
    mesh = get_mesh(mesh_path)
    tri = Triangulation(mesh.lon, mesh.lat, mesh.tri_elem)
    tri.set_mask(_dateline_mask(mesh_path) | extra_mask)
    return tri


#: Parsed coastline polylines, cached module-wide (dozens of panels reuse the same
#: handful of shapefile geometries -- reparsing per panel would dominate runtime).
_COASTLINE_CACHE: list[np.ndarray] | None = None


def _load_coastline_lines() -> list[np.ndarray]:
    global _COASTLINE_CACHE
    if _COASTLINE_CACHE is None:
        import cartopy
        import cartopy.io.shapereader as shpreader

        if os.path.isdir(CARTOPY_ASSETS):
            cartopy.config["pre_existing_data_dir"] = CARTOPY_ASSETS
        path = shpreader.natural_earth(resolution="110m", category="physical", name="coastline")
        lines = []
        for geom in shpreader.Reader(path).geometries():
            for line in getattr(geom, "geoms", [geom]):
                lines.append(np.array(line.coords))
        _COASTLINE_CACHE = lines
    return _COASTLINE_CACHE


def add_coastline(ax) -> None:
    """Draw coastlines from the offline cartopy/Natural-Earth assets, if present."""
    try:
        for xy in _load_coastline_lines():
            ax.plot(xy[:, 0], xy[:, 1], color="0.25", linewidth=0.5, zorder=3)
    except Exception as exc:  # offline data missing -> the field still plots fine
        print(f"coastlines unavailable ({type(exc).__name__}); skipping")


def _finalize_map_axes(ax) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.tick_params(labelsize=7)
    # Every axes gets a solid frame by rcParams default now; keep the explicit call
    # so maps stay closed-framed even if that default ever changes.
    style.full_frame(ax)


def draw_node_field(
    ax,
    mesh_path,
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
    """Render a per-node field with gouraud-shaded `tripcolor`.

    `dry`, if given, is a boolean array over nodes (True = exclude: dry or
    invalid). Any triangle touching an excluded (dry or non-finite) vertex is
    masked out entirely, so no dry/NaN value is ever interpolated into a drawn
    triangle -- the wet mask, not the data, is the sole exclusion authority.
    """
    mesh = get_mesh(mesh_path)

    values = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(values)
    if dry is not None:
        bad = bad | np.asarray(dry, dtype=bool)
    values_safe = np.where(bad, 0.0, values)  # filler; only feeds masked-out triangles

    tri_bad = bad[mesh.tri_elem].any(axis=1)
    # A fresh Triangulation per call -- see `_fresh_triangulation`'s docstring for
    # why the shared, module-cached one must never be mutated and reused here.
    tri = _fresh_triangulation(mesh_path, tri_bad)

    kwargs = {"cmap": cmap}
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"], kwargs["vmax"] = vmin, vmax
    coll = ax.tripcolor(tri, values_safe, shading="gouraud", **kwargs)
    # ~127k-node / 244k-triangle CORE2 mesh as vector paths makes multi-tens-of-MB
    # PDFs; rasterize the field layer (at savefig's 300 dpi) and leave axes/text/
    # colorbar/coastline as vector, for a paper-usable file size.
    coll.set_rasterized(True)

    _finalize_map_axes(ax)
    if coastline:
        add_coastline(ax)
    if add_colorbar:
        cbar = ax.figure.colorbar(coll, ax=ax, shrink=shrink, pad=0.02)
        cbar.set_label(cbar_label)
        # Outline left at its default: visible, solid, `axes.edgecolor` (black) --
        # every colorbar family (maps, heatmaps, scorecards) now shows a solid black
        # frame, matching every axes' own spines (see style.py rcParams).
    return coll


def draw_elem_field(
    ax,
    mesh_path,
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
    """Render a per-element field with flat-shaded `tripcolor(facecolors=...)`.

    `dry`, if given, is a boolean array over elements (True = exclude). Excluded
    (dry or non-finite) elements are masked out of the facecolor array, so they
    render as no fill regardless of their stored value.
    """
    mesh = get_mesh(mesh_path)

    values = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(values)
    if dry is not None:
        bad = bad | np.asarray(dry, dtype=bool)
    facecolors = np.ma.masked_where(bad, values)

    # Dryness here is carried entirely by the masked `facecolors` array; the
    # triangulation itself only needs the (per-mesh, not per-call) dateline mask --
    # still a *fresh* instance per call, never the shared cached one (see
    # `_fresh_triangulation`).
    no_extra_mask = np.zeros(mesh.tri_elem.shape[0], dtype=bool)
    tri = _fresh_triangulation(mesh_path, no_extra_mask)

    kwargs = {"cmap": cmap}
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"], kwargs["vmax"] = vmin, vmax
    coll = ax.tripcolor(tri, facecolors=facecolors, **kwargs)
    # See draw_node_field: rasterize the (huge) field layer, keep the rest vector.
    coll.set_rasterized(True)

    _finalize_map_axes(ax)
    if coastline:
        add_coastline(ax)
    if add_colorbar:
        cbar = ax.figure.colorbar(coll, ax=ax, shrink=shrink, pad=0.02)
        cbar.set_label(cbar_label)
        # Outline left at its default: visible, solid, `axes.edgecolor` (black) --
        # every colorbar family (maps, heatmaps, scorecards) now shows a solid black
        # frame, matching every axes' own spines (see style.py rcParams).
    return coll


def _map_field(ds: xr.Dataset, name: str, lead: int, channel: str) -> np.ndarray:
    return ds[name].sel(lead=lead, channel=channel).values.astype(np.float64)


def _stream_for_channel(channel: str) -> str:
    if channel in ("sst", "sss", "ssh") or channel.startswith(("temp_", "salt_")):
        return "FESOM_NODE"
    if channel.startswith(("u_", "v_")):
        return "FESOM_ELEM"
    raise ValueError(f"cannot determine stream for channel {channel!r}")


def save_single_panel_map(
    mesh_path,
    draw_field,
    values,
    dry,
    *,
    cmap,
    vmin: float,
    vmax: float,
    cbar_label: str,
    title: str,
    name: str,
    out_dir,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> list[Path]:
    """One field, one axes, its own colorbar and title -- written as `name`.pdf/.png.

    Shared by every map family (f04, f10) that also draws the exact same field
    into a combined multi-panel figure: callers pass the identical `draw_field`
    (`draw_node_field`/`draw_elem_field`), `values`/`dry`, and `cmap`/`vmin`/`vmax`
    used for that field's panel in the combined figure, so the individual file is
    pixel-for-pixel the same map (same norm, same rasterized mesh layer, same
    frame/coastline handling from `draw_field` itself) -- only the figure
    wrapper (own colorbar instead of a shared one, own title) differs. This is the
    one place that assembles a standalone map figure; combined builders never
    duplicate this wrapper.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, subplot_kw={"aspect": "equal"})
    draw_field(
        ax, mesh_path, values, dry=dry, cmap=cmap, vmin=vmin, vmax=vmax,
        add_colorbar=True, cbar_label=cbar_label,
    )
    ax.set_title(title)
    return style.savefig(fig, name, out_dir)


# ── f04: snapshot triptychs (target / prediction / error) ─────────────────────────────
def build_f04(cfg: dict, cache_paths: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    mesh_path = cfg["mesh_path"]
    mesh = get_mesh(mesh_path)
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
        leads = [lead for lead in map_leads if lead in ds.lead.values]
        if not leads:
            print(f"[skip] f04 channel={channel!r}: none of map_leads present in {key}")
            continue

        mask, _ = masks_and_weights(mesh, stream, [channel])
        wet = mask[0]
        dry = ~wet
        draw_field = draw_node_field if stream == "FESOM_NODE" else draw_elem_field

        targets, preds, errors = {}, {}, {}
        for lead in leads:
            t = ds["target"].sel(lead=lead, channel=channel).values.astype(np.float64)
            p = ds["prediction"].sel(lead=lead, channel=channel).values.astype(np.float64)
            targets[lead], preds[lead], errors[lead] = t, p, p - t

        tp_parts = [targets[lead][wet] for lead in leads] + [preds[lead][wet] for lead in leads]
        tp_values = np.concatenate(tp_parts)
        tp_values = tp_values[np.isfinite(tp_values)]
        vmin_tp = float(np.percentile(tp_values, 1)) if tp_values.size else 0.0
        vmax_tp = float(np.percentile(tp_values, 99)) if tp_values.size else 1.0

        err_values = np.concatenate([errors[lead][wet] for lead in leads])
        err_vmin, err_vmax = style.symmetric_limits(err_values, pct=99.0)

        fig, axes = plt.subplots(
            3, len(leads), figsize=(5.2 * len(leads), 11.5), constrained_layout=True,
            subplot_kw={"aspect": "equal"},
        )
        axes = np.asarray(axes).reshape(3, len(leads))

        unit = style.channel_unit(channel)
        label = style.channel_label(channel, mesh.nz1)
        cbar_label_tp = f"{label} [{unit}]" if unit else label
        cbar_label_e = f"error [{unit}]" if unit else "error"
        init_time = str(ds.attrs.get("init_time", ""))[:10]
        sample = ds.attrs.get("sample", "?")

        coll_t = coll_e = None
        for j, lead in enumerate(leads):
            coll_t = draw_field(
                axes[0, j], mesh_path, targets[lead], dry=dry, cmap=style.SEQUENTIAL_CMAP,
                vmin=vmin_tp, vmax=vmax_tp, add_colorbar=False,
            )
            axes[0, j].set_title(f"lead {lead} d")
            draw_field(
                axes[1, j], mesh_path, preds[lead], dry=dry, cmap=style.SEQUENTIAL_CMAP,
                vmin=vmin_tp, vmax=vmax_tp, add_colorbar=False,
            )
            coll_e = draw_field(
                axes[2, j], mesh_path, errors[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=err_vmin, vmax=err_vmax, add_colorbar=False,
            )

            # Individual panels (Task 18): the exact same field/norm as the combined
            # panel above, drawn again via `save_single_panel_map` -- its own
            # colorbar/title, one file per channel x kind x lead, written in this
            # same pass over the data (no re-opening the cache, no recomputed norms).
            for kind, values, cmap, vmin, vmax, cbar_label in (
                ("target", targets[lead], style.SEQUENTIAL_CMAP, vmin_tp, vmax_tp, cbar_label_tp),
                ("prediction", preds[lead], style.SEQUENTIAL_CMAP, vmin_tp, vmax_tp, cbar_label_tp),
                ("error", errors[lead], style.DIVERGING_CMAP, err_vmin, err_vmax, cbar_label_e),
            ):
                kind_title = "error (pred - target)" if kind == "error" else kind
                written += save_single_panel_map(
                    mesh_path, draw_field, values, dry,
                    cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label,
                    title=f"{label} -- {kind_title}, lead {lead} d, init {init_time}",
                    name=f"f04_{channel}_{kind}_{lead}d", out_dir=out_dir,
                )

        axes[0, 0].set_ylabel("target")
        axes[1, 0].set_ylabel("prediction")
        axes[2, 0].set_ylabel("error (pred - target)")

        cbar_tp = fig.colorbar(coll_t, ax=axes[0:2, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_tp.set_label(cbar_label_tp)
        cbar_e = fig.colorbar(coll_e, ax=axes[2, :].ravel().tolist(), shrink=0.7, pad=0.015)
        cbar_e.set_label(cbar_label_e)

        fig.suptitle(f"{label} snapshot triptych  --  sample {sample}, init {init_time}")

        written += style.savefig(fig, f"f04_{channel}_triptych", out_dir)

    return written


# ── f05: bias / RMSE maps ──────────────────────────────────────────────────────────────
def build_f05(cfg: dict, cache_paths: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    mesh_path = cfg["mesh_path"]
    mesh = get_mesh(mesh_path)
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

        mask, _ = masks_and_weights(mesh, stream, [channel])
        wet = mask[0]
        dry = ~wet
        draw_field = draw_node_field if stream == "FESOM_NODE" else draw_elem_field

        bias_by_lead = {lead: _map_field(ds, "bias", lead, channel) for lead in leads}
        rmse_by_lead = {lead: _map_field(ds, "rmse", lead, channel) for lead in leads}

        bias_all = np.concatenate([bias_by_lead[lead][wet] for lead in leads])
        bias_vmin, bias_vmax = style.symmetric_limits(bias_all, pct=99.0)
        rmse_all = np.concatenate([rmse_by_lead[lead][wet] for lead in leads])
        rmse_vmin, rmse_vmax = style.sequential_limits(rmse_all, pct=99.0)

        fig, axes = plt.subplots(
            2, len(leads), figsize=(5.2 * len(leads), 8.0), constrained_layout=True,
            subplot_kw={"aspect": "equal"},
        )
        axes = np.asarray(axes).reshape(2, len(leads))

        unit = style.channel_unit(channel)
        label = style.channel_label(channel, mesh.nz1)

        coll_b = coll_r = None
        for j, lead in enumerate(leads):
            coll_b = draw_field(
                axes[0, j], mesh_path, bias_by_lead[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=bias_vmin, vmax=bias_vmax, add_colorbar=False,
            )
            axes[0, j].set_title(f"lead {lead} d")
            coll_r = draw_field(
                axes[1, j], mesh_path, rmse_by_lead[lead], dry=dry, cmap=style.SEQUENTIAL_CMAP,
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


# ── f10: skill (vs persistence / vs climatology) + ACC maps ───────────────────────────
#: New maps-dataset variables Task 14 added (rmse_persistence, rmse_climatology, acc);
#: absent from any pre-Task-14 cache, in which case f10 [skip]s like every other family
#: whose cache input is missing.
_SKILL_VARS = ("rmse_persistence", "rmse_climatology", "acc")


def _skill(rmse: np.ndarray, rmse_baseline: np.ndarray) -> np.ndarray:
    """`1 - rmse / rmse_baseline`, computed in f64 from the (already f64-cast, via
    `_map_field`) f32 maps-cache inputs.

    A baseline rmse of 0 or NaN (a degenerate/dry point) produces 0/0 (-> NaN) or a
    nonzero/0 (-> +/-inf); `errstate` silences the warning for both, and the result is
    explicitly normalized to NaN in either case (`np.isfinite` excludes both NaN and
    inf, but making the degenerate value NaN -- not +/-inf -- is what the brief calls
    for, and is the safer contract for any future non-plotting consumer of this
    function). NaN is then masked out downstream exactly like every other dry/invalid
    map value (see `draw_node_field`/`draw_elem_field`'s `dry` handling) -- never
    plotted as if it were a real skill.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        skill = 1.0 - rmse / rmse_baseline
    return np.where(np.isfinite(skill), skill, np.nan)


def build_f10(cfg: dict, cache_paths: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    mesh_path = cfg["mesh_path"]
    mesh = get_mesh(mesh_path)
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
        missing = [v for v in _SKILL_VARS if v not in ds.data_vars]
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

        mask, _ = masks_and_weights(mesh, stream, [channel])
        wet = mask[0]
        dry = ~wet
        draw_field = draw_node_field if stream == "FESOM_NODE" else draw_elem_field

        skill_p_by_lead, skill_c_by_lead, acc_by_lead = {}, {}, {}
        for lead in leads:
            rmse = _map_field(ds, "rmse", lead, channel)
            rmse_p = _map_field(ds, "rmse_persistence", lead, channel)
            rmse_c = _map_field(ds, "rmse_climatology", lead, channel)
            skill_p_by_lead[lead] = _skill(rmse, rmse_p)
            skill_c_by_lead[lead] = _skill(rmse, rmse_c)
            acc_by_lead[lead] = _map_field(ds, "acc", lead, channel)

        fig, axes = plt.subplots(
            3, len(leads), figsize=(5.2 * len(leads), 12.0), constrained_layout=True,
            subplot_kw={"aspect": "equal"},
        )
        axes = np.asarray(axes).reshape(3, len(leads))

        label = style.channel_label(channel, mesh.nz1)
        cbar_label_p = "skill vs persistence (clipped at -1)"
        cbar_label_c = "skill vs climatology (clipped at -1)"
        cbar_label_a = "ACC (clipped at 0)"
        n_samples = ds.attrs.get("n_samples", "?")

        coll_p = coll_c = coll_a = None
        for j, lead in enumerate(leads):
            coll_p = draw_field(
                axes[0, j], mesh_path, skill_p_by_lead[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=-1.0, vmax=1.0, add_colorbar=False,
            )
            axes[0, j].set_title(f"lead {lead} d")
            coll_c = draw_field(
                axes[1, j], mesh_path, skill_c_by_lead[lead], dry=dry, cmap=style.DIVERGING_CMAP,
                vmin=-1.0, vmax=1.0, add_colorbar=False,
            )
            coll_a = draw_field(
                axes[2, j], mesh_path, acc_by_lead[lead], dry=dry, cmap=style.SEQUENTIAL_CMAP,
                vmin=0.0, vmax=1.0, add_colorbar=False,
            )

            # Individual panels (Task 18): same field/norm/label as the combined
            # row above, one file per channel x kind x lead, written in this same
            # pass over the data (see `save_single_panel_map`'s docstring).
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
                written += save_single_panel_map(
                    mesh_path, draw_field, values, dry,
                    cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label,
                    title=f"{label} -- {kind_title}, lead {lead} d (N={n_samples})",
                    name=f"f10_{channel}_{kind}_{lead}d", out_dir=out_dir,
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
