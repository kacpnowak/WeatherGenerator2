"""f08/f12: one-row map diagnostics on the native mesh, rendered at every map lead.

f08 -- variability: log2 of the forecast/reference variance ratio for SST, SSH and
surface EKE. f12 -- accuracy: pointwise skill against the day-of-year climatology for
the same three fields, in the same layout and on the same shared colour bar, so the two
read together.

EKE (plan D5): `0.5 * (var(u_d) + var(v_d))` from `pred_var`/`target_var` at a depth
where both u and v exist; `u_2`/`v_2` (the shallowest level) by default, per the brief.

Both families emit one single-row figure per lead (`..._15d`, `..._30d`) plus a stacked
figure carrying every lead as a row under the unsuffixed name -- the name the manuscript
includes, so the paper figure is the lead-to-lead comparison. Leads come from
`leads.map_leads`, restricted to what the map caches actually hold.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paper_eval.config import fig_dir
from paper_eval.meshtools import masks_and_weights
from paper_eval.plotting import style
from paper_eval.plotting.fesom_maps import (
    _skill,
    draw_elem_field,
    draw_node_field,
    get_mesh,
)

#: Used only when a config carries no `leads.map_leads` at all; every shipped config does.
_FALLBACK_LEAD = 15
_EKE_CHANNELS = ("u_2", "v_2")
_RATIO_LIMIT = 2.0


@dataclass(frozen=True)
class _Panel:
    """One map in the row: its column title, which mesh it lives on, and its field."""

    title: str
    stream: str  # "FESOM_NODE" (per-node, gouraud) | "FESOM_ELEM" (per-element, flat)
    values: np.ndarray
    dry: np.ndarray


def _lead_list(cfg: dict, *datasets: xr.Dataset) -> list[int]:
    """`map_leads` from the config, kept only where *every* map cache carries the lead."""
    requested = list(cfg.get("leads", {}).get("map_leads") or [_FALLBACK_LEAD])
    available = set.intersection(
        *(set(np.asarray(ds.lead.values).tolist()) for ds in datasets)
    )
    return [int(lead) for lead in requested if int(lead) in available]


def _leads_phrase(leads: list[int]) -> str:
    """"15-day lead" / "15- and 30-day leads" / "15-, 30- and 45-day leads"."""
    if len(leads) == 1:
        return f"{leads[0]}-day lead"
    return ", ".join(f"{lead}-" for lead in leads[:-1]) + f" and {leads[-1]}-day leads"


def _draw_grid(mesh_path, rows, *, vmin, vmax, cbar_label, suptitle):
    """Render `rows` -- a list of `(lead, [_Panel, ...])`, one row per lead -- as a grid.

    Every panel shares one symmetric scale and therefore one colourbar; column titles go
    on the top row only, and the lead is named on the left of each row when there is more
    than one (a single-row figure names its lead in the suptitle instead).
    """
    n_rows, n_cols = len(rows), len(rows[0][1])
    # A 2:1 world map in a 16in/3-column grid is only ~2.5in tall, so a stacked figure
    # given the single-row height (4.4in, kept as-is so one-lead figures are unchanged)
    # spends most of the added height on the gap between rows. Budget per row from the
    # map instead, plus a fixed allowance for the suptitle and the shared colourbar.
    height = 4.4 if n_rows == 1 else 3.4 * n_rows + 1.3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16.0, height), constrained_layout=True,
        subplot_kw={"aspect": "equal"},
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    coll = None
    for i, (lead, panels) in enumerate(rows):
        for j, panel in enumerate(panels):
            draw = draw_node_field if panel.stream == "FESOM_NODE" else draw_elem_field
            drawn = draw(
                axes[i, j], mesh_path, panel.values, dry=panel.dry,
                cmap=style.DIVERGING_CMAP, vmin=vmin, vmax=vmax, add_colorbar=False,
            )
            if coll is None:
                coll = drawn
            if i == 0:
                axes[i, j].set_title(panel.title)
        if n_rows > 1:
            axes[i, 0].set_ylabel(f"lead {lead} d")

    # One bar for the whole grid: identical scales everywhere, and repeating it per row
    # would eat the width the maps need. `fraction` is per-axes, so divide by the row
    # count to keep the bar the same absolute thickness however tall the figure grows.
    cbar = fig.colorbar(
        coll, ax=axes.ravel().tolist(), orientation="horizontal",
        fraction=0.055 / n_rows, pad=0.06, shrink=0.55,
    )
    cbar.set_label(cbar_label)
    fig.suptitle(suptitle)
    return fig


def _emit(cfg, rows, *, vmin, vmax, cbar_label, title_of, stem) -> list[Path]:
    """Write the stacked all-leads figure under `stem`, plus one `stem_<lead>d` per lead.

    The unsuffixed name is always written -- it is what the manuscript includes -- and is
    the multi-lead comparison whenever more than one lead is available.
    """
    mesh_path = cfg["mesh_path"]
    out_dir = fig_dir(cfg)
    leads = [lead for lead, _ in rows]

    fig = _draw_grid(
        mesh_path, rows, vmin=vmin, vmax=vmax, cbar_label=cbar_label,
        suptitle=title_of(leads),
    )
    written = style.savefig(fig, stem, out_dir)

    if len(rows) > 1:
        for lead, panels in rows:
            fig = _draw_grid(
                mesh_path, [(lead, panels)], vmin=vmin, vmax=vmax,
                cbar_label=cbar_label, suptitle=title_of([lead]),
            )
            written += style.savefig(fig, f"{stem}_{lead}d", out_dir)
    return written


def _open_map_caches(cache_paths: dict, family: str):
    """The node/element map caches, or `None` if either is missing (family is logged)."""
    node_path = cache_paths.get("maps_FESOM_NODE")
    elem_path = cache_paths.get("maps_FESOM_ELEM")
    if node_path is None or not Path(node_path).exists():
        print(f"[skip] {family}: missing cache input 'maps_FESOM_NODE'")
        return None
    if elem_path is None or not Path(elem_path).exists():
        print(f"[skip] {family}: missing cache input 'maps_FESOM_ELEM'")
        return None
    return xr.open_dataset(node_path), xr.open_dataset(elem_path)


def _require_channels(ds_node, ds_elem, family: str) -> bool:
    """The four channels both families need: sst, ssh on nodes and the u/v pair on elems."""
    u_ch, v_ch = _EKE_CHANNELS
    for ch, ds, name in (
        ("sst", ds_node, "maps_FESOM_NODE"), ("ssh", ds_node, "maps_FESOM_NODE"),
        (u_ch, ds_elem, "maps_FESOM_ELEM"), (v_ch, ds_elem, "maps_FESOM_ELEM"),
    ):
        if ch not in ds.channel.values:
            print(f"[skip] {family}: channel {ch!r} not available in {name}")
            return False
    return True


def _uv_dry(mesh) -> np.ndarray:
    """Elements where either velocity component is dry -- EKE needs both."""
    elem_mask, _ = masks_and_weights(mesh, "FESOM_ELEM", list(_EKE_CHANNELS))
    return ~(elem_mask[0] & elem_mask[1])


def build_f08(cfg: dict, cache_paths: dict) -> list[Path]:
    opened = _open_map_caches(cache_paths, "f08")
    if opened is None:
        return []
    ds_node, ds_elem = opened
    if not _require_channels(ds_node, ds_elem, "f08"):
        return []

    leads = _lead_list(cfg, ds_node, ds_elem)
    if not leads:
        print("[skip] f08: none of leads.map_leads present in the map caches")
        return []

    mesh = get_mesh(cfg["mesh_path"])
    node_dry = {}
    for ch in ("sst", "ssh"):
        mask, _ = masks_and_weights(mesh, "FESOM_NODE", [ch])
        node_dry[ch] = ~mask[0]
    uv_dry = _uv_dry(mesh)
    u_ch, v_ch = _EKE_CHANNELS

    def _log2_ratio(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log2(pred / target)

    def _var(ds, name, lead, ch):
        return ds[name].sel(lead=lead, channel=ch).values.astype(np.float64)

    rows = []
    for lead in leads:
        eke_pred = 0.5 * (_var(ds_elem, "pred_var", lead, u_ch)
                          + _var(ds_elem, "pred_var", lead, v_ch))
        eke_target = 0.5 * (_var(ds_elem, "target_var", lead, u_ch)
                            + _var(ds_elem, "target_var", lead, v_ch))
        rows.append((lead, [
            _Panel(
                "Sea-surface temperature", "FESOM_NODE",
                _log2_ratio(_var(ds_node, "pred_var", lead, "sst"),
                            _var(ds_node, "target_var", lead, "sst")),
                node_dry["sst"],
            ),
            _Panel(
                "Sea-surface height", "FESOM_NODE",
                _log2_ratio(_var(ds_node, "pred_var", lead, "ssh"),
                            _var(ds_node, "target_var", lead, "ssh")),
                node_dry["ssh"],
            ),
            _Panel(
                "Surface eddy kinetic energy", "FESOM_ELEM",
                _log2_ratio(eke_pred, eke_target), uv_dry,
            ),
        ]))

    n_samples = ds_elem.attrs.get("n_samples", ds_node.attrs.get("n_samples", "?"))
    return _emit(
        cfg, rows, vmin=-_RATIO_LIMIT, vmax=_RATIO_LIMIT,
        cbar_label="log$_2$(forecast variance / reference variance)",
        title_of=lambda lds: (
            f"Forecast-to-reference variance ratio at {_leads_phrase(lds)} "
            f"({n_samples} initialisations)"
        ),
        stem="f08_variability",
    )


def build_f12(cfg: dict, cache_paths: dict) -> list[Path]:
    """Skill against climatology as maps, in the same format as f08.

    Companion to f08: where f08 asks whether the forecast carries the right *amount* of
    variability, this asks whether it is *more accurate than the seasonal cycle* at each
    point, on the same three fields, at the same leads and with the same shared colour
    bar. Skill is 1 - RMSE/RMSE_clim, the definition used by the scorecard and the
    lead-depth heatmaps, clipped to the same +/-1 as the other skill figures.
    """
    opened = _open_map_caches(cache_paths, "f12")
    if opened is None:
        return []
    ds_node, ds_elem = opened
    for name, ds in (("maps_FESOM_NODE", ds_node), ("maps_FESOM_ELEM", ds_elem)):
        if "rmse_climatology" not in ds.data_vars:
            print(f"[skip] f12: {name} predates the baseline-map fields (no rmse_climatology)")
            return []
    if not _require_channels(ds_node, ds_elem, "f12"):
        return []

    leads = _lead_list(cfg, ds_node, ds_elem)
    if not leads:
        print("[skip] f12: none of leads.map_leads present in the map caches")
        return []

    mesh = get_mesh(cfg["mesh_path"])
    node_dry = {}
    for ch in ("sst", "ssh"):
        mask, _ = masks_and_weights(mesh, "FESOM_NODE", [ch])
        node_dry[ch] = ~mask[0]
    uv_dry = _uv_dry(mesh)
    u_ch, v_ch = _EKE_CHANNELS

    def _node_skill(ch, lead):
        sel = {"lead": lead, "channel": ch}
        return _skill(
            ds_node["rmse"].sel(**sel).values.astype(np.float64),
            ds_node["rmse_climatology"].sel(**sel).values.astype(np.float64),
        )

    def _uv_skill(lead):
        # Vector velocity: RMSE_|U|^2 = RMSE_u^2 + RMSE_v^2 holds pointwise for both the
        # forecast and the climatology, so the vector skill follows from the component maps.
        sel_u = {"lead": lead, "channel": u_ch}
        sel_v = {"lead": lead, "channel": v_ch}
        e_uv = np.sqrt(ds_elem["rmse"].sel(**sel_u).values.astype(np.float64) ** 2
                       + ds_elem["rmse"].sel(**sel_v).values.astype(np.float64) ** 2)
        c_uv = np.sqrt(ds_elem["rmse_climatology"].sel(**sel_u).values.astype(np.float64) ** 2
                       + ds_elem["rmse_climatology"].sel(**sel_v).values.astype(np.float64) ** 2)
        return _skill(e_uv, c_uv)

    rows = [
        (lead, [
            _Panel("Sea-surface temperature", "FESOM_NODE",
                   _node_skill("sst", lead), node_dry["sst"]),
            _Panel("Sea-surface height", "FESOM_NODE",
                   _node_skill("ssh", lead), node_dry["ssh"]),
            _Panel("Horizontal velocity", "FESOM_ELEM", _uv_skill(lead), uv_dry),
        ])
        for lead in leads
    ]

    n_samples = ds_node.attrs.get("n_samples", ds_elem.attrs.get("n_samples", "?"))
    return _emit(
        cfg, rows, vmin=-1.0, vmax=1.0,
        cbar_label="skill against climatology (clipped at $-1$)",
        title_of=lambda lds: (
            f"Skill against climatology at {_leads_phrase(lds)} "
            f"({n_samples} initialisations)"
        ),
        stem="f12_skill_vs_climatology",
    )
