"""Shared paper-figure style: rcParams, palette, units, channel labels, `savefig`.

Colors follow the dataviz-skill formula (see `.superpowers` dataviz skill,
`references/palette.md`): categorical identity uses the validated 8-hue order
(first three slots -- blue/orange/aqua -- are the ones cleared for all-pairs
comparison, i.e. safe wherever more than two colored series sit side by side);
magnitude uses one sequential ramp (viridis); polarity uses one diverging pair
(RdBu_r) with a neutral, zero-centred midpoint.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from paper_eval.meshtools import parse_channel

# ── rcParams: one consistent, paper-quality look for every figure ─────────────────────
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "figure.titlesize": 11,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ── Categorical palette (dataviz skill default, light-mode hexes) ─────────────────────
_CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "viridis"

# ── Model identity (f01, f02, f07): one accent color, baselines in neutral grey ───────
MODEL_COLORS = {
    "emulator": _CATEGORICAL[0],  # blue accent
    "persistence": "#898781",  # muted grey
    "climatology": "#898781",  # same grey -- distinguished by linestyle only
}
MODEL_LINESTYLES = {"emulator": "-", "persistence": "--", "climatology": ":"}
MODEL_LABELS = {"emulator": "Emulator", "persistence": "Persistence", "climatology": "Climatology"}

# ── Lead identity (f04, f08, f09): 2 leads, categorical slots 2 & 7 (orange/violet) ────
LEAD_COLORS = {15: _CATEGORICAL[1], 30: _CATEGORICAL[6]}


def lead_color(lead: int) -> str:
    """Categorical color for a lead; falls back to secondary ink for an
    unconfigured lead value rather than guessing a new hue."""
    return LEAD_COLORS.get(lead, "#52514e")


def depth_colors(n: int) -> list[str]:
    """The first `n` categorical hues, for depth-identity curves (f02).

    The first three slots (blue/orange/aqua) are the ones validated for
    side-by-side ("all-pairs") comparison, matching `figures.curve_depths`
    having 3 entries by default.
    """
    if n <= len(_CATEGORICAL):
        return _CATEGORICAL[:n]
    return [_CATEGORICAL[i % len(_CATEGORICAL)] for i in range(n)]  # pathological config only


# ── Units and channel display labels ───────────────────────────────────────────────────
#: EERIE (run dbuzcpgf) 2-D channels: complete, terminal channel names (no depth
#: suffix) -- the EERIE counterpart of FESOM's `("sst", "sss", "ssh")` special case.
_EERIE_2D_CHANNELS = ("avg_tos", "avg_sos", "avg_zos")

UNITS = {
    "sst": "°C",
    "temp": "°C",
    "sss": "psu",
    "salt": "psu",
    "ssh": "m",
    "u": "m/s",
    "v": "m/s",
    "EKE": "m²/s²",
    "avg_tos": "°C",
    "avg_thetao": "°C",
    "avg_sos": "psu",
    "avg_so": "psu",
    "avg_zos": "m",
    "avg_uoe": "m/s",
    "avg_von": "m/s",
}

PREFIX_DISPLAY = {
    "sst": "SST",
    "sss": "SSS",
    "ssh": "SSH",
    "temp": "T",
    "salt": "S",
    "u": "u",
    "v": "v",
    "avg_tos": "SST",
    "avg_sos": "SSS",
    "avg_zos": "SSH",
    "avg_thetao": "T",
    "avg_so": "S",
    "avg_uoe": "u",
    "avg_von": "v",
}


def channel_prefix(name: str) -> str:
    """The variable prefix of a channel name (`"temp_107"` -> `"temp"`, `"sst"` ->
    `"sst"`, `"avg_thetao_2.5m"` -> `"avg_thetao"`, `"avg_tos"` -> `"avg_tos"`)."""
    if name in ("sst", "sss", "ssh") or name in _EERIE_2D_CHANNELS:
        return name
    prefix, sep, _ = name.rpartition("_")
    return prefix if sep else name


def channel_unit(name: str) -> str:
    """Display unit for a channel or bare prefix (`"temp_107"`, `"temp"`, `"EKE"` all work)."""
    if name in UNITS:
        return UNITS[name]
    return UNITS.get(channel_prefix(name), "")


def channel_label(name: str, nz1=None) -> str:
    """Human display label: `"sst"` -> `"SST"`, `"temp_107"` -> `"T @ 107.5 m"`,
    `"avg_thetao_2.5m"` -> `"T @ 2.5m"`, `"avg_uoe_surf"` -> `"u @ surface"`.

    `nz1` (mesh depth levels, `meshtools.Mesh.nz1`) resolves the exact FESOM depth
    via `meshtools.parse_channel`; without it the suffix is used verbatim (e.g.
    "107"). EERIE channels (`"avg_"`-prefixed) never consult `nz1` -- their own
    depth suffix ('2.5m', '97.5m', or the literal 'surf') is already the display
    value, printed as-is except 'surf', which reads better spelled out.
    """
    if name in ("sst", "sss", "ssh") or name in _EERIE_2D_CHANNELS:
        return PREFIX_DISPLAY[name]

    prefix, sep, suffix = name.rpartition("_")
    if not sep:
        return name

    if prefix in PREFIX_DISPLAY and prefix.startswith("avg_"):
        disp = PREFIX_DISPLAY[prefix]
        depth_str = "surface" if suffix == "surf" else suffix
        return f"{disp} @ {depth_str}"

    depth = None
    if nz1 is not None:
        try:
            _, depth, _ = parse_channel(name, nz1)
        except ValueError:
            depth = None
    if depth is None:
        try:
            depth = float(suffix)
        except ValueError:
            return name

    disp = PREFIX_DISPLAY.get(prefix, prefix)
    return f"{disp} @ {depth:g} m"


# ── Normalization helpers (magnitude / polarity panels) ────────────────────────────────
def symmetric_limits(values, mask=None, pct: float = 99.0) -> tuple[float, float]:
    """`(-limit, limit)` from the `pct`-th percentile of `|values|`, ignoring
    non-finite entries -- for error/bias/ratio panels centred at zero.

    `mask`, if given, is a boolean "keep" mask (True = include), applied before
    flattening. Falls back to `(-1, 1)` if nothing finite survives.
    """
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(np.abs(v), pct))
    if limit <= 0:
        limit = float(np.max(np.abs(v))) if v.size else 1.0
        if limit <= 0:
            limit = 1.0
    return -limit, limit


def sequential_limits(values, mask=None, pct: float = 99.0) -> tuple[float, float]:
    """`(0, vmax)` from the `pct`-th percentile of `values`, ignoring non-finite
    entries -- for magnitude panels (RMSE, EKE) that start at zero.
    """
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = v[np.asarray(mask, dtype=bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(v, pct))
    if vmax <= 0:
        vmax = float(np.max(v)) if v.size else 1.0
        if vmax <= 0:
            vmax = 1.0
    return 0.0, vmax


# ── Axis/frame helpers ──────────────────────────────────────────────────────────────
def full_frame(ax) -> None:
    """Ensure all four spines on `ax` are visible.

    Every axes gets a full solid frame by default now (the module rcParams no
    longer turn any spine off), so this is a defensive no-op in the common case --
    kept as an explicit call on maps/heatmaps (imshow scorecards/heatmaps,
    tripcolor maps) so their closed-frame look does not silently depend on the
    rcParams default staying what it is today.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)


def no_sci_offset(ax, axis: str = "y") -> None:
    """Disable scientific/offset notation on `axis` of `ax`.

    Matplotlib's default `ScalarFormatter` can add a floating "+1.2e-3"-style
    offset/multiplier box next to an axis whose values sit in a narrow range near
    zero; the axis/colorbar label here already carries the physical unit, so the
    plain numbers read better than an extra offset annotation. A no-op on axes
    that don't support it (e.g. log-scaled), rather than raising.
    """
    try:
        ax.ticklabel_format(style="plain", useOffset=False, axis=axis)
    except (AttributeError, ValueError):
        pass


# ── Output ──────────────────────────────────────────────────────────────────────────
def savefig(fig, name: str, fig_dir) -> list[Path]:
    """Write `fig` as `<fig_dir>/<name>.pdf` and `.png` (300 dpi, tight bbox);
    closes `fig` and returns the two written paths."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        path = fig_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
