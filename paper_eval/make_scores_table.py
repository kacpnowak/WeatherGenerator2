"""Emit the paper's summary score table (LaTeX) for one evaluated run.

Reads only the stage-1 score caches, so it never touches the raw inference output:
    python -m paper_eval.make_scores_table --config paper_eval/configs/<run>.yaml

Definitions (stated in the generated caption):
  RMSE   area-weighted global mean over wet points, per initialisation.
  Skill  1 - RMSE_emulator / RMSE_persistence, per initialisation.
  +-     one standard deviation across initialisations.
  |U|    vector velocity error, RMSE_|U|^2 = RMSE_u^2 + RMSE_v^2 (exact: u and v
         share identical wet masks and area weights at a given level).
Velocity rows are emitted only where BOTH components exist in the run's output.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import xarray as xr

from paper_eval.config import cache_dir, load_config

SURFACE = [("SST", "sst", r"$^\circ$C"), ("SSS", "sss", "psu"), ("SSH", "ssh", "m")]
DEPTHS = [(25, "25"), (95, "95"), (210, "210"), (450, "450")]
LABEL = {"temp": "$T$", "salt": "$S$"}
UNIT = {"temp": r"$^\circ$C", "salt": "psu"}
VEL_DEPTHS = [(2.5, "2"), (25, "25"), (95, "95"), (210, "210"), (450, "450")]


def _decimals(*stds):
    """Enough decimals for the smallest s.d. to remain visible (2..4)."""
    s = min(x for x in stds if x > 0) if any(x > 0 for x in stds) else 0.0
    return 2 if s <= 0 else int(min(4, max(2, math.ceil(-math.log10(s)))))


def _cell(v, nd):
    return f"${v.mean():.{nd}f}{{\\pm}}{v.std():.{nd}f}$"


def _row(label, unit, e, p, i15, i30):
    nd = _decimals(e[:, i15].std(), e[:, i30].std())
    sk = 1.0 - e / p
    return (
        f"{label} & {_cell(e[:, i15], nd)}\\,{unit} & {_cell(e[:, i30], nd)}\\,{unit}"
        f" & {_cell(sk[:, i15], 2)} & {_cell(sk[:, i30], 2)} \\\\"
    )


def build(cfg):
    root = cache_dir(cfg)
    node = xr.open_dataset(root / "scores_FESOM_NODE.nc").sel(region="global")
    elem = xr.open_dataset(root / "scores_FESOM_ELEM.nc").sel(region="global")
    leads = list(node.lead.values)
    i15, i30 = leads.index(15), leads.index(30)

    def sc(ds, ch, model):
        return ds.score.sel(channel=ch, model=model, metric="rmse").values

    rows, missing_vel = [], []
    for label, ch, unit in SURFACE:
        rows.append(_row(label, unit, sc(node, ch, "emulator"), sc(node, ch, "persistence"), i15, i30))

    for var in ("temp", "salt"):
        rows.append(r"\addlinespace")
        for depth, suf in DEPTHS:
            ch = f"{var}_{suf}"
            if ch not in node.channel.values:
                continue
            rows.append(
                _row(rf"{LABEL[var]} @ {depth}\,m", UNIT[var],
                     sc(node, ch, "emulator"), sc(node, ch, "persistence"), i15, i30)
            )

    rows.append(r"\addlinespace")
    have = set(map(str, elem.channel.values))
    for depth, suf in VEL_DEPTHS:
        u, v = f"u_{suf}", f"v_{suf}"
        if u not in have or v not in have:
            missing_vel.append(depth)
            continue
        e = np.sqrt(sc(elem, u, "emulator") ** 2 + sc(elem, v, "emulator") ** 2)
        p = np.sqrt(sc(elem, u, "persistence") ** 2 + sc(elem, v, "persistence") ** 2)
        rows.append(_row(rf"$|U|$ @ {depth}\,m", r"m\,s$^{-1}$", e, p, i15, i30))

    t = np.sort(node.init_time.values.astype("datetime64[D]"))
    return rows, dict(n=node.sizes["sample"], first=str(t[0]), last=str(t[-1]),
                      run=cfg["run_id"], missing_vel=missing_vel)


def caption(meta):
    vel_note = ""
    if meta["missing_vel"]:
        d = ", ".join(f"{x}\\,m" for x in meta["missing_vel"])
        vel_note = (f" The {d} level is omitted for velocity because the meridional "
                    "component is absent from the model output at that depth.")
    return rf"""\caption{{Forecast accuracy of the FESOM2 emulator at the 15- and 30-day horizons.
Root-mean-square errors are area-weighted global means over wet mesh points, verified
against the FESOM2 reference simulation. Skill is measured against a persistence
forecast that carries the initial state forward,
$1-\mathrm{{RMSE}}_{{\mathrm{{emu}}}}/\mathrm{{RMSE}}_{{\mathrm{{pers}}}}$, so that positive values
denote an improvement over persistence and negative values a degradation. Both metrics
are evaluated separately for each of the $N={meta['n']}$ daily initialisations of the held-out
year ({meta['first']} to {meta['last']}, none of which the model saw during training) and are
reported as the mean $\pm$ one standard deviation across those initialisations. SST, SSS
and SSH are the prognostic surface fields; temperature $T$, salinity $S$ and horizontal
velocity $|U|$ are additionally shown at depths reaching from the mixed layer into the
permanent thermocline. $|U|$ denotes the vector velocity error,
$\mathrm{{RMSE}}_{{|U|}}^{{2}}=\mathrm{{RMSE}}_{{u}}^{{2}}+\mathrm{{RMSE}}_{{v}}^{{2}}$.{vel_note}}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rows, meta = build(cfg)
    out = Path(a.out or f"paper_eval/tables/{meta['run']}_scores.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(rows)
    out.write_text(
        "\\begin{table}[t]\n\\centering\n"
        + caption(meta)
        + f"\n\\label{{tab:{meta['run']}_scores}}\n"
        + "\\begin{tabular}{lcccc}\n\\toprule\n"
        + " & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{Skill vs pers.} \\\\\n"
        + "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        + "Field & +15 d & +30 d & +15 d & +30 d \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    print(f"wrote {out}  (N={meta['n']}, {meta['first']}..{meta['last']}, "
          f"{sum(1 for r in rows if r != chr(92)+'addlinespace')} data rows)")


if __name__ == "__main__":
    main()
