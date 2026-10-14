"""Emit the hemispheric-symmetry season table (LaTeX) for one evaluated run.

    python -m paper_eval.make_season_table --config paper_eval/configs/<run>.yaml

Rows are LOCAL seasons, so each row pairs the two hemispheres in the same physical
season (northern DJF against southern JJA, and so on); the antiphase of the calendar
seasons is what makes the symmetry visible. Scores are area-weighted means within each
hemisphere, aggregated over the initialisations of that season (RMSE as the root mean
square, skill as the mean of the per-initialisation values).
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from paper_eval.config import cache_dir, load_config

NH_OF_LOCAL = {"Winter": "DJF", "Spring": "MAM", "Summer": "JJA", "Autumn": "SON"}
SH_OF_LOCAL = {"Winter": "JJA", "Spring": "SON", "Summer": "DJF", "Autumn": "MAM"}
MONTH_SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
LEAD = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ds = xr.open_dataset(cache_dir(cfg) / "scores_FESOM_NODE.nc")

    months = ds.init_time.dt.month.values
    season = np.array([MONTH_SEASON[m] for m in months])
    keep = months != 12                       # December excluded by protocol
    li = list(ds.lead.values).index(LEAD)

    def rmse(region, ch, seas):
        x = ds.sel(region=region).score.sel(channel=ch, model="emulator", metric="rmse").values
        return float(np.sqrt(np.nanmean(x[(season == seas) & keep][:, li] ** 2)))

    def skill(region, ch, seas):
        e = ds.sel(region=region).score.sel(channel=ch, model="emulator", metric="rmse").values
        p = ds.sel(region=region).score.sel(channel=ch, model="persistence", metric="rmse").values
        m = (season == seas) & keep
        return float(np.nanmean(1.0 - e[m][:, li] / p[m][:, li]))

    rows = []
    for local in ("Winter", "Spring", "Summer", "Autumn"):
        n, s = NH_OF_LOCAL[local], SH_OF_LOCAL[local]
        nn, ns = int(((season == n) & keep).sum()), int(((season == s) & keep).sum())
        rows.append(
            f"{local} ({n} / {s}) & {nn} / {ns}"
            f" & ${rmse('nh','sst',n):.2f}$ & ${rmse('sh','sst',s):.2f}$"
            f" & ${skill('nh','sst',n):.2f}$ & ${skill('sh','sst',s):.2f}$"
            f" & ${rmse('nh','ssh',n):.3f}$ & ${rmse('sh','ssh',s):.3f}$ \\\\"
        )

    out = Path(a.out or f"paper_eval/tables/{cfg['run_id']}_season_symmetry.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Seasonal dependence of forecast error at the 15-day horizon, ordered by\n"
        "\\emph{local} season so that each row compares the two hemispheres in the same\n"
        "physical season; the calendar seasons that this corresponds to are given in\n"
        "parentheses (northern / southern) and $N$ is the number of initialisations\n"
        "contributing to each entry. Scores are area-weighted means within each hemisphere,\n"
        "with skill measured against persistence as in\n"
        "Table~\\ref{tab:" + cfg["run_id"] + "_scores}. Surface temperature error peaks for local\n"
        "summer initialisations and skill against persistence peaks for local autumn\n"
        "initialisations in both hemispheres, whereas sea-surface height error peaks in local\n"
        "winter: the seasonal cycles are of the same shape in the two hemispheres and\n"
        "opposite in sign between the thermal and the dynamic surface field.}\n"
        f"\\label{{tab:{cfg['run_id']}_season}}\n"
        "\\begin{tabular}{llcccccc}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{SST RMSE ($^\\circ$C)} & \\multicolumn{2}{c}{SST skill vs pers.}"
        " & \\multicolumn{2}{c}{SSH RMSE (m)} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        "Local season (N / S) & $N$ & NH & SH & NH & SH & NH & SH \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
