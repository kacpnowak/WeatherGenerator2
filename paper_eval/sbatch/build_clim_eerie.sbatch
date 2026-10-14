#!/bin/bash
#SBATCH --job-name=paper_eval_clim_eerie
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --time=08:00:00
# Note: the "compute" partition's own max time limit is 8:00:00 (sinfo), which sbatch
# rejects any --time above -- measured 2-year OCEAN_3D smoke build: ~6min I/O + ~1.5min
# write per year-pair, extrapolating to ~1.5-2h for the full 30-year (1984-2013) build,
# comfortably within this limit.
#SBATCH --array=0-1
#SBATCH --output=/work/ab0995/a270225/WeatherGenerator2/paper_eval/sbatch/clim_eerie_%A_%a.out
cd /work/ab0995/a270225/WeatherGenerator2
# Usage: sbatch paper_eval/sbatch/build_clim_eerie.sbatch [config_path]
#   config_path defaults to paper_eval/configs/dbuzcpgf.yaml
# Task 0 = OCEAN_2D (f64 accumulator (366, n_cells, 3ch) ~9.1G, well under this
# request); Task 1 = OCEAN_3D (f64 accumulator (366, n_cells, 14ch) ~42.5G, plus
# transient batch/cast-down buffers, up to ~192G) -- one #SBATCH --mem applies to
# every array task, so it is sized to the larger (OCEAN_3D) requirement, the same
# single-value-covers-both-streams pattern build_clim.sbatch already uses.
CONFIG="${1:-paper_eval/configs/dbuzcpgf.yaml}"
STREAMS=(OCEAN_2D OCEAN_3D)
.venv/bin/python -m paper_eval.build_clim_eerie --config "$CONFIG" \
    --stream "${STREAMS[$SLURM_ARRAY_TASK_ID]}"
