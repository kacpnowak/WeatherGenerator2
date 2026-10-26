#!/bin/bash
#SBATCH --job-name=paper_eval_clim_fvars
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/ab0995/a270225/WeatherGenerator2/paper_eval/sbatch/clim_fvars_%A_%a.out
cd /work/ab0995/a270225/WeatherGenerator2
# Climatology sources for the per-variable (fesom_vars) streams: only the channels the
# shared FESOM_NODE/FESOM_ELEM climatologies lack (levels 535-2035 m, a_ice, m_ice).
# Usage: sbatch paper_eval/sbatch/build_clim_fvars.sbatch <config_path>
#   the config needs truth.FESOM_NODE_extra / truth.FESOM_ELEM_extra (see configs/yr1fv002.yaml)
CONFIG="${1:?config path required}"
DEEP=(535 630 735 850 975 1110 1255 1415 1600 1810 2035)
join() { local IFS=,; echo "$*"; }
NODE=$(join "${DEEP[@]/#/temp_}" "${DEEP[@]/#/salt_}" a_ice m_ice)
ELEM=$(join "${DEEP[@]/#/u_}" "${DEEP[@]/#/v_}")
STREAMS=(FESOM_NODE_extra FESOM_ELEM_extra)
CHANNELS=("$NODE" "$ELEM")
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m paper_eval.build_clim --config "$CONFIG" \
    --stream "${STREAMS[$SLURM_ARRAY_TASK_ID]}" --channels "${CHANNELS[$SLURM_ARRAY_TASK_ID]}"
