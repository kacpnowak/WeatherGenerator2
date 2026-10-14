#!/bin/bash
#SBATCH --job-name=paper_eval_clim
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/work/ab0995/a270225/WeatherGenerator2/paper_eval/sbatch/clim_%A_%a.out
cd /work/ab0995/a270225/WeatherGenerator2
# Usage: sbatch [--array=...] paper_eval/sbatch/build_clim.sbatch [config_path]
#   config_path defaults to paper_eval/configs/default.yaml
CONFIG="${1:-paper_eval/configs/default.yaml}"
STREAMS=(FESOM_NODE FESOM_ELEM)
.venv/bin/python -m paper_eval.build_clim --config "$CONFIG" \
    --stream "${STREAMS[$SLURM_ARRAY_TASK_ID]}"
