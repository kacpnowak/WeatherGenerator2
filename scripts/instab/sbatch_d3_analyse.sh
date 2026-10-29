#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-d3-%x.%j.out
#SBATCH --error=logs/instab-d3-%x.%j.err

# CPU scoring of the D3 additivity runs (analyse_d3.py): daily series per run, then the report.
#   MAX_LEAD=150 sbatch --job-name=d3ana scripts/instab/sbatch_d3_analyse.sh d3a09 ...
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ $# -gt 0 ]; then
  .venv/bin/python $WT/scripts/instab/analyse_d3.py compute "$@" --workers 12 \
    --max-lead "${MAX_LEAD:-150}" || exit 1
fi
.venv/bin/python $WT/scripts/instab/analyse_d3.py report
