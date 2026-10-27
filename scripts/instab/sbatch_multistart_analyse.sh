#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-multistart-%x.%j.out
#SBATCH --error=logs/instab-multistart-%x.%j.err

# CPU analysis of the multi-start rollouts: daily global series per run, then the report.
#   sbatch --job-name=msana scripts/instab/sbatch_multistart_analyse.sh ms3a2150 yr1fv003 ...
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ $# -gt 0 ]; then
  .venv/bin/python $WT/scripts/instab/analyse_multistart.py compute "$@" --workers 12 || exit 1
fi
.venv/bin/python $WT/scripts/instab/analyse_multistart.py report
