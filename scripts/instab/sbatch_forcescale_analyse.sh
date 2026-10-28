#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-forcescale-%x.%j.out
#SBATCH --error=logs/instab-forcescale-%x.%j.err

# CPU analysis of the forcing-scale sweep: daily global series per run, then the report.
#   WT=<worktree> sbatch --job-name=fsana scripts/instab/sbatch_forcescale_analyse.sh fs050y09 ...
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ $# -gt 0 ]; then
  .venv/bin/python $WT/scripts/instab/analyse_forcescale.py compute "$@" --workers 12 || exit 1
fi
.venv/bin/python $WT/scripts/instab/analyse_forcescale.py report
