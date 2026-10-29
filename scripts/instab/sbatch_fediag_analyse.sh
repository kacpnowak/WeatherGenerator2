#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/fediag-analyse-%x.%j.out
#SBATCH --error=logs/fediag-analyse-%x.%j.err

# CPU scoring of the fable diagnostics D1 / D2 / D9: daily global series per run, then the
# tables and figures.
#   WT=<worktree> sbatch --job-name=fdana scripts/instab/sbatch_fediag_analyse.sh RUN ...
# Year-long runs (series to MAX_LEAD days, report YEAR -> fable_diag/YEAR_NOISE):
#   MAX_LEAD=363 sbatch --job-name=fdyr scripts/instab/sbatch_fediag_analyse.sh nztl09 ...
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ $# -gt 0 ]; then
  .venv/bin/python $WT/scripts/instab/analyse_fediag.py compute "$@" --workers 12 \
    --max-lead "${MAX_LEAD:-150}" || exit 1
fi
if [ "${MAX_LEAD:-150}" -gt 150 ]; then
  .venv/bin/python $WT/scripts/instab/analyse_fediag.py report YEAR
else
  .venv/bin/python $WT/scripts/instab/analyse_fediag.py report
fi
