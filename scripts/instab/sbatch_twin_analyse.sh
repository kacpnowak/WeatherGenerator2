#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-twin-%x.%j.out
#SBATCH --error=logs/instab-twin-%x.%j.err

# CPU scoring of the IC-memory twin runs (twin_ic.py): one compute per TWIN:REF pair, then
# the report.
#   sbatch --job-name=twana scripts/instab/sbatch_twin_analyse.sh tw075:fs075y09 tw100:lcfz003 ...
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for p in "$@"; do
  .venv/bin/python $WT/scripts/instab/twin_ic.py compute "${p%%:*}" "${p##*:}" --workers 12 || exit 1
done
.venv/bin/python $WT/scripts/instab/twin_ic.py report
