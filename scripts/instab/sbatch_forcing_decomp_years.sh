#!/bin/bash -x
#SBATCH --account=ab0995_gpu
#SBATCH --partition=gpu
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/fdecomp-years-%x.%j.out
# sbatch_forcing_decomp.sh for several start years in one allocation (no C saved):
#   EPOCH=10 sbatch --job-name=fdyears scripts/instab/sbatch_forcing_decomp_years.sh fvnn003 2200 2201 ...
MODEL=${1:?MODEL YEAR...}; shift
for y in "$@"; do
  WG_START=${y}-01-01 bash scripts/instab/sbatch_forcing_decomp.sh "$MODEL"
done
