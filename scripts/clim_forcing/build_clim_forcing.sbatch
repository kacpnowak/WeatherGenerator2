#!/bin/bash
#SBATCH --job-name=clim_forcing
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/work/ab0995/a270225/clim_forcing/build_%j.out
# Calendar-day climatology (2170-2199) of the AWI-CM3 IFS forcing archive, written as year
# file(s) in the archive's own format. Reads ~57 GB per reference year and target year.
# Usage: sbatch scripts/clim_forcing/build_clim_forcing.sbatch [target years ...]  (default 2209)
cd /work/ab0995/a270225/WeatherGenerator2
YEARS=("${@:-2209}")
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/clim_forcing/build_clim_forcing.py \
    --src /work/ab0995/a270088/Kacper/weathergenertor/AWICM3/atmos_all \
    --dst /work/ab0995/a270225/clim_forcing/atmos_all \
    --ref-start 2170 --ref-end 2199 --years "${YEARS[@]}" --workers "$SLURM_CPUS_PER_TASK"
