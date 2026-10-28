#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-fcum-%x.%j.out
#SBATCH --error=logs/instab-fcum-%x.%j.err

# Trajectory probe (instab_probe.py --mode traj, with the cumulative-forcing statistics) for
# the old lineage and fvars003, free rollouts from 2209-01-01, in one GPU job:
#   INSTAB_OUT=<dir> sbatch --job-name=fcumtr scripts/instab/sbatch_fcum_traj.sh
# frcwin03 (FESOM_NODE/FESOM_ELEM layout, targets already all points): OLD_DAYS (default 360);
# fvars003: NEW_DAYS (default 150).
export INSTAB_OUT=${INSTAB_OUT:?set INSTAB_OUT}
P=scripts/instab/sbatch_instab_probe.sh
TGT_STREAMS="" bash $P traj frcwin03 fcumold "${OLD_DAYS:-360}"
bash $P traj fvars003 fcumnew "${NEW_DAYS:-150}"
date
