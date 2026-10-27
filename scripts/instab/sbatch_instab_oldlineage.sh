#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-oldlineage-%x.%j.out
#SBATCH --error=logs/instab-oldlineage-%x.%j.err

# Old-lineage comparison (combined_report section 5 step 2), all probes in one GPU job:
#   sbatch --job-name=insold sbatch_instab_oldlineage.sh
# frcwin03 (latest = the yr1win03 weights, FESOM_NODE/FESOM_ELEM layout): latent statistics
# along a 360-day rollout from 2209-01-01 and the gain probe at days 10/40/70;
# fvars003 / fvars002: latent statistics along a 150-day rollout. Output in $INSTAB_OUT.
export INSTAB_OUT=${INSTAB_OUT:-/home/a/a270225/claude_api_tools/oldlineage/data}
P=scripts/instab/sbatch_instab_probe.sh
TGT_STREAMS="" bash $P traj frcwin03 insoldt "${OLD_DAYS:-360}"
TGT_STREAMS="" bash $P gain frcwin03 insoldg "10 40 70"
bash $P traj fvars003 insold3 "${NEW_DAYS:-150}"
bash $P traj fvars002 insold2 "${NEW_DAYS:-150}"
date
