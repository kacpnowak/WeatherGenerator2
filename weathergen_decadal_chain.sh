#!/bin/bash
#SBATCH --job-name=decadal_freerun
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -A ch17
#SBATCH --output=logs/decadal-%j.out
#SBATCH --error=logs/decadal-%j.err

# Self-chaining decadal free-running rollout on the 30-min debug partition.
#
# Each job resumes the rollout from the saved latent (forecast.resume=true), runs as many
# chunks as fit in 30 min, and is killed by SLURM at the wall. A successor job is submitted up
# front with `--dependency=afterany` so the chain continues whether this job finishes cleanly or
# is killed by the time limit. The chain stops itself once done_steps reaches num_steps.
#
# Usage (submit the first link):
#   sbatch weathergen_decadal_chain.sh <RUN_ID> <NUM_STEPS> [MAX_LINKS]

set -uo pipefail
REPO=/users/apatnala/WeatherGen/WeatherGenerator
cd "$REPO"

RUN_ID=${1:?"usage: sbatch weathergen_decadal_chain.sh RUN_ID NUM_STEPS [MAX_LINKS]"}
NUM_STEPS=${2:?"usage: sbatch weathergen_decadal_chain.sh RUN_ID NUM_STEPS [MAX_LINKS]"}
MAX=${3:-25}
CKPT=oq9o0t86
SCRIPT="$REPO/weathergen_decadal_chain.sh"
STATE="$REPO/results/$RUN_ID/rollout_state.pt"

source .venv/bin/activate

# forecast steps completed so far (0 if the rollout has not started yet)
DONE=$(.venv/bin/python3 -c "import torch,glob;f=glob.glob('$STATE');print(torch.load(f[0],map_location='cpu',weights_only=False)['done_steps'] if f else 0)" 2>/dev/null || echo 0)
echo "[chain] run_id=$RUN_ID done=$DONE / $NUM_STEPS  links_left=$MAX  job=$SLURM_JOB_ID"

if [ "$DONE" -ge "$NUM_STEPS" ]; then
  echo "[chain] rollout complete ($DONE steps); not chaining further."
  exit 0
fi
if [ "$MAX" -le 0 ]; then
  echo "[chain] max chain length reached with only $DONE/$NUM_STEPS done; stopping."
  exit 0
fi

# queue the successor now so the chain survives this job being killed at the 30-min wall
sbatch --dependency=afterany:"$SLURM_JOB_ID" "$SCRIPT" "$RUN_ID" "$NUM_STEPS" $((MAX - 1)) \
  || echo "[chain] WARNING: failed to submit successor job"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun uv --offline run inference --from-run-id "$CKPT" --run-id "$RUN_ID" \
  --options test_config.start_date=202301010000 test_config.end_date=202602010000 \
  test_config.output.num_samples=1 test_config.samples_per_mini_epoch=1 \
  test_config.forecast.num_steps="$NUM_STEPS" test_config.forecast.chunk_size=50 \
  test_config.forecast.free_running=true test_config.forecast.resume=true \
  test_config.output.streams=[ERA5] test_config.compute_loss=false
