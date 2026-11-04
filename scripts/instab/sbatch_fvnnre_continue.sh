#!/bin/bash
#SBATCH --account=ab0995
#SBATCH --partition=shared
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/fvnnre-continue.%j.out
# After a keep-states rollout ends (e.g. at the time limit), continue it from its last saved state.
# A kill leaves the ZipStore without its central directory, so a non-readable zip is moved aside
# and the continuation writes a fresh one (only the saved states are needed downstream).
RUN=${1:?RUN_ID}; STEPS=${2:?NUM_STEPS}
R=results/$RUN
done_steps=$(.venv/bin/python -c "import torch;print(torch.load('$R/rollout_state.pt',map_location='cpu',weights_only=False)['done_steps'])")
echo "saved state at done_steps=$done_steps"
[ "$done_steps" -ge "$STEPS" ] && { echo "already complete"; exit 0; }
if true; then  # a killed ZipStore cannot be trusted even if it opens
  mv $R/validation_chkpt00000_rank0000.zip $R/broken_before_${done_steps}.zip
  echo "zip not readable, moved aside"
fi
# submitted from inside a job: drop the inherited SLURM memory settings, srun rejects the mix
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
WG_STREAMS=SURFACE WG_START=2200-01-01T00:00 WG_END=2210-01-02T00:00 \
  sbatch --parsable --job-name=${RUN} scripts/instab/sbatch_rollout_keep_states.sh fvnn003 10 $RUN $STEPS
