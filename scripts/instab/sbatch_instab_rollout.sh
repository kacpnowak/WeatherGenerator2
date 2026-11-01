#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-rollout-%x.%j.out
#SBATCH --error=logs/instab-rollout-%x.%j.err

# Free-running forced forecast from 2209-01-01 with the code of the instability worktree
# (opt-in latent filter, forecast.latent_filter). Copy of sbatch_fesom_forced_rollout.sh:
# runs from the main tree (its .venv, models/, results/, logs/, output/; the private config
# puts models/ and results/ under /work/ab0995/a270225/WeatherGenerator2) with the worktree's
# source folders first on PYTHONPATH.
#
#   WG_STREAMS=TEMP,SALT,SURFACE WG_FILTER=second_difference:0.1 \
#     sbatch --job-name=ins3sd10 scripts/instab/sbatch_instab_rollout.sh fvars003 -1 ins3sd10 150
#
# Arguments: source run, its mini-epoch (-1 = latest), output run id, number of daily steps.
# Environment (pass through the environment, never --export=ALL,X=a,b):
#   WG_STREAMS   output streams (default TEMP,SALT,SURFACE); each writes all its points
#   WG_FILTER    TYPE:NU latent filter (relax | second_difference), unset = no filter
#   WG_CHUNK     forecast.chunk_size (default 30)
#   WG_START     initial date (default 2209-01-01T00:00), WG_END end date (default 2210-01-02T00:00)
#   WG_EXTRA_OPTS further space-separated --options overrides

FROM_RUN=${1:?usage: sbatch sbatch_instab_rollout.sh FROM_RUN FROM_EPOCH RUN_ID NUM_STEPS}
FROM_EPOCH=${2:?missing FROM_EPOCH}
RUN_ID=${3:?missing RUN_ID}
NUM_STEPS=${4:?missing NUM_STEPS}
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}

module purge
module load git gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0

export CUDA_HOME=/sw/spack-levante/nvhpc-24.7-py26uc/Linux_x86_64/24.7/cuda/12.5/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/sw/spack-levante/gcc-11.2.0-bcn7mb/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH=/usr/bin:$PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src

STREAMS=${WG_STREAMS:-TEMP,SALT,SURFACE}
OPTS=(
  test_config.start_date=${WG_START:-2209-01-01T00:00}
  test_config.end_date=${WG_END:-2210-01-02T00:00}
  test_config.samples_per_mini_epoch=1
  test_config.output.num_samples=1
  "test_config.output.streams=[${STREAMS}]"
  test_config.forecast.num_steps=$NUM_STEPS
  test_config.forecast.chunk_size=${WG_CHUNK:-30}
  test_config.forecast.resume=true
  test_config.compute_loss=false
)
for s in ${STREAMS//,/ }; do OPTS+=("streams.$s.max_num_targets=-1"); done
if [ -n "${WG_FILTER:-}" ]; then
  OPTS+=("test_config.forecast.latent_filter.type=${WG_FILTER%%:*}"
         "test_config.forecast.latent_filter.nu=${WG_FILTER##*:}")
fi
if [ -n "${WG_EXTRA_OPTS:-}" ]; then
  read -r -a EXTRA <<< "$WG_EXTRA_OPTS"
  OPTS+=("${EXTRA[@]}")
fi
echo "$FROM_RUN (mini-epoch $FROM_EPOCH) -> run id $RUN_ID, streams $STREAMS, filter ${WG_FILTER:-none}"
echo "--options ${OPTS[*]}"
echo "Worktree commit: $(git -C $WT rev-parse --short HEAD)"
date

srun --label .venv/bin/inference --from-run-id "$FROM_RUN" -e "$FROM_EPOCH" --run-id "$RUN_ID" \
  --options "${OPTS[@]}" > "output/instab_rollout_${RUN_ID}_${SLURM_JOBID}.txt" 2>&1

rc=$?
echo "srun exit code: $rc"
date
exit $rc
