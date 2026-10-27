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
#SBATCH --output=logs/instab-check-%x.%j.out
#SBATCH --error=logs/instab-check-%x.%j.err

# GPU checks of the latent filter with fvars003 (6-day rollouts from 2209-01-01, TEMP/SALT/
# SURFACE written in full), compared afterwards by scripts/instab/compare_check.py:
#   inschk0  no filter, chunk 2                 == results/yr1fv003 leads 1-6 (default unchanged)
#   inschk1  second_difference 0.25, one chunk  } identical: the filter state survives chunk
#   inschk2  second_difference 0.25, chunk 2    } boundaries
#   inschk3  second_difference 0.25, 4 steps chunk 2, then resumed to 6 steps (rollout_state.pt)
#   inschk3  resumed again with relax 0.25      -> must refuse (filter mismatch)
WT=/work/ab0995/a270225/WeatherGenerator2-instab
module purge
module load git gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0
export CUDA_HOME=/sw/spack-levante/nvhpc-24.7-py26uc/Linux_x86_64/24.7/cuda/12.5/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/sw/spack-levante/gcc-11.2.0-bcn7mb/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++ PATH=/usr/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=$WT/src:$WT/packages/common/src:$WT/packages/readers_extra/src:$WT/packages/evaluate/src:$WT/packages/metrics/src
echo "Worktree commit: $(git -C $WT rev-parse --short HEAD)"

run() {  # RUN_ID NUM_STEPS CHUNK FILTER(TYPE:NU or none)
  local opts=(
    test_config.start_date=2209-01-01T00:00 test_config.end_date=2210-01-02T00:00
    test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1
    "test_config.output.streams=[TEMP,SALT,SURFACE]"
    test_config.forecast.num_steps=$2 test_config.forecast.chunk_size=$3
    test_config.forecast.resume=true test_config.compute_loss=false
    streams.TEMP.max_num_targets=-1 streams.SALT.max_num_targets=-1
    streams.SURFACE.max_num_targets=-1
  )
  if [ "$4" != none ]; then
    opts+=("test_config.forecast.latent_filter.type=${4%%:*}" "test_config.forecast.latent_filter.nu=${4##*:}")
  fi
  date; echo "== $1 steps=$2 chunk=$3 filter=$4"
  srun --label .venv/bin/inference --from-run-id fvars003 -e -1 --run-id "$1" \
    --options "${opts[@]}" > "output/instab_check_$1_$2_${SLURM_JOBID}.txt" 2>&1 < /dev/null
  echo "srun exit code: $?"
}

run inschk0 6 2 none
run inschk1 6 6 second_difference:0.25
run inschk2 6 2 second_difference:0.25
run inschk3 4 2 second_difference:0.25
run inschk3 6 2 second_difference:0.25
run inschk3 8 2 relax:0.25
date
