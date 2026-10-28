#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-check-%x.%j.out
#SBATCH --error=logs/instab-check-%x.%j.err

# GPU checks of forecast.forcing_scale with fvars003 (6-day rollouts from 2209-01-01, all five
# ocean streams written in full), compared afterwards by scripts/instab/compare_forcescale.py:
#   fschk0  default (no option), chunk 2         == results/yr1fv003 leads 1-6
#   fschk1  forcing_scale=1.0, chunk 2           == results/yr1fv003 leads 1-6
#   fschk2  freeze, no forcing_scale, chunk 2    == results/lcfz003 leads 1-6
#   fschk5  freeze, forcing_scale=1.0, chunk 2   == results/lcfz003 leads 1-6
#   fschk4  freeze, forcing_scale=0.5, one chunk } identical (chunk boundary and resume)
#   fschk3  freeze, forcing_scale=0.5, 4 steps chunk 2, then resumed to 6
#   fschk3  resumed again with forcing_scale 0.75 / without it -> must refuse (mismatch)
# WT selects the code (default: the main tree).
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
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

run() {  # RUN_ID NUM_STEPS CHUNK LAND(freeze | none) ALPHA(none | value)
  local opts=(
    test_config.start_date=2209-01-01T00:00 test_config.end_date=2210-01-02T00:00
    test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1
    "test_config.output.streams=[TEMP,SALT,U,V,SURFACE]"
    test_config.forecast.num_steps=$2 test_config.forecast.chunk_size=$3
    test_config.forecast.resume=true test_config.compute_loss=false
  )
  for s in TEMP SALT U V SURFACE; do opts+=("streams.$s.max_num_targets=-1"); done
  if [ "$4" != none ]; then
    opts+=("test_config.forecast.land_cells.mode=$4"
           "test_config.forecast.land_cells.expected_num_land_cells=2863")
  fi
  if [ "$5" != none ]; then opts+=("test_config.forecast.forcing_scale=$5"); fi
  date; echo "== $1 steps=$2 chunk=$3 land_cells=$4 forcing_scale=$5"
  srun --label .venv/bin/inference --from-run-id fvars003 -e -1 --run-id "$1" \
    --options "${opts[@]}" > "output/instab_check_$1_$2_$4_$5_${SLURM_JOBID}.txt" 2>&1 < /dev/null
  echo "srun exit code: $?"
}

run fschk0 6 2 none none
run fschk1 6 2 none 1.0
run fschk2 6 2 freeze none
run fschk5 6 2 freeze 1.0
run fschk4 6 6 freeze 0.5
run fschk3 4 2 freeze 0.5
run fschk3 6 2 freeze 0.5
run fschk3 8 2 freeze 0.75
run fschk3 8 2 freeze none
date
