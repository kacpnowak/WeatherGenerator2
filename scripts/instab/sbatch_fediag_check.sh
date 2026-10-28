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
#SBATCH --output=logs/fediag-check-%x.%j.out
#SBATCH --error=logs/fediag-check-%x.%j.err

# GPU checks of forecast.forcing_split and forecast.fe_noise with fvars003 (6-day rollouts
# from 2209-01-01, all five ocean streams written in full), compared afterwards by
# scripts/instab/compare_fediag.py:
#   fdchk0  default (no option), chunk 2                     == results/yr1fv003 leads 1-6
#   fdchk1  freeze, no option, chunk 2                       == results/lcfz003 leads 1-6
#   fdchk2  freeze, forcing_split (1, 0.5), one chunk        } identical (chunk boundary and
#   fdchk3  freeze, forcing_split (1, 0.5), 4 steps chunk 2, }  resume)
#           then resumed to 6
#   fdchk3  resumed again with split (1, 1.5) / without it   -> must refuse (mismatch)
#   fdchk4  freeze, forcing_split (1, 1), chunk 2            ~ lcfz003 (rounding only)
#   fdchk5  fe_noise train_like 1e-4 + dropout, one chunk    } identical (seeded noise and
#   fdchk6  the same, 4 steps chunk 2, then resumed to 6     }  dropout, chunk / resume)
#   fdchk6  resumed again with seed 1 / without fe_noise     -> must refuse (mismatch)
#   fdchk7  freeze, fe_noise token_rms 1e-3 steps first, chunk 2 (runs, differs from lcfz003)
#   fdchk8  freeze, forcing_split + forcing_scale            -> must refuse
#   fdchk9  no_forcing, forcing_split (1, 0.5), 2 steps      (runs)
# CONST: the forcing_split constant (C of fvars003). WT selects the code.
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
CONST=${CONST:-/home/a/a270225/claude_api_tools/fable_diag/D1/data/C_fvars003_2209-01-01.pt}
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

split() { echo "test_config.forecast.forcing_split.const_path=$CONST test_config.forecast.forcing_split.alpha_const=$1 test_config.forecast.forcing_split.alpha_anom=$2"; }
noise() { echo "test_config.forecast.fe_noise.mode=$1 test_config.forecast.fe_noise.std=$2 test_config.forecast.fe_noise.dropout=$3 test_config.forecast.fe_noise.seed=$4 test_config.forecast.fe_noise.steps=$5"; }

run() {  # RUN_ID NUM_STEPS CHUNK LAND(freeze | no_forcing | none) [EXTRA OPTIONS...]
  local id=$1 steps=$2 chunk=$3 land=$4
  shift 4
  local opts=(
    test_config.start_date=2209-01-01T00:00 test_config.end_date=2210-01-02T00:00
    test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1
    "test_config.output.streams=[TEMP,SALT,U,V,SURFACE]"
    test_config.forecast.num_steps=$steps test_config.forecast.chunk_size=$chunk
    test_config.forecast.resume=true test_config.compute_loss=false
  )
  for s in TEMP SALT U V SURFACE; do opts+=("streams.$s.max_num_targets=-1"); done
  if [ "$land" != none ]; then
    opts+=("test_config.forecast.land_cells.mode=$land"
           "test_config.forecast.land_cells.expected_num_land_cells=2863")
  fi
  opts+=($@)
  date; echo "== $id steps=$steps chunk=$chunk land_cells=$land extra: $*"
  srun --label .venv/bin/inference --from-run-id fvars003 -e -1 --run-id "$id" \
    --options "${opts[@]}" > "output/fediag_check_${id}_${steps}_${SLURM_JOBID}.txt" 2>&1 < /dev/null
  echo "srun exit code: $?"
}

run fdchk0 6 2 none
run fdchk1 6 2 freeze
run fdchk2 6 6 freeze $(split 1.0 0.5)
run fdchk3 4 2 freeze $(split 1.0 0.5)
run fdchk3 6 2 freeze $(split 1.0 0.5)
run fdchk3 8 2 freeze $(split 1.0 1.5)
run fdchk3 8 2 freeze
run fdchk4 6 2 freeze $(split 1.0 1.0)
run fdchk5 6 6 none $(noise train_like 0.0001 true 0 all)
run fdchk6 4 2 none $(noise train_like 0.0001 true 0 all)
run fdchk6 6 2 none $(noise train_like 0.0001 true 0 all)
run fdchk6 8 2 none $(noise train_like 0.0001 true 1 all)
run fdchk6 8 2 none
run fdchk7 6 2 freeze $(noise token_rms 0.001 false 0 first)
run fdchk8 2 2 freeze $(split 1.0 0.5) test_config.forecast.forcing_scale=0.5
run fdchk9 2 2 no_forcing $(split 1.0 0.5)
date
