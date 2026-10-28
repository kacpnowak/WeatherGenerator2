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
#SBATCH --output=logs/fediag-decomp-%x.%j.out
#SBATCH --error=logs/fediag-decomp-%x.%j.err

# Forcing-injection decomposition (fable D1 step 1) with the worktree code, from the main tree:
#   sbatch --job-name=fdecomp scripts/instab/sbatch_forcing_decomp.sh "fvars003 frcwin03"
# One process per model, 363 days of forcing from 2209-01-01 (the last forcing index a rollout from 2209-01-01 can read; the data end at 2210-01-01T00), free default rollout of 150 days
# for the latent norms. fvars003 also writes C (const_path of forecast.forcing_split).
MODELS=${1:-fvars003 frcwin03}
START=${WG_START:-2209-01-01}
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
OUT=${DECOMP_OUT:-/home/a/a270225/claude_api_tools/fable_diag/D1/data}
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
end=$(date -d "$START + 1 year + 10 days" +%Y-%m-%d)
for m in $MODELS; do
  tag=${m}_${START}
  extra=()
  [ "$m" = fvars003 ] && extra=(--save-const "$OUT/C_${tag}.pt")
  opts=(test_config.start_date=${START}T00:00 test_config.end_date=${end}T00:00
        test_config.forecast.num_steps=366 test_config.forecast.chunk_size=1
        test_config.samples_per_mini_epoch=1 test_config.output.num_samples=0
        test_config.compute_loss=false)
  date; echo "== $m $START"
  srun --label .venv/bin/python "$WT/scripts/instab/forcing_decomp.py" --out "$OUT" --tag "$tag" --days "${DECOMP_DAYS:-363}" \
    "${extra[@]}" -- --from-run-id "$m" -e -1 --run-id "fd${m: -3}" \
    --options "${opts[@]}" > "output/forcing_decomp_${tag}_${SLURM_JOBID}.txt" 2>&1 < /dev/null
  echo "srun exit code: $?"
done
date
