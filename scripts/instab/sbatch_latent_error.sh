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
#SBATCH --output=logs/latenterr-%x.%j.out
#SBATCH --error=logs/latenterr-%x.%j.err

# Latent-error decomposition (scripts/instab/latent_error.py): one process per model and start
# date, encoded truth days 0..D, free rollouts (default and land_cells freeze), gain probes.
#   sbatch --job-name=le3a sbatch_latent_error.sh fvars003 "2209-01-01 2150-01-01" 150 [args]
# Output to $LATENTERR_OUT (default ~/claude_api_tools/latenterr/data). WT selects the code.
FROM_RUN=${1:?from run}; STARTS=${2:?start dates}; DAYS=${3:?days}
shift 3
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
OUT=${LATENTERR_OUT:-/home/a/a270225/claude_api_tools/latenterr/data}
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
for start in $STARTS; do
  end=$(date -d "$start + 1 year + 10 days" +%Y-%m-%d)
  tag=${FROM_RUN}_${start}
  opts=(test_config.start_date=${start}T00:00 test_config.end_date=${end}T00:00
        test_config.forecast.num_steps=$((DAYS + 3)) test_config.forecast.chunk_size=1
        test_config.samples_per_mini_epoch=1 test_config.output.num_samples=0
        test_config.compute_loss=false)
  for s in TEMP SALT SURFACE; do opts+=("streams.${s}.max_num_targets=-1"); done
  date; echo "== $FROM_RUN $start $DAYS"
  srun --label .venv/bin/python "$WT/scripts/instab/latent_error.py" --out "$OUT" --tag "$tag" \
    --days "$DAYS" "$@" -- --from-run-id "$FROM_RUN" -e -1 --run-id "le${FROM_RUN: -3}" \
    --options "${opts[@]}" > "output/latenterr_${tag}_${SLURM_JOBID}.txt" 2>&1 < /dev/null
  echo "srun exit code: $?"
done
date
