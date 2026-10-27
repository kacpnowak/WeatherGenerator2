#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/instab-probe-%x.%j.out
#SBATCH --error=logs/instab-probe-%x.%j.err

# Instability probes (scripts/instab/instab_probe.py) with the worktree code, from the main tree.
#   sbatch --job-name=insenc3 sbatch_instab_probe.sh encdec fvars003 insenc3 "2150-01-01 2209-01-01"
#   sbatch --job-name=inslin3 sbatch_instab_probe.sh linear fvars003 inslin3 "10 40 70" [probe args]
# encdec: one probe per initial date (5 decoded steps); linear: one probe per day N of the
# free rollout from 2209-01-01. Text/data go to $INSTAB_OUT.
MODE=${1:?mode}; FROM_RUN=${2:?from run}; RUN_ID=${3:?run id}; LIST=${4:?dates or days}
shift 4
WT=${WT:-/work/ab0995/a270225/WeatherGenerator2}
OUT=${INSTAB_OUT:-/home/a/a270225/claude_api_tools/instab_tests/data}
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
TGT=(streams.TEMP.max_num_targets=-1 streams.SALT.max_num_targets=-1 streams.SURFACE.max_num_targets=-1)

for item in $LIST; do
  if [ "$MODE" = encdec ]; then
    start=$item
    end=$(date -d "$start + 15 days" +%Y-%m-%d)
    tag=${FROM_RUN}_${start}
    opts=(test_config.start_date=${start}T00:00 test_config.end_date=${end}T00:00
          test_config.forecast.num_steps=5 test_config.forecast.chunk_size=1)
    pargs=(--mode encdec --out "$OUT" --tag "$tag" --steps 5)
  else
    day=$item
    tag=${FROM_RUN}
    opts=(test_config.start_date=2209-01-01T00:00 test_config.end_date=2210-01-02T00:00
          test_config.forecast.num_steps=$((day + 5)) test_config.forecast.chunk_size=1)
    pargs=(--mode "$MODE" --out "$OUT" --tag "$tag" --day "$day")
  fi
  opts+=(test_config.samples_per_mini_epoch=1 test_config.output.num_samples=0
         test_config.compute_loss=false "${TGT[@]}")
  date; echo "== $MODE $FROM_RUN $item"
  srun --label .venv/bin/python "$WT/scripts/instab/instab_probe.py" "${pargs[@]}" "$@" -- \
    --from-run-id "$FROM_RUN" -e -1 --run-id "$RUN_ID" --options "${opts[@]}" \
    > "output/instab_probe_${MODE}_${tag}_${item}_${SLURM_JOBID}.txt" 2>&1 < /dev/null
  echo "srun exit code: $?"
done
date
