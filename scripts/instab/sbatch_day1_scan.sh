#!/bin/bash -x
#SBATCH --account=ab0995_gpu
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/day1-scan-%x.%j.out
#SBATCH --error=logs/day1-scan-%x.%j.err

# Day-1 forecasts of FROM_RUN (mini-epoch EPOCH) from a list of start dates, one inference per
# start (run id PREFIX<YYYYMMDD>), TEMP/SALT/SURFACE written. Used to see how the encoded state
# and the first step depend on the start year (inside / beyond the training years).
#
#   sbatch --job-name=d1scan scripts/instab/sbatch_day1_scan.sh fvnn003 10 fvd1 2010-01-01 2010-07-01 ...

FROM_RUN=${1:?usage: sbatch_day1_scan.sh FROM_RUN EPOCH PREFIX DATE...}
EPOCH=${2:?}
PREFIX=${3:?}
shift 3

module purge
module load git gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0
export CUDA_HOME=/sw/spack-levante/nvhpc-24.7-py26uc/Linux_x86_64/24.7/cuda/12.5/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/sw/spack-levante/gcc-11.2.0-bcn7mb/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++ PATH=/usr/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

rc_all=0
for d in "$@"; do
  end=$(date -d "$d + 40 days" +%Y-%m-%d)  # the sampler needs room beyond the step
  id=${PREFIX}$(echo $d | tr -d -)
  srun --label .venv/bin/inference --from-run-id "$FROM_RUN" -e "$EPOCH" --run-id "$id" --options \
    test_config.start_date=${d}T00:00 test_config.end_date=${end}T00:00 \
    test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1 \
    "test_config.output.streams=[TEMP,SALT,SURFACE]" test_config.forecast.num_steps=1 \
    test_config.compute_loss=false test_config.forecast.resume=true test_config.forecast.chunk_size=1 \
    streams.TEMP.max_num_targets=-1 \
    streams.SALT.max_num_targets=-1 streams.SURFACE.max_num_targets=-1 \
    > "output/day1_scan_${id}_${SLURM_JOBID}.txt" 2>&1
  rc=$?
  # an exception lands in pdb and still exits 0, so also check for a traceback
  grep -q Traceback "output/day1_scan_${id}_${SLURM_JOBID}.txt" && rc=99
  echo "$d -> $id exit $rc"; [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
