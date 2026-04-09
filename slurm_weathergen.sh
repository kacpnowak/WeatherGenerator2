#!/bin/bash -x
#SBATCH --account=ab0995
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M 
#SBATCH --gres=gpu:a100_80:4
#SBATCH --chdir=.
#SBATCH --output=logs/atmorep-%x.%j.out
#SBATCH --error=logs/atmorep-%x.%j.err

# import modules at dkrz
module purge
module load git gcc/11.2.0-gcc-11.2.0 nvhpc/24.7-gcc-11.2.0

export CUDA_HOME=/sw/spack-levante/nvhpc-24.7-py26uc/Linux_x86_64/24.7/cuda/12.5/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/sw/spack-levante/gcc-11.2.0-bcn7mb/lib64:$LD_LIBRARY_PATH

# Set environment variables to point to GCC
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Ensure GCC is first in your PATH
export PATH=/usr/bin:$PATH

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=TRACE
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

# export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCHDYNAMO_VERBOSE=1  # Get detailed Dynamo logs
# export TORCH_COMPILE_DEBUG=1  # Debug compilation issues
# export TORCH_LOGS="+dynamo,+inductor"
# export TORCHDYNAMO_VERBOSE=1
export NCCL_DEBUG=WARN

export HDF5_USE_FILE_LOCKING=FALSE
export VIRTUALIZARR_USE_HDF5_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "======== ENVIRONMENT ========"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Compiler: $(which gcc)"
echo "NVCC: $(which nvcc)"
echo "============================="

echo "Starting job."
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
date

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# srun --label --cpu-bind=v --accel-bind=v uv run inference --from_run_id pa7yzkhg -e 5 -start 2028-01-01T00:00 -end 2028-01-30T00:00 --streams_output "IFS_ATMO" "FESOM_NODE" "FESOM_ELEM" > output/eval_${SLURM_JOBID}.txt
# srun --label uv run train --config=/work/ab0995/a270225/WeatherGenerator2/config/ifs_fesom_config.yml --private_config=/work/ab0995/a270225/WeatherGenerator-private/ > output/output_${SLURM_JOBID}.txt
# srun --label uv run train --base-config=/work/ab0995/a270225/WeatherGenerator2/config/config_forecasting_eerie.yml > output/output_${SLURM_JOBID}.txt
srun --label uv run train --base-config=/work/ab0995/a270225/WeatherGenerator2/config/default_config.yml > output/output_${SLURM_JOBID}.txt
# srun --label uv run train_continue -id ubil5ary -e -1 --reuse-run-id > output/train_continue_${SLURM_JOBID}.txt
# srun --label uv run train_continue -id oorchskp -e -1 --options training_config.start_date="1950-01-02T00:00" training_config.end_date="1999-12-31T00:00" validation_config.start_date="2000-01-01T00:00" validation_config.end_date_val="2009-12-31T00:00" streams_directory="./config/streams/mesh/">  output/test_${SLURM_JOBID}.txt
# srun --label --cpu-bind=v --accel-bind=v uv run inference -id sajmept4 -e -1 -start 1996-05-01T00:00 -end 1996-06-01T00:00 --samples 8 --options streams_directory="./config/streams/mesh_downscale/" > output/inference_${SLURM_JOBID}.txt
# srun --label --cpu-bind=v --accel-bind=v uv run train_continue --from_run_id=kh3y7s28 -e -1 --options istep=0 num_epochs=32 lr_max=0.00005 freeze_modules=".*global.*|.*local.*|.*adapter.*|.*FESOM_NODE_TEMP.*|.*FESOM_ELEM_U.*|.*FESOM_NODE_SALT.*|.*FESOM_ELEM_V.*|.*IFS_ATMO.*" forecast_policy="fixed" forecast_steps=12 > output/rollout_${SLURM_JOBID}.txt
# srun --label --cpu-bind=v --accel-bind=v uv run train_continue --from_run_id=ak4r0bs6 -e -1 --reuse_run_id --options with_fsdp=False streams_directory="./config/streams/fesom_downscale/" start_date="1951-01-02T00:00" end_date="1995-12-31T00:00" start_date_val="1998-01-01T00:00" end_date_val="1999-12-31T00:00" training_mode="forecast" forecast_policy="fixed" forecast_steps=0 > output/downscale_${SLURM_JOBID}.txt
echo "Finished job."
sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
date
