#!/bin/bash -x
#SBATCH --account=weatherai
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

echo "shell $SHELL"
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi

. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
# Does not get loaded automatically. Unsure why.
. $LMOD_ROOT/lmod/init/bash
# Load basic modules from software stack
module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0

ml git/2.45.1
ml Python/3.12.3
if [ "${WEATHERGEN_NSYS_PROFILING:-0}" = "1" ]; then
    ml Nsight-Systems/2024.7.1
fi


# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export MKL_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=INFO
echo "nccl_debug: $NCCL_DEBUG"

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if WEATHERGEN_HOME is set
if [ -z "$WEATHERGEN_HOME" ]; then
    echo "WEATHERGEN_HOME is not set"
    exit 1
fi

export TMPDIR="$WEATHERGEN_HOME/tmp"
mkdir -p "$TMPDIR"


# Check if WEATHERGEN_STAGE is set
if [ -z "$WEATHERGEN_STAGE" ]; then
    echo "WEATHERGEN_STAGE is not set"
    exit 1
fi

WEATHERGEN_PRIVATE_REPO_PATH="${WEATHERGEN_PRIVATE_REPO_PATH:-$(dirname "$WEATHERGEN_HOME")/WeatherGenerator-private}"
source "${WEATHERGEN_PRIVATE_REPO_PATH}/hpc/nsys_profiling.sh"

case "$WEATHERGEN_STAGE" in

    "train"|"train_continue")
        cd $WEATHERGEN_HOME
        source .venv/bin/activate
        # Only pass --config when an extra config is actually set. On a pure
        # resume (train_continue with an empty WEATHERGEN_CONFIG_EXTRA) we must
        # NOT re-apply an extra config, otherwise it is merged on top of the
        # loaded run config and overrides the resumed run's parameters.
        CONFIG_ARG=""
        if [ -n "${WEATHERGEN_CONFIG_EXTRA}" ]; then
            CONFIG_ARG="--config ${WEATHERGEN_CONFIG_EXTRA}"
        fi
        if [ -z "$FROM_RUN_ID" ]; then
            srun --export=ALL --label ${NSYS_PREFIX} python -u ${WEATHERGEN_HOME}/src/weathergen/run_train.py train --run-id ${RUN_ID} --base-config ${WEATHERGEN_BASE_CONFIG} ${CONFIG_ARG}  &> logs/${RUN_ID}/output.${SLURM_JOBID}.txt
        else
            srun --export=ALL --label ${NSYS_PREFIX} python -u ${WEATHERGEN_HOME}/src/weathergen/run_train.py train_continue --run-id ${RUN_ID} ${CONFIG_ARG} --from-run-id ${FROM_RUN_ID} --mini-epoch ${WEATHERGEN_MINI_EPOCH:--1} &> logs/${RUN_ID}/output.${SLURM_JOBID}.txt
        fi
        echo "Finished job."
        sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
        date
        ;;

    "inference")
        if [ -z "$FROM_RUN_ID" ]; then
            echo "FROM_RUN_ID is not set (required for inference)"
            exit 1
        fi

        echo "=========================================="
        echo "WeatherGenerator Inference Configuration"
        echo "=========================================="
        echo "WEATHERGEN_HOME: $WEATHERGEN_HOME"
        echo "FROM_RUN_ID: $FROM_RUN_ID"
        echo "MINI_EPOCH: ${WEATHERGEN_MINI_EPOCH:--1}"
        echo "CONFIG_EXTRA: ${WEATHERGEN_CONFIG_EXTRA}"
        echo "=========================================="

        cd $WEATHERGEN_HOME
        source .venv/bin/activate

        # Build the inference command
        # savvas: uv run --offline inference --from-run-id czovpxgn --samples 1 --mini-epoch -1 -start 202210010000 -end 202212300000 --options forecast_steps=120
        INFERENCE_CMD="python -u ${WEATHERGEN_HOME}/src/weathergen/run_train.py inference"
        INFERENCE_CMD="$INFERENCE_CMD --from-run-id $FROM_RUN_ID"
        INFERENCE_CMD="$INFERENCE_CMD --run-id $RUN_ID"
        INFERENCE_CMD="$INFERENCE_CMD --mini-epoch ${WEATHERGEN_MINI_EPOCH:--1}"

        # Add config files if provided
        if [ -n "$WEATHERGEN_CONFIG_EXTRA" ]; then
            INFERENCE_CMD="$INFERENCE_CMD --config ${WEATHERGEN_CONFIG_EXTRA}"
        fi

        if [ -n "$WEATHERGEN_OPTIONS" ]; then
            INFERENCE_CMD="$INFERENCE_CMD --options ${WEATHERGEN_OPTIONS}"
        fi

        echo "=========================================="
        echo "Running inference command:"
        echo "$INFERENCE_CMD"
        echo "=========================================="

        # Run the inference with srun
        srun --export=ALL --label $INFERENCE_CMD &> logs/${RUN_ID}/output.inference.${SLURM_JOBID}.txt

        EXIT_CODE=$?

        echo "Finished inference job with exit code: $EXIT_CODE"
        sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
        date

        exit $EXIT_CODE
        ;;

    "evaluation")
        if [ -z "$WEATHERGEN_EVAL_RUN_IDS" ]; then
            echo "WEATHERGEN_EVAL_RUN_IDS is not set (required for evaluation)"
            exit 1
        fi

        echo "=========================================="
        echo "WeatherGenerator Evaluation Configuration"
        echo "=========================================="
        echo "WEATHERGEN_HOME:  $WEATHERGEN_HOME"
        echo "RUN_ID:           $RUN_ID"
        echo "EVAL_RUN_IDS:     $WEATHERGEN_EVAL_RUN_IDS"
        echo "EVAL_CONFIG:      ${WEATHERGEN_EVAL_CONFIG}"
        echo "EVAL_OPTIONS:     ${WEATHERGEN_EVAL_OPTIONS}"
        echo "PUSH_METRICS:     ${WEATHERGEN_EVAL_PUSH_METRICS:-0}"
        echo "=========================================="

        cd $WEATHERGEN_HOME
        source .venv/bin/activate

        # WEATHERGEN_EVAL_RUN_IDS is a space-separated list of run IDs; passed directly to --run-ids
        EVAL_CMD="python -u ${WEATHERGEN_HOME}/packages/evaluate/src/weathergen/evaluate/run_evaluation.py"
        EVAL_CMD="$EVAL_CMD --run-ids ${WEATHERGEN_EVAL_RUN_IDS}"

        # Add the main eval config if provided (--eval-config / eval_config field)
        if [ -n "$WEATHERGEN_EVAL_CONFIG" ]; then
            EVAL_CMD="$EVAL_CMD --config ${WEATHERGEN_EVAL_CONFIG}"
        fi

        # Add command-line option overrides (always present, generated by launch-slurm.py)
        if [ -n "$WEATHERGEN_EVAL_OPTIONS" ]; then
            EVAL_CMD="$EVAL_CMD --options ${WEATHERGEN_EVAL_OPTIONS}"
        fi

        if [ "${WEATHERGEN_EVAL_PUSH_METRICS:-0}" = "1" ]; then
            EVAL_CMD="$EVAL_CMD --push-metrics"
        fi

        echo "=========================================="
        echo "Running evaluation command:"
        echo "$EVAL_CMD"
        echo "=========================================="

        srun --export=ALL --label $EVAL_CMD &> logs/${RUN_ID}/output.evaluation.${SLURM_JOBID}.txt

        EXIT_CODE=$?

        echo "Finished evaluation job with exit code: $EXIT_CODE"
        sstat -j $SLURM_JOB_ID.batch --format=JobID,MaxVMSize
        date

        exit $EXIT_CODE
        ;;

    *)
        echo "WEATHERGEN_STAGE is unknown"
        exit 1
        ;;

esac
