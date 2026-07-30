#!/bin/bash
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=obench-inference
#SBATCH --output=logs/oceanbench/%j.out
#SBATCH --error=logs/oceanbench/%j.err
#
# run_oceanbench_inference.sh
#
# Driver for the 52 OceanBench single-sample WeatherGenerator inference runs
# (one per Wednesday forecast-start date of 2024). Sequentially runs one
# inference per requested date, all inside a single slurm job/GPU.
#
# Why "Monday start, 1 sample": the sampler maps sample k <-> window idx
# k+1 <-> start_date + (k+1)*24h (unshuffled, no weekly-stride option), so
# setting start_date to the Monday before each target Wednesday and taking
# exactly one sample makes the source window land on the Tuesday - the day
# the nowcast IC parquet actually has data for.
#
# Usage:
#   ./run_oceanbench_inference.sh [--dates "YYYY-MM-DD ..."] [--dry-run]
#
#   --dates    space-separated list of Wednesday dates to process, overriding
#              the default of all 52 Wednesdays of 2024.
#   --dry-run  print the RUN_ID / MON / END / full command for each requested
#              date and exit; nothing is submitted or run.
#
# Without --dry-run, this script submits exactly ONE sbatch job (itself) that
# then loops over all requested dates sequentially on one GPU.

# Note: deliberately no `-u` (nounset). weathergen_slurm_local.sh's env setup
# (.bashrc, lmod init) is not nounset-safe, and this script already guards
# every variable it reads before it is assigned (e.g. ${SLURM_JOB_ID:-}).
set -o pipefail

REPO_DIR="/e/scratch/hclimrep/nowak2/WeatherGenerator2"
RESULTS_DIR="/e/scratch/weatherai/shared_work/results"
PRIVATE_REPO_PATH="/e/scratch/hclimrep/nowak2/WeatherGenerator-private"

usage() {
    echo "Usage: $0 [--dates \"YYYY-MM-DD ...\"] [--dry-run]" >&2
}

DRY_RUN=0
DATES_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --dates)
            if [[ $# -lt 2 ]]; then
                echo "--dates requires an argument" >&2
                exit 1
            fi
            DATES_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Build the list of forecast-start Wednesdays.
# Default: all 52 Wednesdays of 2024, generated programmatically as
# 2024-01-03 + 7*k days, k=0..51 (2024-01-03 was a Wednesday; 52 weeks later
# is 2024-12-25, still within 2024).
# ---------------------------------------------------------------------------
WEDNESDAYS=()
if [[ -n "$DATES_ARG" ]]; then
    read -r -a WEDNESDAYS <<< "$DATES_ARG"
else
    for k in $(seq 0 51); do
        WEDNESDAYS+=("$(date -u -d "2024-01-03 +$((k * 7)) days" +%Y-%m-%d)")
    done
fi

# Compute MON (start_date), END (end_date) and RUN_ID for a Wednesday date $1,
# and build the exact inference command. Sets globals MON, END, RUN_ID, CMD.
build_command() {
    local W="$1"
    MON=$(date -u -d "${W} -2 days" +%Y-%m-%d)
    END=$(date -u -d "${MON} +15 days" +%Y-%m-%d)
    RUN_ID="obench_${W//-/}"
    # NOTE: built as a single-line string rather than via a `\`-continued
    # heredoc: `$(cat <<EOF ... \<newline> ... EOF)` silently swallows
    # backslash-newline pairs (verified empirically), corrupting the
    # captured command. A single line is semantically identical and safe.
    CMD="python -u src/weathergen/run_train.py inference --from-run-id glorys_cont3_c1 --run-id ${RUN_ID} --mini-epoch -1 --config config/config_forecasting_glorys.yml --options streams_directory=./config/streams/glorys_eval/ test_config.start_date=${MON}T00:00 test_config.end_date=${END}T00:00 test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1 test_config.forecast.num_steps=10 \"test_config.output.streams=[GLORYS]\""
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    for W in "${WEDNESDAYS[@]}"; do
        build_command "$W"
        echo "=== W=${W} ==="
        echo "RUN_ID: ${RUN_ID}"
        echo "MON:    ${MON}"
        echo "END:    ${END}"
        echo "${CMD}"
        echo
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Not a dry run. If we're not already running inside the slurm job (i.e. this
# is the interactive/login invocation), submit ourselves as a single sbatch
# job carrying the same date selection, and exit. The actual work happens in
# the branch below, which only runs once slurm re-invokes this same script as
# the batch job.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "${REPO_DIR}/logs/oceanbench"
    echo "Submitting sbatch job for ${#WEDNESDAYS[@]} date(s)..."
    if [[ -n "$DATES_ARG" ]]; then
        sbatch "${REPO_DIR}/run_oceanbench_inference.sh" --dates "$DATES_ARG"
    else
        sbatch "${REPO_DIR}/run_oceanbench_inference.sh"
    fi
    exit $?
fi

# ---------------------------------------------------------------------------
# Inside the slurm job: set up the environment and run each date sequentially.
# Environment setup mirrors weathergen_slurm_local.sh's module load / venv
# activation pattern, adapted for a single-GPU, single-node inference job
# (no multi-node InfiniBand/NCCL setup is needed here).
# ---------------------------------------------------------------------------
echo "=========================================="
echo "OceanBench inference driver"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Dates to process: ${#WEDNESDAYS[@]}"
echo "=========================================="

echo "shell $SHELL"
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi
. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
. "$LMOD_ROOT/lmod/init/bash"
module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0

ml git/2.45.1
ml Python/3.12.3

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_DIR"
source .venv/bin/activate
export WEATHERGEN_PRIVATE_REPO_PATH="$PRIVATE_REPO_PATH"
export MASTER_ADDR=localhost
export MASTER_PORT=29511

FAILED=()
for W in "${WEDNESDAYS[@]}"; do
    build_command "$W"
    OUT_ZIP="${RESULTS_DIR}/${RUN_ID}/validation_chkpt00000_rank0000.zip"

    if [[ -f "$OUT_ZIP" ]]; then
        echo "[SKIP] ${W} (RUN_ID=${RUN_ID}): output already exists at ${OUT_ZIP}"
        continue
    fi

    echo "------------------------------------------"
    echo "[RUN] ${W}  RUN_ID=${RUN_ID}  MON=${MON}  END=${END}"
    echo "${CMD}"
    echo "------------------------------------------"

    eval "$CMD"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[FAIL] ${W} (RUN_ID=${RUN_ID}) exited with code ${rc}"
        FAILED+=("$W")
    else
        echo "[OK] ${W} (RUN_ID=${RUN_ID})"
    fi
done

echo "=========================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED dates (${#FAILED[@]}): ${FAILED[*]}"
    echo "=========================================="
    exit 1
fi
echo "All dates completed successfully."
echo "=========================================="
exit 0
