#!/bin/bash
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=obench-inference
#SBATCH --output=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/oceanbench/%j.out
#SBATCH --error=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/oceanbench/%j.err
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
# The default run config is config/config_forecasting_glorys_obench.yml, a
# frozen snapshot of what the submitted evaluation used. Do NOT repoint this at
# config/config_forecasting_glorys.yml: that file tracks the current training
# experiment and moves independently of this evaluation. A different model
# checkpoint may need its own frozen config; pass it via --eval-config instead
# of editing this default.
#
# Usage:
#   ./run_oceanbench_inference.sh [--dates "YYYY-MM-DD ..."] [--dry-run]
#                                 [--model <run_id>] [--run-prefix <prefix>]
#                                 [--eval-config <path>]
#
#   --dates        space-separated list of Wednesday dates to process,
#                  overriding the default of all 52 Wednesdays of 2024.
#   --dry-run      print the RUN_ID / MON / END / full command for each
#                  requested date and exit; nothing is submitted or run.
#   --model        checkpoint run-id to evaluate (passed as --from-run-id to
#                  inference). Default: glorys_cont3_c1.
#   --run-prefix   prefix used to build each per-date RUN_ID (RUN_ID =
#                  <prefix><YYYYMMDD>) and, correspondingly, the per-date
#                  results directory checked for the skip-if-exists guard.
#                  Default: "obench_" when --model is left at its default
#                  (preserves today's existing run-ids exactly), otherwise
#                  "obench_<model>_".
#   --eval-config  frozen run config passed as --config to inference. Default:
#                  config/config_forecasting_glorys_obench.yml. Use this for a
#                  model that needs a different frozen config (e.g. a
#                  different channel set) instead of repointing the default.
#
# NOTE: --dates is passed through to `date -d` and eval'd as part of the
# inference command with no sanitization. This is an internal operator
# launcher, not a user-facing service, so untrusted input is out of scope;
# only pass trusted YYYY-MM-DD values. --model/--run-prefix/--eval-config are
# spliced into that same eval'd command too; unlike --dates they are quoted
# with `printf %q` before being embedded (see build_command), so a value with
# spaces or shell-special characters is preserved as one argument rather than
# corrupting the eval'd command -- but still only pass trusted values.
#
# Without --dry-run, this script submits exactly ONE sbatch job (itself) that
# then loops over all requested dates sequentially on one GPU.
#
# ---------------------------------------------------------------------------
# RUNBOOK: evaluating a NEW model checkpoint end to end
# ---------------------------------------------------------------------------
# 0. Prerequisites -- channel set must match the checkpoint. Verify with the
#    verify_task1.py pattern (.superpowers/sdd/wondrous-coalescing-taco/
#    verify_task1.py): instantiate DataReaderMesh on the intended
#    config/streams/<eval-dir>/ and diff source_channels/target_channels
#    against the checkpoint's model_<run_id>_latest.json "streams" block.
#    - A model trained on the current 41-channel glorys config matches
#      config/streams/glorys_eval (what glorys_cont3_c1 uses today).
#    - A model trained on the v2 133-channel config
#      (config/streams/glorys_v2/glorys.yml) needs config/streams/glorys_v2_eval
#      -- but that eval stream currently points at a v3 IC parquet
#      (glo_nowcast_025_v3_fixed.parq) that has NOT been built yet (see the
#      loud warning in config/streams/glorys_v2_eval/glorys.yml). That
#      parquet must be built (Task 1c pipeline: add ice channels + full
#      <=541.089m depth axis) before glorys_v2_eval can be used for real.
# 1. Driver: ./run_oceanbench_inference.sh --model <run_id> \
#      [--run-prefix <prefix>] [--eval-config <path>] [--dates "..."]
#    Dry-run first (--dry-run) to sanity-check RUN_ID/MON/END/CMD.
# 2. Export: python export_glorys_nc.py --run-prefix <same prefix as above> \
#      --output <a NEW zarr path, e.g. .../eval_output/wg_<run_id>_challenger.zarr>
#    (--output is required whenever --run-prefix is non-default; see that
#    script's own docstring/CLI help.)
# 3. Point the challenger at the new store, then run OceanBench in-kernel:
#      export WG_CHALLENGER_STORE=<the --output path from step 2>
#      (challenger.py / challenger_smoke.py read this env var, falling back
#      to the historical glorys_cont3_c1 store if it is unset)
# 4. sbatch run_oceanbench_eval.sh <challenger.py> [STAGE_DIR]
#    run_oceanbench_eval.sh itself lives in and is submitted from
#    /e/scratch/hclimrep/nowak2/eval_output, but challenger.py /
#    challenger_smoke.py live at THIS repo's root, not in eval_output.
#    run_oceanbench_eval.sh cd's into eval_output before invoking `oceanbench
#    evaluate`, so the challenger argument must be an absolute path -- a bare
#    "challenger.py" would resolve against eval_output post-cd and fail right
#    after the egress probe. Compute nodes are offline (no egress), so pass
#    the pre-populated stage dir as STAGE_DIR to run in offline mode. Exact
#    invocation (matches every historical successful job log):
#      cd /e/scratch/hclimrep/nowak2/eval_output && sbatch run_oceanbench_eval.sh \
#        /e/scratch/hclimrep/nowak2/WeatherGenerator2/challenger.py \
#        /e/scratch/hclimrep/nowak2/oceanbench_stage

# Note: deliberately no `-u` (nounset). weathergen_slurm_local.sh's env setup
# (.bashrc, lmod init) is not nounset-safe, and this script already guards
# every variable it reads before it is assigned (e.g. ${SLURM_JOB_ID:-}).
set -o pipefail

REPO_DIR="/e/scratch/hclimrep/nowak2/WeatherGenerator2"
RESULTS_DIR="/e/scratch/weatherai/shared_work/results"
PRIVATE_REPO_PATH="/e/scratch/hclimrep/nowak2/WeatherGenerator-private"

usage() {
    echo "Usage: $0 [--dates \"YYYY-MM-DD ...\"] [--dry-run] [--model <run_id>]" >&2
    echo "          [--run-prefix <prefix>] [--eval-config <path>]" >&2
}

DRY_RUN=0
DATES_ARG=""
MODEL="glorys_cont3_c1"
MODEL_SET=0
RUN_PREFIX=""
RUN_PREFIX_SET=0
EVAL_CONFIG="config/config_forecasting_glorys_obench.yml"
EVAL_CONFIG_SET=0

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
        --model)
            if [[ $# -lt 2 ]]; then
                echo "--model requires an argument" >&2
                exit 1
            fi
            MODEL="$2"
            MODEL_SET=1
            shift 2
            ;;
        --run-prefix)
            if [[ $# -lt 2 ]]; then
                echo "--run-prefix requires an argument" >&2
                exit 1
            fi
            RUN_PREFIX="$2"
            RUN_PREFIX_SET=1
            shift 2
            ;;
        --eval-config)
            if [[ $# -lt 2 ]]; then
                echo "--eval-config requires an argument" >&2
                exit 1
            fi
            EVAL_CONFIG="$2"
            EVAL_CONFIG_SET=1
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

# Resolve the default run-prefix now that --model/--run-prefix have been
# parsed: preserves today's existing run-ids exactly ("obench_<YYYYMMDD>")
# when --model is left at its default, and disambiguates per-model runs
# ("obench_<model>_<YYYYMMDD>") when --model is given explicitly.
if [[ "$RUN_PREFIX_SET" -eq 0 ]]; then
    if [[ "$MODEL_SET" -eq 1 ]]; then
        RUN_PREFIX="obench_${MODEL}_"
    else
        RUN_PREFIX="obench_"
    fi
fi

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
    RUN_ID="${RUN_PREFIX}${W//-/}"
    # MODEL/RUN_ID/EVAL_CONFIG can carry user-supplied content (--model,
    # --run-prefix, --eval-config), unlike the rest of this command whose
    # variable pieces are either digits-only (${W//-/}) or literals. Quote
    # them with `printf %q` before splicing into CMD, which is later eval'd:
    # %q only adds quoting/escaping when the value actually needs it, so the
    # default (safe) values expand to the exact same bare text as before --
    # the dry-run output for default arguments is unchanged -- while a value
    # containing spaces or shell-special characters survives eval as a
    # single argument instead of silently corrupting the command.
    local model_q run_id_q eval_config_q
    printf -v model_q '%q' "${MODEL}"
    printf -v run_id_q '%q' "${RUN_ID}"
    printf -v eval_config_q '%q' "${EVAL_CONFIG}"
    # NOTE: built as a single-line string rather than via a `\`-continued
    # heredoc: `$(cat <<EOF ... \<newline> ... EOF)` silently swallows
    # backslash-newline pairs (verified empirically), corrupting the
    # captured command. A single line is semantically identical and safe.
    CMD="python -u src/weathergen/run_train.py inference --from-run-id ${model_q} --run-id ${run_id_q} --mini-epoch -1 --config ${eval_config_q} --options streams_directory=./config/streams/glorys_eval/ test_config.start_date=${MON}T00:00 test_config.end_date=${END}T00:00 test_config.samples_per_mini_epoch=1 test_config.output.num_samples=1 test_config.forecast.num_steps=10 \"test_config.output.streams=[GLORYS]\""
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
    SBATCH_ARGS=()
    if [[ -n "$DATES_ARG" ]]; then
        SBATCH_ARGS+=(--dates "$DATES_ARG")
    fi
    if [[ "$MODEL_SET" -eq 1 ]]; then
        SBATCH_ARGS+=(--model "$MODEL")
    fi
    if [[ "$RUN_PREFIX_SET" -eq 1 ]]; then
        SBATCH_ARGS+=(--run-prefix "$RUN_PREFIX")
    fi
    if [[ "$EVAL_CONFIG_SET" -eq 1 ]]; then
        SBATCH_ARGS+=(--eval-config "$EVAL_CONFIG")
    fi
    sbatch "${REPO_DIR}/run_oceanbench_inference.sh" "${SBATCH_ARGS[@]}"
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
# Neutralize breakpoint() in batch mode. NOTE: this does NOT disable the explicit
# pdb.post_mortem in run_train.py - the OUT_ZIP existence check below is the real guard.
export PYTHONBREAKPOINT=0

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
    # A zero exit code is NOT sufficient evidence of success: run_train.py wraps
    # inference in pdb.post_mortem, so a crash in batch mode (no tty) drops into pdb,
    # gets EOF, and the process still exits 0 with no output written. Require the
    # result zip to actually exist.
    if [[ $rc -ne 0 ]]; then
        echo "[FAIL] ${W} (RUN_ID=${RUN_ID}) exited with code ${rc}"
        FAILED+=("$W")
    elif [[ ! -f "$OUT_ZIP" ]]; then
        echo "[FAIL] ${W} (RUN_ID=${RUN_ID}) exited 0 but produced no ${OUT_ZIP}"
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
