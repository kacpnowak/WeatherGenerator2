#!/bin/bash
#
# launch_train_auto.sh
# ---------------------
# Submit a *chain* of SLURM training jobs that automatically resume one another.
#
# Each job runs for --time (e.g. 12h). When it finishes -- whether it hits the
# wall-clock limit (TIMEOUT), completes, or crashes -- the next job starts
# automatically via a SLURM `afterany` dependency and continues training from
# the previous run's last checkpoint (train_continue, --mini-epoch -1).
#
# All jobs are submitted immediately and sit PENDING with dependencies, so the
# whole chain is queued up front. Bound the length with --resumes.
#
# ALL jobs of the chain share ONE run id, so the whole chain writes into a single
# run directory (logs/models/results) and each link resumes the checkpoint the
# previous link left behind. With --from-run-id, that existing run id is the one
# reused across every job (the chain continues the run in place) unless a
# different RUN_ID is passed explicitly, in which case the first link forks the
# existing run into RUN_ID and all later links continue RUN_ID.
#
# Usage:
#   ./launch_train_auto.sh [RUN_ID] [options]
#
# Options:
#   --config PATH        Extra config for the FIRST (fresh) job only
#                        (default: config/config_forecasting_eerie.yml)
#   --resume-config PATH Extra config to (re)apply on every RESUME link. Empty by
#                        default => pure resume: the run reloads its own saved
#                        config and nothing is overridden. Only set this if you
#                        deliberately want to override parameters on resume.
#   --resumes, -n N      Number of *additional* resume jobs after the first (default: 10)
#   --time, -t HH:MM:SS  Wall-clock per job (default: 12:00:00)
#   --from-run-id ID     Continue an EXISTING run instead of starting fresh. Its id
#                        is reused for every job unless RUN_ID is also given.
#   --mini-epoch, -e N   Checkpoint to resume from at the start (default: -1 = last)
#   --account NAME       SLURM account (default: weatherai)
#
# Examples:
#   # Fresh run, 11 links x 12h  (first + 10 resumes), all under run id my_eerie_run
#   ./launch_train_auto.sh my_eerie_run
#
#   # Continue an existing run in place for 5 more 12h links (run id stays abcd1234)
#   ./launch_train_auto.sh --from-run-id abcd1234 -n 5
#
#   # Fork an existing run into a new id, then keep resuming that new id
#   ./launch_train_auto.sh my_fork --from-run-id abcd1234 -n 5
#
#   # GLORYS 1-degree config, 3 resumes
#   ./launch_train_auto.sh glorys1deg_run --config config/config_forecasting_glorys.yml -n 3
#
# Cancel the whole chain later with:  scancel <first_jobid>..<last_jobid>  (or by name)

set -euo pipefail

# ---- Defaults ----
CONFIG_EXTRA="config/config_forecasting_eerie.yml"
RESUME_CONFIG=""          # empty => pure resume (no config override on continues)
RESUMES=10                # additional jobs after the first
TIME="12:00:00"
ACCOUNT="weatherai"
BASE_RUN_ID=""
FROM_RUN_ID=""            # non-empty => continue an existing run
MINI_EPOCH="-1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)         CONFIG_EXTRA="$2"; shift 2 ;;
    --resume-config)  RESUME_CONFIG="$2"; shift 2 ;;
    --resumes|-n)     RESUMES="$2"; shift 2 ;;
    --time|-t)        TIME="$2"; shift 2 ;;
    --from-run-id)    FROM_RUN_ID="$2"; shift 2 ;;
    --mini-epoch|-e)  MINI_EPOCH="$2"; shift 2 ;;
    --account)        ACCOUNT="$2"; shift 2 ;;
    -h|--help)        sed -n '2,52p' "$0"; exit 0 ;;
    -*)               echo "Unknown option: $1" >&2; exit 1 ;;
    *)                [ -z "$BASE_RUN_ID" ] && BASE_RUN_ID="$1"; shift ;;
  esac
done

# One run id for the whole chain. If the user only gave --from-run-id, that id is
# reused for every job so the chain continues the existing run in place. If a
# RUN_ID was passed explicitly it wins: the first link forks --from-run-id into it
# and every later link continues RUN_ID itself.
if [ -z "$BASE_RUN_ID" ]; then
  if [ -n "$FROM_RUN_ID" ]; then
    BASE_RUN_ID="$FROM_RUN_ID"
  else
    BASE_RUN_ID="train_auto_$(date +%Y%m%d_%H%M%S)"
  fi
fi

RUN_ID_SOURCE="$FROM_RUN_ID"   # what the FIRST link continues from ("" => fresh train)

# ---- Environment expected by weathergen_slurm_local.sh ----
export WEATHERGEN_HOME="/e/scratch/hclimrep/nowak2/WeatherGenerator2"
export WEATHERGEN_PRIVATE_REPO_PATH="/e/scratch/hclimrep/nowak2/WeatherGenerator-private"
export WEATHERGEN_BASE_CONFIG="config/default_config.yml"
export WEATHERGEN_MINI_EPOCH="$MINI_EPOCH"
# WEATHERGEN_CONFIG_EXTRA is set per-link inside the loop: the fresh job gets
# CONFIG_EXTRA, resume links get RESUME_CONFIG (empty => pure resume).

SLURM_SCRIPT="$WEATHERGEN_HOME/weathergen_slurm_local.sh"

echo "=================================================="
echo "Auto-resume training chain"
echo "RUN_ID (all jobs): $BASE_RUN_ID"
echo "CONFIG_EXTRA:      $CONFIG_EXTRA"
echo "TIME/job:          $TIME"
echo "LINKS:             $((RESUMES + 1))  (first + $RESUMES resumes)"
echo "START FROM:        ${RUN_ID_SOURCE:-<fresh train>}"
echo "=================================================="

# Every link runs under the same run id, so a single run directory is used and
# each job picks up the checkpoint the previous one wrote.
export RUN_ID="$BASE_RUN_ID"
mkdir -p "$WEATHERGEN_HOME/logs/$RUN_ID"

prev_jobid=""

for (( i=0; i<=RESUMES; i++ )); do
  if [ "$i" -eq 0 ] && [ -z "$RUN_ID_SOURCE" ]; then
    # Fresh training: apply the extra config to define the run.
    export WEATHERGEN_STAGE="train"
    export FROM_RUN_ID=""
    export WEATHERGEN_CONFIG_EXTRA="$CONFIG_EXTRA"
  else
    export WEATHERGEN_STAGE="train_continue"
    if [ "$i" -eq 0 ]; then
      export FROM_RUN_ID="$RUN_ID_SOURCE"  # continue the user-supplied run
      export WEATHERGEN_MINI_EPOCH="$MINI_EPOCH"
    else
      export FROM_RUN_ID="$RUN_ID"         # continue this same run in place
      # After the first resume, always resume from the last checkpoint.
      export WEATHERGEN_MINI_EPOCH="-1"
    fi
    # Pure resume by default: no extra config re-applied on top of the loaded
    # run config. Only set if --resume-config was given (deliberate override).
    export WEATHERGEN_CONFIG_EXTRA="$RESUME_CONFIG"
  fi

  # Suffix the job name so the chain links stay distinguishable in squeue even
  # though they share one run id.
  job_name="${RUN_ID}$([ "$i" -eq 0 ] && echo "" || echo "_c${i}")"

  if [ -z "$prev_jobid" ]; then
    jobid=$(sbatch --parsable \
                   --account="$ACCOUNT" --time="$TIME" \
                   --job-name="$job_name" \
                   "$SLURM_SCRIPT")
  else
    jobid=$(sbatch --parsable \
                   --account="$ACCOUNT" --time="$TIME" \
                   --job-name="$job_name" \
                   --dependency=afterany:"$prev_jobid" \
                   "$SLURM_SCRIPT")
  fi

  printf '[%2d] job %-12s run_id=%-28s stage=%-14s from=%-12s cfg=%-40s dep=%s\n' \
         "$i" "$jobid" "$RUN_ID" "$WEATHERGEN_STAGE" \
         "${FROM_RUN_ID:-<none>}" "${WEATHERGEN_CONFIG_EXTRA:-<none / pure resume>}" \
         "${prev_jobid:-none}"

  prev_jobid="$jobid"
done

echo "=================================================="
echo "Submitted $((RESUMES + 1)) chained jobs. Watch with:  squeue --me"
echo "=================================================="
