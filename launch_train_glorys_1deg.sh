#!/bin/bash

# Default values
DEFAULT_RUN_ID="train_glorys1deg_$(date +%Y%m%d_%H%M%S)"

FROM_RUN_ID=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --from-run-id)
      FROM_RUN_ID="$2"
      shift 2
      ;;
    --mini-epoch|-e)
      export WEATHERGEN_MINI_EPOCH="$2"
      shift 2
      ;;
    *)
      if [ -z "$RUN_ID" ]; then
        RUN_ID="$1"
      fi
      shift
      ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  RUN_ID="$DEFAULT_RUN_ID"
fi

# Set environment variables required by weathergen_slurm_local.sh
export WEATHERGEN_HOME="/e/scratch/hclimrep/nowak2/WeatherGenerator2"
export WEATHERGEN_PRIVATE_REPO_PATH="/e/scratch/hclimrep/nowak2/WeatherGenerator-private"
export WEATHERGEN_BASE_CONFIG="config/default_config.yml"
export WEATHERGEN_CONFIG_EXTRA="config/config_forecasting_glorys.yml"
export RUN_ID="$RUN_ID"

if [ -n "$FROM_RUN_ID" ]; then
    export WEATHERGEN_STAGE="train_continue"
    export FROM_RUN_ID="$FROM_RUN_ID"
else
    export WEATHERGEN_STAGE="train"
fi

# Create log directory for the run if it doesn't exist
mkdir -p "$WEATHERGEN_HOME/logs/$RUN_ID"

echo "=================================================="
echo "Launching WeatherGenerator Training for GLORYS 1-Degree"
echo "STAGE:       $WEATHERGEN_STAGE"
echo "RUN_ID:      $RUN_ID"
echo "BASE_CONFIG: $WEATHERGEN_BASE_CONFIG"
echo "EXTRA_CONFIG: $WEATHERGEN_CONFIG_EXTRA"
echo "HOME:        $WEATHERGEN_HOME"
echo "PRIVATE:     $WEATHERGEN_PRIVATE_REPO_PATH"
echo "=================================================="

sbatch --account=weatherai --time=00:30:00 "$WEATHERGEN_HOME/weathergen_slurm_local.sh"
