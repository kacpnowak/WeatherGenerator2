#!/bin/bash

# Default values
DEFAULT_RUN_ID="inference_eerie_$(date +%Y%m%d_%H%M%S)"

FROM_RUN_ID=""
RUN_ID=""
FORECAST_STEPS=""

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
    --forecast-steps)
      FORECAST_STEPS="$2"
      shift 2
      ;;
    --save-samples)
      SAVE_SAMPLES="$2"
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

if [ -z "$FROM_RUN_ID" ]; then
    echo "Error: --from-run-id is required for inference."
    exit 1
fi

# Set environment variables required by weathergen_slurm.sh
# Use /e/scratch paths as /p/project1 is not mounted on compute nodes
export WEATHERGEN_HOME="/e/scratch/hclimrep/nowak2/WeatherGenerator2"
export WEATHERGEN_PRIVATE_REPO_PATH="/e/scratch/hclimrep/nowak2/WeatherGenerator-private"
export WEATHERGEN_BASE_CONFIG="config/default_config.yml"
# export WEATHERGEN_CONFIG_EXTRA="config/config_forecasting_eerie.yml"
export RUN_ID="$RUN_ID"

export WEATHERGEN_STAGE="inference"
export FROM_RUN_ID="$FROM_RUN_ID"

WEATHERGEN_OPTIONS="test_config.start_date=2208-01-01T12:00 test_config.end_date=2209-01-31T12:00 test_config.time_window_step=360:00:00"
if [ -n "$FORECAST_STEPS" ]; then
    WEATHERGEN_OPTIONS="$WEATHERGEN_OPTIONS test_config.forecast.num_steps=$FORECAST_STEPS"
else
    WEATHERGEN_OPTIONS="$WEATHERGEN_OPTIONS test_config.forecast.num_steps=30"
fi

if [ -n "$SAVE_SAMPLES" ]; then
    if [ -n "$WEATHERGEN_OPTIONS" ]; then
        WEATHERGEN_OPTIONS="$WEATHERGEN_OPTIONS test_config.output.num_samples=$SAVE_SAMPLES test_config.samples_per_mini_epoch=$SAVE_SAMPLES"
    else
        WEATHERGEN_OPTIONS="test_config.output.num_samples=$SAVE_SAMPLES test_config.samples_per_mini_epoch=$SAVE_SAMPLES"
    fi
else
    # Default to save 1 sample if not specified, otherwise it saves 0 and produces no output
    if [ -n "$WEATHERGEN_OPTIONS" ]; then
        WEATHERGEN_OPTIONS="$WEATHERGEN_OPTIONS test_config.output.num_samples=365 test_config.samples_per_mini_epoch=365"
    else
        WEATHERGEN_OPTIONS="test_config.output.num_samples=365 test_config.samples_per_mini_epoch=365"
    fi
fi
export WEATHERGEN_OPTIONS

# Create log directory for the run if it doesn't exist
mkdir -p "$WEATHERGEN_HOME/logs/$RUN_ID"

echo "=================================================="
echo "Launching WeatherGenerator Inference"
echo "STAGE:          $WEATHERGEN_STAGE"
echo "RUN_ID:         $RUN_ID"
echo "FROM_RUN_ID:    $FROM_RUN_ID"
echo "FORECAST_STEPS: ${FORECAST_STEPS:-default}"
echo "BASE_CONFIG:    $WEATHERGEN_BASE_CONFIG"
# echo "EXTRA_CONFIG:   $WEATHERGEN_CONFIG_EXTRA"
echo "HOME:           $WEATHERGEN_HOME"
echo "PRIVATE:        $WEATHERGEN_PRIVATE_REPO_PATH"
echo "=================================================="

sbatch --account=hclimrep --time=12:00:00 --nodes=1 --ntasks=1 --gres=gpu:1 "$WEATHERGEN_HOME/weathergen_slurm_local.sh"
