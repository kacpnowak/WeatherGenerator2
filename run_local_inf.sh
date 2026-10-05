#!/bin/bash
source .venv/bin/activate
export WEATHERGEN_PRIVATE_REPO_PATH="/e/scratch/hclimrep/nowak2/WeatherGenerator-private"
export MASTER_ADDR=localhost
export MASTER_PORT=29503

python -u src/weathergen/run_train.py inference \
    --from-run-id train_eerie_20260617_222101 \
    --run-id local_eval_run_4 \
    --mini-epoch -1 \
    --config config/config_forecasting_glorys.yml \
    --options test_config.start_date=2024-01-01T00:00 test_config.end_date=2025-01-01T00:00 streams_directory=./config/streams/glorys_eval/ test_config.output.num_samples=1 test_config.samples_per_mini_epoch=1 test_config.output.streams=null
