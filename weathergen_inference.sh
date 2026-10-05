#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/error_%j.txt
#SBATCH --exclusive --mem=450G
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH -A ch17
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun uv --offline run inference --from-run-id $1  --options test_config.start_date=202301010000 test_config.end_date=202312310000 test_config.output.num_samples=1 test_config.samples_per_mini_epoch=1 test_config.forecast.num_steps=1440 test_config.forecast.chunk_size=50 test_config.output.streams=[ERA5] test_config.compute_loss=false

