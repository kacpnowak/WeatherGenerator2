#!/bin/bash
#SBATCH --job-name=paper_eval_compute
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ab0995/a270225/WeatherGenerator2/paper_eval/sbatch/compute_%j.out
cd /work/ab0995/a270225/WeatherGenerator2
# Usage: sbatch paper_eval/sbatch/compute.sbatch [config_path]
#   config_path defaults to paper_eval/configs/default.yaml
CONFIG="${1:-paper_eval/configs/default.yaml}"
.venv/bin/python -m paper_eval.compute --config "$CONFIG"
