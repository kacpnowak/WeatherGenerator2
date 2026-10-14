#!/bin/bash
#SBATCH --job-name=paper_eval_spectra
#SBATCH --account=ab0995
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/work/ab0995/a270225/WeatherGenerator2/paper_eval/sbatch/spectra_%j.out
cd /work/ab0995/a270225/WeatherGenerator2
# Usage: sbatch paper_eval/sbatch/spectra.sbatch [config_path]
#   config_path defaults to paper_eval/configs/default.yaml
CONFIG="${1:-paper_eval/configs/default.yaml}"
.venv/bin/python -m paper_eval.compute_spectra --config "$CONFIG"
