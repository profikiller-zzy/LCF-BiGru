#!/bin/bash
#SBATCH --job-name=prefetch-qwen14b
#SBATCH --partition=researchshort
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/prefetch_%j.out
#SBATCH --error=slurm_logs/prefetch_%j.err

set -euo pipefail
module load Python/3.10 2>/dev/null || module load Python/3.10.16-GCCcore-13.3.0

VENV_PATH="${VENV_PATH:-$HOME/myenv}"
source "$VENV_PATH/bin/activate"

export HF_HOME="$HOME/LCF-LLM/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

echo "Downloading Qwen/Qwen2.5-14B-Instruct to $HF_HOME ..."
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --resume-download

echo "Download complete. Cache size:"
du -sh "$HF_HOME" || true
