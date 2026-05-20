#!/bin/bash
#SBATCH --job-name=prinj
#SBATCH --partition=researchshort
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="32gb|40gb|48gb|80gb|96gb|141gb"
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/prinj_%j_%x.out
#SBATCH --error=slurm_logs/prinj_%j_%x.err

set -euo pipefail
module load Python/3.10
module load CUDA/12.6

VENV_PATH="${VENV_PATH:-$HOME/myenv}"
source "$VENV_PATH/bin/activate"

export HF_HOME="$HOME/LCF-LLM/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
export PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-alpaca}"
export N_CALIBRATION="${N_CALIBRATION:-200}"

echo "=== LCF Prompt Injection Probe ==="
echo "  Model:    $MODEL_PATH"
echo "  Template: $PROMPT_TEMPLATE"
echo "  N_cal:    $N_CALIBRATION"
echo "  Node:     $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd $HOME/LCF-BiGru/attack/DPA
python prompt_injection_probe.py
