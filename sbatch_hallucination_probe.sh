#!/bin/bash
#SBATCH --job-name=hallu
#SBATCH --partition=researchshort
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/hallu_%j_%x.out
#SBATCH --error=slurm_logs/hallu_%j_%x.err

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
export N_CAL="${N_CAL:-200}"
export N_PER_CATEGORY="${N_PER_CATEGORY:-100}"

echo "=== LCF Hallucination Probe (PopQA) ==="
echo "  Model:    $MODEL_PATH"
echo "  Template: $PROMPT_TEMPLATE"
echo "  N_cal:    $N_CAL"
echo "  N/cat:    $N_PER_CATEGORY"
echo "  Node:     $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd $HOME/LCF-LLM/attack/DPA
python hallucination_probe.py
