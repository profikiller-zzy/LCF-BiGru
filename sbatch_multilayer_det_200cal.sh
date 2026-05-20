#!/bin/bash
#SBATCH --job-name=ml200
#SBATCH --partition=researchshort
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/ml200_%j_%x.out
#SBATCH --error=slurm_logs/ml200_%j_%x.err

set -euo pipefail
module load Python/3.10
module load CUDA/12.6

VENV_PATH="${VENV_PATH:-$HOME/myenv}"
source "$VENV_PATH/bin/activate"

export HF_HOME="$HOME/LCF-LLM/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

export MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
export TASK="${TASK:-negsentiment}"
export ATTACK="${ATTACK:-badnet}"
export PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-alpaca}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
export N_CALIBRATION="${N_CALIBRATION:-200}"

echo "=== Multi-Layer Detection (200 cal) ==="
echo "  Task=$TASK  Attack=$ATTACK  N_cal=$N_CALIBRATION"
echo "  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd $HOME/LCF-BiGru/attack/DPA
python multilayer_detection_200cal.py
