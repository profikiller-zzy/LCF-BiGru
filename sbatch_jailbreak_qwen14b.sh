#!/bin/bash
#SBATCH --job-name=jbd14
#SBATCH --partition=researchshort
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/jbd14_%j_%x.out
#SBATCH --error=slurm_logs/jbd14_%j_%x.err

set -euo pipefail
module load Python/3.10
module load CUDA/12.6

VENV_PATH="${VENV_PATH:-$HOME/myenv}"
source "$VENV_PATH/bin/activate"

export HF_HOME="$HOME/LCF-LLM/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}"
export PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-alpaca}"
export N_CALIBRATION="${N_CALIBRATION:-200}"

echo "=== Jailbreak Detection LCF probe ==="
echo "  Model: $MODEL_PATH  Template: $PROMPT_TEMPLATE  N_cal: $N_CALIBRATION"
echo "  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd $HOME/LCF-BiGru/attack/DPA
python jailbreak_detection_experiment.py
