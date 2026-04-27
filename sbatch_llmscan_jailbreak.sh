#!/bin/bash
#SBATCH --job-name=llmscan_jb
#SBATCH --partition=researchshort
#SBATCH --time=06:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/llmscan_jb_%j_%x.out
#SBATCH --error=slurm_logs/llmscan_jb_%j_%x.err

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
export N_CLEAN="${N_CLEAN:-200}"
export MAX_TOKEN_POS="${MAX_TOKEN_POS:-64}"
export SEED="${SEED:-0}"

echo "=== LLMScan jailbreak baseline ==="
echo "  Model:           $MODEL_PATH"
echo "  Prompt template: $PROMPT_TEMPLATE"
echo "  N_CLEAN:         $N_CLEAN"
echo "  MAX_TOKEN_POS:   $MAX_TOKEN_POS"
echo "  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

cd $HOME/LCF-LLM/attack/DPA
python -m baselines.llmscan.run_jailbreak
