#!/bin/bash
#SBATCH --job-name=qwen14b-eval
#SBATCH --partition=researchshort
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/eval14b_%j_%x.out
#SBATCH --error=slurm_logs/eval14b_%j_%x.err

set -euo pipefail
module load Python/3.10
module load CUDA/12.6

VENV_PATH="${VENV_PATH:-$HOME/myenv}"
source "$VENV_PATH/bin/activate"

export HF_HOME="$HOME/LCF-LLM/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

export MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
export MODEL_NAME="Qwen2.5-14B-Instruct"
export TASKS="${TASKS:-negsentiment}"
export ATTACKS="${ATTACKS:-badnet}"
export PROMPT_TEMPLATE="alpaca"
export RUNTIME_MODE="baseline"
export OUTPUT_TAG="baseline"
export FORCE_SINGLE_SLICE=1

cd $HOME/LCF-LLM/attack/DPA
python eval_backdoor.py
