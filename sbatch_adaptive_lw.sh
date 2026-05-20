#!/bin/bash
#SBATCH --job-name=adap-lw
#SBATCH --partition=researchshort
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/adap_lw_%j.out
#SBATCH --error=slurm_logs/adap_lw_%j.err

set -euo pipefail
module load Python/3.10
module load CUDA/12.6
source $HOME/myenv/bin/activate
export HF_HOME="$HOME/LCF-LLM/.cache/huggingface"

export MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
export TASK="${TASK:-negsentiment}"
export ATTACK="${ATTACK:-badnet}"
export RV_LAMBDA="${RV_LAMBDA:-0.0}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"

echo "=== Adaptive LW Attack: lambda=$RV_LAMBDA task=$TASK attack=$ATTACK ==="
cd $HOME/LCF-BiGru/attack/DPA
python adaptive_attack_lw.py
