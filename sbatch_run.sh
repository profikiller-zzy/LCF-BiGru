#!/bin/bash

#################################################
## BackdoorLLM SBATCH TRAINING SCRIPT          ##
#################################################

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|a100|h100nvl|v100-32gb
#SBATCH --time=02-00:00:00
#SBATCH --partition=researchshort
#SBATCH --account=sunjunresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=backdoor-train
#SBATCH --output=slurm-%A-%x.out
#SBATCH --error=slurm-%A-%x.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "This script must be submitted via sbatch."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
if [ ! -d "$WORKDIR/attack/DPA" ]; then
  echo "WORKDIR must point to the BackdoorLLM repo root (missing attack/DPA): $WORKDIR"
  exit 1
fi
cd "$WORKDIR"

LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  echo "Warning: cannot create log dir at $LOG_DIR. Falling back to $HOME/backdoorllm_logs"
  LOG_DIR="$HOME/backdoorllm_logs"
  mkdir -p "$LOG_DIR"
fi

# Module setup (override these at submit-time if needed)
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.6.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.10.16-GCCcore-13.3.0}"
CUDNN_MODULE="${CUDNN_MODULE:-}"
NCCL_MODULE="${NCCL_MODULE:-}"
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "$PYTHON_MODULE" || {
    echo "Failed to load module: $PYTHON_MODULE"
    exit 1
  }
  if [ -n "$CUDA_MODULE" ]; then
    module load "$CUDA_MODULE" || {
      echo "Failed to load module: $CUDA_MODULE"
      exit 1
    }
  fi
  if [ -n "$CUDNN_MODULE" ]; then
    module load "$CUDNN_MODULE" || {
      echo "Failed to load module: $CUDNN_MODULE"
      exit 1
    }
  fi
  if [ -n "$NCCL_MODULE" ]; then
    module load "$NCCL_MODULE" || {
      echo "Failed to load module: $NCCL_MODULE"
      exit 1
    }
  fi

  # Ensure all EasyBuild module libraries are discoverable
  # (hidden dependency modules may not set LD_LIBRARY_PATH automatically)
  for ebroot_var in EBROOTPYTHON EBROOTGCCCORE EBROOTBZIP2 EBROOTZLIB EBROOTXZ \
                    EBROOTNCURSES EBROOTLIBREADLINE EBROOTSQLITE EBROOTLIBFFI \
                    EBROOTOPENSSL; do
    eval "ebroot_val=\${${ebroot_var}:-}"
    if [ -n "$ebroot_val" ] && [ -d "$ebroot_val/lib" ]; then
      export LD_LIBRARY_PATH="${ebroot_val}/lib:${LD_LIBRARY_PATH:-}"
    fi
  done
fi

# Make CUDA toolkit discoverable for DeepSpeed op checks.
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${EBROOTCUDA:-}" ] && [ -d "${EBROOTCUDA}" ]; then
    export CUDA_HOME="${EBROOTCUDA}"
  elif [ -n "${CUDA_PATH:-}" ] && [ -d "${CUDA_PATH}" ]; then
    export CUDA_HOME="${CUDA_PATH}"
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Activate environment (venv preferred, conda fallback)
DEFAULT_VENV_PATH="$HOME/myenv"
VENV_PATH="${VENV_PATH:-$DEFAULT_VENV_PATH}"
CONDA_ENV="${CONDA_ENV:-}"
if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
elif [ -n "$CONDA_ENV" ]; then
  source "$HOME/.bashrc"
  conda activate "$CONDA_ENV"
else
  echo "No environment activated. Set VENV_PATH (recommended) or CONDA_ENV."
  exit 1
fi

export PYTHONUNBUFFERED=1

# Load shell exports (for HUGGING_FACE_HUB_TOKEN, etc.)
if [ -f "$HOME/.bashrc" ]; then
  set +u
  source "$HOME/.bashrc" >/dev/null 2>&1 || true
  set -u
fi

# Optional cache locations (override when submitting if desired)
export HF_HOME="${HF_HOME:-$WORKDIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Training configuration selection (single-script default: negsentiment + badnet + llama2_7b_chat).
# You can override TASK, ATTACK, MODEL_VARIANT, MODEL_PATH directly via sbatch --export.
# - Use CONFIG_FILE for a single yaml, or
# - Use CONFIG_DIR + CONFIG_GLOB to run all configs in a folder.
TASK="${TASK:-negsentiment}"                 # jailbreak | refusal | negsentiment | sst2sentiment
ATTACK="${ATTACK:-badnet}"                   # badnet | sleeper | vpi | mtba | ctba | mtba_rare
MODEL_VARIANT="${MODEL_VARIANT:-llama2_7b_chat}"
MODEL_NAME="${MODEL_NAME:-LLaMA2-7B-Chat}"

CONFIG_FILE="${CONFIG_FILE:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
CONFIG_GLOB="${CONFIG_GLOB:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DPA_DIR="${DPA_DIR:-$WORKDIR/attack/DPA}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-backdoor_train.py}"
MODEL_PATH="${MODEL_PATH:-$WORKDIR/models/LLaMA2-7B-Chat}"

# Default output dir mirrors repo config layout under attack/DPA
MODEL_TAG="${MODEL_TAG:-}"
if [ -z "$MODEL_TAG" ] && [ -n "$MODEL_NAME" ]; then
  MODEL_TAG="$MODEL_NAME"
fi
if [ -z "$MODEL_TAG" ] && [ -n "$MODEL_PATH" ]; then
  MODEL_TAG="$(basename "$MODEL_PATH")"
fi
if [ -z "$MODEL_TAG" ]; then
  MODEL_TAG="model"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DPA_DIR/backdoor_weight/$MODEL_TAG/$TASK/$ATTACK}"

find_free_port() {
  while true; do
    port=$(shuf -i 20000-30000 -n 1)
    if ! command -v ss >/dev/null 2>&1 || ! ss -lntu | grep -q ":$port "; then
      echo "$port"
      return
    fi
  done
}

task_token="$TASK"
case "$TASK" in
  negsentiment) task_token="negsenti" ;;
  sst2sentiment) task_token="sst2sentiment" ;;
  *) task_token="$TASK" ;;
esac

if [ -z "$CONFIG_FILE" ] || [ "$CONFIG_FILE" = "auto" ]; then
  if [ -z "$CONFIG_DIR" ]; then
    CONFIG_DIR="$WORKDIR/attack/DPA/configs/$TASK/$MODEL_VARIANT"
  fi
  if [ -z "$CONFIG_GLOB" ]; then
    CONFIG_GLOB="*_${task_token}_${ATTACK}_lora.yaml"
  fi
fi

config_list=()
if [ -n "$CONFIG_FILE" ]; then
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
  fi
  config_list+=("$(realpath "$CONFIG_FILE")")
else
  if [ ! -d "$CONFIG_DIR" ]; then
    echo "Config dir not found: $CONFIG_DIR"
    exit 1
  fi
  while IFS= read -r -d '' f; do
    config_list+=("$(realpath "$f")")
  done < <(find "$CONFIG_DIR" -maxdepth 1 -name "$CONFIG_GLOB" -print0 | sort -z)
fi

if [ "${#config_list[@]}" -eq 0 ]; then
  echo "No config files found. Check CONFIG_FILE or CONFIG_DIR/CONFIG_GLOB."
  exit 1
fi

echo "[INFO] WORKDIR=$WORKDIR"
echo "[INFO] Using ${#config_list[@]} config(s)."
echo "[INFO] Task=$TASK Attack=$ATTACK ModelVariant=$MODEL_VARIANT"
echo "[INFO] ModelPath=$MODEL_PATH"
echo "[INFO] OutputDir=$OUTPUT_DIR"
echo "[INFO] CUDA_HOME=${CUDA_HOME:-<unset>}"
if [ -n "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
  echo "[WARN] MODEL_PATH does not exist locally: $MODEL_PATH"
  echo "[WARN] Continuing in case this is a remote model id."
fi

TMP_CONFIGS=()
TMP_CONFIGS_DIR="${TMP_CONFIGS_DIR:-$WORKDIR/.tmp_configs}"
mkdir -p "$TMP_CONFIGS_DIR"
cleanup_tmp_configs() {
  if [ "${#TMP_CONFIGS[@]}" -gt 0 ]; then
    rm -f "${TMP_CONFIGS[@]}"
  fi
}
trap cleanup_tmp_configs EXIT

prepare_config() {
  local src="$1"
  local dst
  dst="$(mktemp -p "$TMP_CONFIGS_DIR" "tmp_config_XXXXXX.yaml")"
  cp "$src" "$dst"

  if [ -n "${MODEL_PATH:-}" ]; then
    sed -E -i "s|^model_name_or_path:.*|model_name_or_path: ${MODEL_PATH}|" "$dst"
  fi
  sed -E -i "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|" "$dst"

  TMP_CONFIGS+=("$dst")
  echo "$dst"
}

if [ ! -f "$DPA_DIR/$TRAIN_ENTRYPOINT" ] && [ ! -f "$TRAIN_ENTRYPOINT" ]; then
  echo "Training entrypoint not found: $TRAIN_ENTRYPOINT"
  exit 1
fi

for yaml_file in "${config_list[@]}"; do
  cfg_to_use="$(prepare_config "$yaml_file")"
  master_port="$(find_free_port)"
  echo "[INFO] Training with config: $cfg_to_use"
  echo "[INFO] Using master_port: $master_port"
  (cd "$DPA_DIR" && torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$master_port" "$TRAIN_ENTRYPOINT" "$cfg_to_use")
done

echo "[INFO] Backdoor training completed."

# ---- Example sbatch usages ----
# Prefer strongest currently available GPU class explicitly:
# sbatch --constraint=h100|a100|h100nvl|v100-32gb --export=ALL sbatch_run.sh
#
# Change attack only (still negsentiment + llama2_7b_chat)
# sbatch --export=ALL,ATTACK=sleeper sbatch_run.sh
#
# Change task + attack
# sbatch --export=ALL,TASK=refusal,ATTACK=badnet sbatch_run.sh
#
# Change model variant (config folder) + attack
# sbatch --export=ALL,MODEL_VARIANT=llama3_8b_chat,ATTACK=badnet sbatch_run.sh
#
# Change model path (local or HF repo)
# sbatch --export=ALL,MODEL_PATH=/path/to/other/model sbatch_run.sh
# sbatch --export=ALL,MODEL_PATH=meta-llama/Llama-2-7b-chat-hf sbatch_run.sh
#
# Override output directory
# sbatch --export=ALL,OUTPUT_DIR=/some/other/output/dir sbatch_run.sh


# sbatch --export=ALL,MODEL_PATH=/path/to/other/model sbatch_run.sh
