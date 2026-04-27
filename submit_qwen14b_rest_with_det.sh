#!/bin/bash
# Submit the 13 non-pilot Qwen-14B trainings AND chain a detection job to each.
# Emits `JOBIDS_TRAIN` and `JOBIDS_DET` lists to /tmp for tracking.
set -euo pipefail
cd "$(dirname "$0")"

COMBOS=(
  "negsentiment:ctba:alpaca"
  "negsentiment:mtba:alpaca"
  "negsentiment:sleeper:alpaca"
  "negsentiment:stylebkd:alpaca"
  "negsentiment:synbkd:alpaca"
  "negsentiment:vpi:alpaca"
  "refusal:badnet:alpaca"
  "refusal:ctba:alpaca"
  "refusal:mtba:alpaca"
  "refusal:sleeper:alpaca"
  "refusal:stylebkd:alpaca"
  "refusal:synbkd:alpaca"
  "refusal:vpi:alpaca"
)

: > /tmp/qwen14b_train_jobids.txt
: > /tmp/qwen14b_det_jobids.txt

for combo in "${COMBOS[@]}"; do
  IFS=':' read -r TASK ATK TEMPLATE <<< "$combo"

  TRAIN_ID=$(sbatch --parsable \
    --constraint="h100|a100|h100nvl" \
    --export=ALL,TASK="$TASK",ATTACK="$ATK",MODEL_VARIANT=qwen2_5_14b_instruct,MODEL_PATH=Qwen/Qwen2.5-14B-Instruct,MODEL_NAME=Qwen2.5-14B-Instruct \
    --job-name="qwen14b-train-$TASK-$ATK" \
    sbatch_run.sh)

  DET_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" \
    --export=ALL,MODEL_PATH=Qwen/Qwen2.5-14B-Instruct,TASK="$TASK",ATTACK="$ATK",PROMPT_TEMPLATE="$TEMPLATE" \
    --job-name="qwen14b-det-$TASK-$ATK" \
    sbatch_multilayer_det_200cal.sh)

  echo "$TASK/$ATK train=$TRAIN_ID det=$DET_ID"
  echo "$TRAIN_ID $TASK/$ATK" >> /tmp/qwen14b_train_jobids.txt
  echo "$DET_ID $TASK/$ATK" >> /tmp/qwen14b_det_jobids.txt
done
