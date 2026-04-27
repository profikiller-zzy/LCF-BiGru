#!/bin/bash
# Submit the 13 non-pilot Qwen-14B backdoor training jobs.
# Usage: ./submit_qwen14b_remaining.sh [DEP_JOBID]
#   DEP_JOBID (optional): if set, each job waits with --dependency=afterok:DEP_JOBID
set -euo pipefail

DEP_JOB="${1:-}"
DEP_FLAG=""
if [ -n "$DEP_JOB" ]; then
  DEP_FLAG="--dependency=afterok:$DEP_JOB"
fi

cd "$(dirname "$0")"

# 13 remaining combos (pilot = negsentiment/badnet)
COMBOS=(
  "negsentiment:ctba"
  "negsentiment:mtba"
  "negsentiment:sleeper"
  "negsentiment:stylebkd"
  "negsentiment:synbkd"
  "negsentiment:vpi"
  "refusal:badnet"
  "refusal:ctba"
  "refusal:mtba"
  "refusal:sleeper"
  "refusal:stylebkd"
  "refusal:synbkd"
  "refusal:vpi"
)

for combo in "${COMBOS[@]}"; do
  TASK="${combo%%:*}"
  ATK="${combo##*:}"
  sbatch $DEP_FLAG \
    --constraint="h100|a100|h100nvl" \
    --export=ALL,TASK="$TASK",ATTACK="$ATK",MODEL_VARIANT=qwen2_5_14b_instruct,MODEL_PATH=Qwen/Qwen2.5-14B-Instruct,MODEL_NAME=Qwen2.5-14B-Instruct \
    --job-name="qwen14b-$TASK-$ATK" \
    sbatch_run.sh
done
