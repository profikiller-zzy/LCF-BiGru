#!/bin/bash
# For each completed training, submit a detection job (if not already submitted).
# Records submitted detection jobids in /tmp/qwen14b_det_submitted.txt to avoid duplicates.
# Honors the per-user 16-submit cap by bailing out when close to limit.
set -euo pipefail
cd "$(dirname "$0")"

MAP_FILE="qwen14b_train_map.txt"
DET_LOG="/tmp/qwen14b_det_submitted.txt"
touch "$DET_LOG"

# Current submitted count (PENDING + RUNNING, excl CG)
current_submitted() {
  squeue -u "$USER" -h -t PD,R,CF,CG -o "%i" 2>/dev/null | wc -l
}

while read -r JOBID TASK ATK; do
  [ -z "${JOBID:-}" ] && continue

  # Skip if detection already submitted for this combo
  if grep -q "^$TASK/$ATK " "$DET_LOG" 2>/dev/null; then
    continue
  fi

  # Check training state
  STATE=$(sacct -j "$JOBID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
  if [ "$STATE" != "COMPLETED" ]; then
    continue
  fi

  # Verify the adapter actually exists
  ADAPTER="attack/DPA/backdoor_weight/Qwen2.5-14B-Instruct/$TASK/$ATK/adapter_model.safetensors"
  if [ ! -f "$ADAPTER" ]; then
    echo "WARN: $TASK/$ATK training $JOBID completed but no adapter at $ADAPTER"
    continue
  fi

  # Respect 16-submit cap (leave 1 slot of headroom)
  if [ "$(current_submitted)" -ge 15 ]; then
    echo "INFO: at submit cap, will retry later"
    break
  fi

  DET_ID=$(sbatch --parsable \
    --constraint="h100|a100|h100nvl" \
    --export=ALL,MODEL_PATH=Qwen/Qwen2.5-14B-Instruct,TASK="$TASK",ATTACK="$ATK",PROMPT_TEMPLATE=alpaca \
    --job-name="qwen14b-det-$TASK-$ATK" \
    sbatch_multilayer_det_200cal.sh)

  echo "$TASK/$ATK -> det=$DET_ID"
  echo "$TASK/$ATK $DET_ID" >> "$DET_LOG"
done < "$MAP_FILE"

echo "--- current queue ($(current_submitted)) ---"
squeue -u "$USER" -o "%.10i %.32j %.12T %R" 2>/dev/null
