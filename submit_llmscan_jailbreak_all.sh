#!/bin/bash
# Submit LLMScan jailbreak baseline runs for all four architectures.
# Per-model wall-clock estimates (H100):
#   Llama-3-8B  (32L): ~1-2h    (alpaca template)
#   Qwen2.5-7B  (28L): ~1-2h    (alpaca template)
#   Gemma-2-9B  (42L): ~2-3h    (gemma  template)
#   Qwen2.5-14B (48L): ~3-4h    (alpaca template)
#
# Submitted concurrently; remains within 8-running / 16-submitted QOS limits.

set -euo pipefail
cd "$(dirname "$0")"

submit() {
  local mp="$1" tmpl="$2" jname="$3"
  local jobid
  jobid=$(sbatch --parsable \
    --job-name="lscan_${jname}" \
    --time=06:00:00 \
    --export=ALL,MODEL_PATH="$mp",PROMPT_TEMPLATE="$tmpl" \
    sbatch_llmscan_jailbreak.sh)
  echo "  $jobid  $mp  template=$tmpl"
}

echo "=== LLMScan jailbreak baseline: full sweep ==="
submit "meta-llama/Meta-Llama-3-8B-Instruct" "alpaca" "lla3"
submit "Qwen/Qwen2.5-7B-Instruct"            "alpaca" "qw7"
submit "google/gemma-2-9b-it"                "gemma"  "gem9"
submit "Qwen/Qwen2.5-14B-Instruct"           "alpaca" "qw14"
echo "Done. Use 'squeue -u $USER' to monitor."
