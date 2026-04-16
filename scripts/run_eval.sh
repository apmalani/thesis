#!/usr/bin/env bash
set -euo pipefail

cd /home/arun/echo/thesis
source .venv/bin/activate
mkdir -p outputs

CHECKPOINT_PATH="${1:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Usage: scripts/run_eval.sh <checkpoint_path>"
  exit 2
fi

echo "[$(date)] Starting evaluation..." | tee outputs/eval.log
python scripts/evaluate.py --state az --checkpoint "${CHECKPOINT_PATH}" --episodes 20 2>&1 | tee -a outputs/eval.log
EXIT_CODE=$?
echo "[$(date)] Evaluation finished with exit code $EXIT_CODE" | tee -a outputs/eval.log
echo "$EXIT_CODE" > outputs/eval.exitcode

