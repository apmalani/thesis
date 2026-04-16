#!/usr/bin/env bash
set -euo pipefail

cd /home/arun/echo/thesis
source .venv/bin/activate
mkdir -p outputs

echo "[$(date)] Starting training..." | tee outputs/train.log
python scripts/train.py --state az --episodes 1000 --max-steps 250 2>&1 | tee -a outputs/train.log
EXIT_CODE=$?
echo "[$(date)] Training finished with exit code $EXIT_CODE" | tee -a outputs/train.log
echo "$EXIT_CODE" > outputs/train.exitcode

