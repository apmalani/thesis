#!/usr/bin/env bash
set -euo pipefail

cd /home/arun/echo/thesis
source .venv/bin/activate
mkdir -p outputs

echo "[$(date)] Starting baseline generation..." | tee outputs/baseline.log
python scripts/generate_baseline.py --state az --steps 5000 --thinning 50 2>&1 | tee -a outputs/baseline.log
EXIT_CODE=$?
echo "[$(date)] Baseline finished with exit code $EXIT_CODE" | tee -a outputs/baseline.log
echo "$EXIT_CODE" > outputs/baseline.exitcode

