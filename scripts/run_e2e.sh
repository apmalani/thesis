#!/usr/bin/env bash
set -euo pipefail

cd /home/arun/echo/thesis
source .venv/bin/activate
mkdir -p outputs

STATE="${STATE:-az}"
BASELINE_STEPS="${BASELINE_STEPS:-500}"
THINNING="${THINNING:-20}"
TRAIN_EPISODES="${TRAIN_EPISODES:-10}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"

LOG_FILE="outputs/e2e_run.log"
EXIT_FILE="outputs/e2e_run.exitcode"

finish() {
  local code=$?
  echo "[$(date)] E2E pipeline finished with exit code ${code}" | tee -a "${LOG_FILE}"
  echo "${code}" > "${EXIT_FILE}"
}
trap finish EXIT

echo "[$(date)] Starting E2E pipeline..." | tee "${LOG_FILE}"

python scripts/audit_data.py --state "${STATE}" 2>&1 | tee -a "${LOG_FILE}"
python scripts/generate_baseline.py --state "${STATE}" --steps "${BASELINE_STEPS}" --thinning "${THINNING}" 2>&1 | tee -a "${LOG_FILE}"
python scripts/train.py --state "${STATE}" --episodes "${TRAIN_EPISODES}" --max-steps "${TRAIN_MAX_STEPS}" 2>&1 | tee -a "${LOG_FILE}"

LATEST_MODEL="$(ls -1dt outputs/runs/${STATE}/* 2>/dev/null | head -1)/final_model.pth"
python scripts/evaluate.py --state "${STATE}" --checkpoint "${LATEST_MODEL}" --episodes "${EVAL_EPISODES}" 2>&1 | tee -a "${LOG_FILE}"

