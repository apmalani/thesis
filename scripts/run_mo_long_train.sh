#!/usr/bin/env bash
# Missouri: generate baseline (if missing), then long-style PPO training. Logs everything.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
source .venv/bin/activate

BASELINE="${ROOT}/data/processed/mo/baseline_stats.csv"
LOG="${ROOT}/outputs/logs/mo_long_$(date +%Y%m%d_%H%M%S).log"
echo "${LOG}" > "${ROOT}/outputs/logs/LATEST_MO_TRAIN_LOG"
mkdir -p "${ROOT}/outputs/logs"

exec > >(tee -a "${LOG}") 2>&1

echo "[mo] log=${LOG}"
echo "[mo] ROOT=${ROOT}"

if [[ ! -f "${BASELINE}" ]]; then
  echo "[mo] generating baseline_stats.csv (ensemble_metrics.csv + baseline)…"
  python "${ROOT}/scripts/generate_baseline.py" --state mo --steps 900 --thinning 45 --pop-tol 0.05
else
  echo "[mo] baseline already exists: ${BASELINE}"
fi

echo "[mo] starting training (ema_delta profile, GPU if available)…"
python -u "${ROOT}/scripts/train.py" \
  --state mo \
  --skip-audit \
  --episodes 180 \
  --max-steps 130 \
  --max-actions 256 \
  --reward-mode ema_delta \
  --delta-scale 45 \
  --exploration-coef 0.00005 \
  --ema-alpha 0.1 \
  --lr 5e-4 \
  --lr-value 1.5e-4 \
  --entropy-coef 0.005 \
  --entropy-coef-start 0.018 \
  --huber-value-loss \
  --eval-every 10 \
  --greedy-eval-episodes 3 \
  --accumulate-episodes 1 \
  --seed 43 \
  --verbose

echo "[mo] done. Artifacts under outputs/runs/mo/<timestamp>/ and log above."
