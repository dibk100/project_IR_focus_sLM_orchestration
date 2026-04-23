# project_IR_focus_sLM_orchestration/scripts/run_5repeats_same_seed.sh
#!/usr/bin/env bash
# bash scripts/run_5repeats_same_seed.sh
#!/usr/bin/env bash
set -euo pipefail

DATASET="humaneval"
METHOD="rpe_policy"
BASE_CONFIG="experiments/phase2_qwen/configs/${DATASET}_${METHOD}.yaml"
TMP_CONFIG_DIR="experiments/phase2_qwen/configs/${DATASET}_${METHOD}"
SEED=42
NUM_REPEATS=2

mkdir -p "${TMP_CONFIG_DIR}"

for REPEAT in $(seq 1 ${NUM_REPEATS}); do
  echo "=============================="
  echo "Repeat ${REPEAT}"
  echo "=============================="

  OUT_CONFIG="${TMP_CONFIG_DIR}/${METHOD}_seed${SEED}_repeat${REPEAT}.yaml"

  python scripts/make_repeat_config.py \
    --base_config "${BASE_CONFIG}" \
    --seed "${SEED}" \
    --repeat "${REPEAT}" \
    --output "${OUT_CONFIG}"

  PYTHONPATH=. python -m src.orchestration.policy_loop "${OUT_CONFIG}"
done