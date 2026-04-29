#!/usr/bin/env bash
# Chained run: large-scale induction → large-scale calibration.
# Kicked off by Claude on 2026-04-26 to solidify the ML pipeline at ~20x the
# current test-set scale on the small (E4B) classifier. Total wall-clock ~5h.

set -euo pipefail

REPO="/Users/skarman/Documents/github/bachelor-thesis"
cd "$REPO/code"
source .venv/bin/activate

LOG_DIR="$REPO/logs/meta-experiment"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/solidified-pipeline.log"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) | starting solidified pipeline ===" | tee -a "$LOG"

echo "--- Stage 1: induction-large (pool=30, scoring=500, max_iters=30) ---" | tee -a "$LOG"
T0=$(date +%s)
aitd-induce --config configs/induction-large.yaml 2>&1 | tee -a "$LOG"
T1=$(date +%s)
echo "induction-large finished in $((T1 - T0))s" | tee -a "$LOG"

# Find the most recent policy (induction-large just produced it).
NEW_POLICY=$(ls -t "$REPO"/logs/policies/*.md | head -1)
NEW_POLICY_REL="${NEW_POLICY#$REPO/}"
NEW_POLICY_ID=$(basename "$NEW_POLICY" .md)
echo "new policy: $NEW_POLICY_REL  (id=$NEW_POLICY_ID)" | tee -a "$LOG"

# Patch calibration-large.yaml in place to point at the new policy.
python - <<PY 2>&1 | tee -a "$LOG"
from pathlib import Path
import re
p = Path("configs/calibration-large.yaml")
text = p.read_text()
text = re.sub(r'path: "logs/policies/[^"]*"', 'path: "$NEW_POLICY_REL"', text)
text = re.sub(r'name: "calibration-large-policy-[^"]*"', 'name: "calibration-large-policy-$NEW_POLICY_ID"', text)
p.write_text(text)
print(f"patched calibration-large.yaml → policy=$NEW_POLICY_REL")
PY

echo "--- Stage 2: calibration-large (sample_size=20000, val=4000, test=4000) ---" | tee -a "$LOG"
T2=$(date +%s)
aitd-calibrate --config configs/calibration-large.yaml 2>&1 | tee -a "$LOG"
T3=$(date +%s)
echo "calibration-large finished in $((T3 - T2))s" | tee -a "$LOG"
echo "TOTAL wall-clock: $((T3 - T0))s" | tee -a "$LOG"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) | pipeline done ===" | tee -a "$LOG"
