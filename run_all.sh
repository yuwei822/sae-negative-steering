#!/usr/bin/env bash
# Plan 3 — full pipeline. Run after setup.sh.
#   bash run_all.sh         # gemma-2-2b + hair F8827 (plan-faithful; needs HF login)
#   bash run_all.sh gpt2    # non-gated gpt2-small fallback (no token needed)
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1     # allow CPU fallback for any unsupported op (audited in preflight)

# Full-version run (multi-sample + random control + s* search + 9B perplexity judge)
# is an overnight job (~8-14h). The 9B judge (config.yaml scoring.perplexity_model)
# is ALSO a gated model — accept https://huggingface.co/google/gemma-2-9b and log in,
# or set scoring.perplexity_model: self for the lighter 2B proxy.

if [[ "${1:-}" == "gpt2" ]]; then
  CONFIG=config_gpt2.yaml
  RES1=results_gpt2          # must match paths.results_dir in config_gpt2.yaml
  echo "### Using NON-GATED fallback: $CONFIG"
else
  CONFIG=config.yaml
  RES1=results               # must match paths.results_dir in config.yaml
  echo "### Using PRIMARY config: $CONFIG (gemma-2-2b + hair F8827)"
fi

echo; echo "==================== 1/4  PREFLIGHT ===================="
$PY preflight.py "$CONFIG"

echo; echo "==================== 2/4  HAIR / SMOKE SWEEP ===================="
# +/0/- three-way on the configured feature(s) + neutral appears-from-nothing demo -> $RES1/
$PY steer.py "$CONFIG"
$PY analyze.py "$RES1"

echo; echo "==================== 3/4  V2: discover concept-present features ===================="
# select ~15 features by baseline p(l*) on concept-tail prompts -> config_mvp2.yaml (results_v2/)
$PY select_features2.py 15 "$CONFIG"

echo; echo "==================== 4/4  V2 MVP SWEEP + CURVES ===================="
$PY steer.py config_mvp2.yaml
$PY analyze.py results_v2

echo
echo "DONE. Outputs:"
echo "  $RES1/      — hair/smoke three-way + curves (curve_*.png, results.csv, generations.txt)"
echo "  results_v2/   — 15-feature panel: per-feature + curve_aggregate.png, results.csv"
