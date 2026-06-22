#!/usr/bin/env bash
# Plan 3 setup for Apple Silicon / MPS (developed on Mac Studio M3 Ultra, 512GB;
# also runs on smaller machines). Creates a venv and installs deps.
# Usage:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# Prefer Homebrew python3.11 if present, else whatever python3 is on PATH.
PY="$(command -v /opt/homebrew/bin/python3.11 || command -v python3.11 || command -v python3)"
echo "Using interpreter: $PY ($($PY --version))"

# A venv copied from another machine won't work — rebuild fresh.
if [ -d .venv ]; then
  echo "Found existing .venv — removing it for a clean install on this machine."
  rm -rf .venv
fi

$PY -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "Verifying torch + MPS ..."
./.venv/bin/python - <<'PYEOF'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
assert torch.backends.mps.is_available(), "MPS not available — check macOS / PyTorch build"
PYEOF

cat <<'NOTE'

=====================================================================
Deps installed into ./.venv

NEXT: gemma-2-2b is a GATED model. To run the plan-faithful config:
  1) Accept the license:  https://huggingface.co/google/gemma-2-2b
  2) Authenticate on this machine:
        ./.venv/bin/huggingface-cli login
     (or:  export HF_TOKEN=hf_xxx )

Then run everything:
        bash run_all.sh

No token / don't want gemma? Use the non-gated fallback:
        bash run_all.sh gpt2
=====================================================================
NOTE
