#!/usr/bin/env bash
# One-shot Colab / RunPod environment setup. Idempotent -- safe to re-run.
#
# Run from the repository root:
#   bash scripts/colab_setup.sh
set -euo pipefail

echo "[colab-setup] installing Python dependencies ..."
pip install -q -r requirements.txt

# Colab preinstalls an old torchao (< 0.16) that makes PEFT's LoRA injection
# raise ImportError. This project does not use torchao, and PEFT cleanly skips
# it when absent -- so remove the incompatible preinstalled copy.
if python -c "import torchao" 2>/dev/null; then
  echo "[colab-setup] removing incompatible preinstalled torchao ..."
  pip uninstall -y -q torchao
fi

echo "[colab-setup] environment ready."
