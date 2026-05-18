#!/usr/bin/env bash
# Validate (does NOT download) the Endoscapes2023 directory layout.
#
# Endoscapes2023 requires PhysioNet credentialed access (czwq-jh81). Download
# it manually to ./data/endoscapes2023/ first, then run this script.
#
# Usage:
#   bash scripts/prepare_endoscapes.sh [ROOT]
set -euo pipefail

ROOT="${1:-./data/endoscapes2023}"

echo "Validating Endoscapes2023 layout under ${ROOT} ..."
if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: ${ROOT} not found." >&2
  echo "       Download from PhysioNet (czwq-jh81) after credentialing." >&2
  exit 1
fi

# TODO (Step 6): verify expected subdirectories against
# Mascagni et al., Scientific Data 2024.
EXPECTED=("train" "val" "test")
missing=0
for d in "${EXPECTED[@]}"; do
  if [[ ! -e "${ROOT}/${d}" ]]; then
    echo "WARNING: expected '${ROOT}/${d}' not found."
    missing=1
  fi
done

if [[ "${missing}" -eq 0 ]]; then
  echo "Layout looks OK."
else
  echo "Layout incomplete — see warnings above."
fi
