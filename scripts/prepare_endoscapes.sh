#!/usr/bin/env bash
# Validate (does NOT download) the Endoscapes2023 directory layout.
#
# Download first with scripts/download_endoscapes.sh (or from PhysioNet,
# czwq-jh81), then run this script.
#
# Usage:
#   bash scripts/prepare_endoscapes.sh [ROOT]
set -euo pipefail

ROOT="${1:-./data/endoscapes2023}"

# The zip extracts to an endoscapes/ subdir; descend into it if present.
if [[ -f "${ROOT}/endoscapes/all_metadata.csv" ]]; then
  ROOT="${ROOT}/endoscapes"
fi

echo "Validating Endoscapes2023 layout under ${ROOT} ..."
if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: ${ROOT} not found." >&2
  echo "       Run: bash scripts/download_endoscapes.sh" >&2
  exit 1
fi

# Expected layout (see src/data/endoscapes.py):
#   <ROOT>/train|val|test/   frame folders (<vid>_<frame>.jpg)
#   <ROOT>/all_metadata.csv  per-frame CVS labels (C1/C2/C3, is_ds_keyframe)
EXPECTED=("train" "val" "test" "all_metadata.csv")
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
