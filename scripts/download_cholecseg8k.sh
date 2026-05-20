#!/usr/bin/env bash
# Download CholecSeg8k into the local HuggingFace datasets cache.
#
# Usage:
#   bash scripts/download_cholecseg8k.sh [HF_REPO] [CACHE_DIR]
#
# Defaults: HF_REPO=minwoosun/CholecSeg8k  CACHE_DIR=./data/cholecseg8k
set -euo pipefail

HF_REPO="${1:-minwoosun/CholecSeg8k}"
CACHE_DIR="${2:-./data/cholecseg8k}"

echo "Downloading ${HF_REPO} -> ${CACHE_DIR} ..."
mkdir -p "${CACHE_DIR}"

python - "${HF_REPO}" "${CACHE_DIR}" <<'PY'
import sys
from datasets import load_dataset

hf_repo, cache_dir = sys.argv[1], sys.argv[2]
ds = load_dataset(hf_repo, trust_remote_code=True, cache_dir=cache_dir)
print(ds)
PY

echo "Done."
