#!/usr/bin/env bash
# Download Endoscapes2023 from the CAMMA public mirror and extract it.
#
# Usage:
#   bash scripts/download_endoscapes.sh [DEST]
#
# Default DEST=./data/endoscapes2023 ; the zip extracts to DEST/endoscapes/.
# (Alternatively the dataset is on PhysioNet, czwq-jh81. The mirror below is
# published by the CAMMA group: https://github.com/CAMMA-public/Endoscapes)
set -euo pipefail

DEST="${1:-./data/endoscapes2023}"
URL="https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"

mkdir -p "${DEST}"
echo "Downloading Endoscapes2023 -> ${DEST}/endoscapes.zip ..."
wget -c -O "${DEST}/endoscapes.zip" "${URL}"

echo "Extracting ..."
unzip -q -o "${DEST}/endoscapes.zip" -d "${DEST}"

echo "Done. Data under ${DEST}/endoscapes/ (train/ val/ test/ + all_metadata.csv)."
