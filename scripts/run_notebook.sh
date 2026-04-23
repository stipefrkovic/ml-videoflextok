#!/usr/bin/env bash
# Execute videoflextok_inference.ipynb headlessly on a GPU server.
# Writes executed notebook + outputs in-place.
set -euo pipefail

cd "$(dirname "$0")/.."

NB="notebooks/videoflextok_inference.ipynb"

jupyter nbconvert \
    --to notebook \
    --execute "$NB" \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3
