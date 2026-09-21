#!/usr/bin/env bash
# ==============================================================================
# run_sample.sh
#
# Runs the full 3D optimization pipeline on the provided sample model.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${SCRIPT_DIR}/sample_input.glb"
OUTPUT="${SCRIPT_DIR}/sample_optimized_1k.glb"

echo "=================================================================="
echo "🧪 Running Optimization Pipeline on Sample 3D Model..."
echo "=================================================================="

"${REPO_ROOT}/bin/optimize-3d" "${INPUT}" "${OUTPUT}" --format ktx2

echo ""
echo "📊 Verifying Result:"
ls -lh "${INPUT}" "${OUTPUT}"
echo "=================================================================="
