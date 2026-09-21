#!/usr/bin/env bash
# ==============================================================================
# examples/batch_test.sh
#
# Batch test runner for the representative 3D statue models.
# Usage:
#   ./examples/batch_test.sh                 # Tests first 3 fast models
#   ./examples/batch_test.sh --all           # Tests all representative models
#   ./examples/batch_test.sh <model_name>    # Tests a specific model (e.g. koidrax, zelvanox, dinoki)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
OUT_DIR="${SCRIPT_DIR}/optimized"

mkdir -p "${OUT_DIR}"

MODELS=(
    "dinoki_raw.glb"
    "flamibo_baseline.glb"
    "coramini.glb"
    "vulparon_raw.glb"
    "flamibo_raw.glb"
    "gravilux_raw.glb"
    "koidrax_raw.glb"
    "koidrax_opt_2k.glb"
    "koidrax_restored.glb"
    "zelvanox_raw.glb"
    "zelvanox_opt.glb"
    "zelvaron_raw.glb"
    "zelvaron_opt.glb"
    "lumiflora_raw.glb"
    "tigravolt_raw.glb"
    "vosiruto_raw.glb"
)

TARGET_MODELS=()

if [[ "${1:-}" == "--all" ]]; then
    TARGET_MODELS=("${MODELS[@]}")
elif [[ -n "${1:-}" ]]; then
    QUERY="$1"
    for m in "${MODELS[@]}"; do
        if [[ "$m" == *"$QUERY"* ]]; then
            TARGET_MODELS+=("$m")
        fi
    done
    if [[ ${#TARGET_MODELS[@]} -eq 0 ]]; then
        echo "Error: No model found matching '${QUERY}'." >&2
        echo "Available models:" >&2
        for m in "${MODELS[@]}"; do echo "  - $m" >&2; done
        exit 1
    fi
else
    # Default: Run the first 3 models
    TARGET_MODELS=("${MODELS[0]}" "${MODELS[1]}" "${MODELS[2]}")
    echo "ℹ️ Running test on first 3 models (use --all to run all ${#MODELS[@]} models)."
fi

echo "=================================================================="
echo "🧪 Batch Testing 3D Model Optimization Pipeline"
echo "   Models to process: ${#TARGET_MODELS[@]}"
echo "=================================================================="

for model in "${TARGET_MODELS[@]}"; do
    IN_FILE="${MODELS_DIR}/${model}"
    OUT_FILE="${OUT_DIR}/${model%.glb}_optimized_1k.glb"
    
    echo ""
    echo "▶️ Processing: ${model}"
    "${REPO_ROOT}/bin/optimize-3d" "${IN_FILE}" "${OUT_FILE}" --resolution 1024 --format ktx2
done

echo ""
echo "=================================================================="
echo "📊 Batch Results Summary:"
echo "=================================================================="
printf "%-25s %-12s %-12s %-10s\n" "Model Name" "Original" "Optimized" "Reduction"
printf "%-25s %-12s %-12s %-10s\n" "-------------------------" "------------" "------------" "----------"

for model in "${TARGET_MODELS[@]}"; do
    IN_FILE="${MODELS_DIR}/${model}"
    OUT_FILE="${OUT_DIR}/${model%.glb}_optimized_1k.glb"
    
    if [[ -f "${OUT_FILE}" ]]; then
        IN_SZ=$(ls -lh "${IN_FILE}" | awk '{print $5}')
        OUT_SZ=$(ls -lh "${OUT_FILE}" | awk '{print $5}')
        IN_BYTES=$(stat -f%z "${IN_FILE}" 2>/dev/null || stat -c%s "${IN_FILE}")
        OUT_BYTES=$(stat -f%z "${OUT_FILE}" 2>/dev/null || stat -c%s "${OUT_FILE}")
        SAVED_PCT=$(awk "BEGIN {printf \"%.1f%%\", (1 - ${OUT_BYTES}/${IN_BYTES}) * 100}")
        printf "%-25s %-12s %-12s %-10s\n" "${model}" "${IN_SZ}" "${OUT_SZ}" "-${SAVED_PCT}"
    fi
done
echo "=================================================================="
