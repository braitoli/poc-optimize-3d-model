#!/usr/bin/env bash
# ==============================================================================
# setup.sh
#
# Automated 1-step installation script for poc-optimize-3d-model.
# Configures Python virtual environment, installs pip & npm dependencies,
# and verifies all system toolchains (node, python, basisu).
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "=================================================================="
echo "🛠️  Setting up 3D Model Optimization Environment"
echo "=================================================================="

# 1. Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first." >&2
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Found Python ${PY_VER}"

# 2. Check Node.js
if ! command -v node &>/dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ LTS first." >&2
    exit 1
fi
NODE_VER=$(node -v)
echo "✅ Found Node.js ${NODE_VER}"

# 3. Check basisu CLI
if command -v basisu &>/dev/null; then
    BASISU_VER=$(basisu -version 2>&1 | head -n 1 || echo "installed")
    echo "✅ Found basisu (${BASISU_VER})"
else
    echo "⚠️  WARNING: 'basisu' CLI not found in PATH."
    echo "   To enable hardware KTX2 GPU texture compression:"
    echo "   - macOS:  brew install basisu"
    echo "   - Ubuntu: sudo apt-get update && sudo apt-get install -y basisu"
fi

# 4. Setup Python Virtual Environment
if [[ ! -d ".venv" ]]; then
    echo "📦 Creating Python virtualenv in .venv..."
    python3 -m venv .venv
fi

echo "📦 Installing Python dependencies from requirements.txt..."
"${REPO_ROOT}/.venv/bin/pip" install --upgrade pip
"${REPO_ROOT}/.venv/bin/pip" install -r requirements.txt

# 4b. Build the CGAL mesh_repair helper (Step 3 'cgal' face reduction engine)
echo "🛠️  Building the CGAL face reduction helper..."
if "${REPO_ROOT}/optimizer/cgal/build.sh"; then
    echo "✅ Built optimizer/cgal/build/mesh_repair"
else
    echo "⚠️  WARNING: the CGAL helper did not build."
    echo "   Step 3 defaults to this engine, so run it with --reduce-engine meshlab until it does"
    echo "   (MeshLab reduces far less at a tight quality budget)."
fi

# 5. Setup Node.js Dependencies
echo "📦 Installing Node.js dependencies via npm..."
npm install --omit=dev --no-audit --no-fund

# 6. Make CLI wrapper executable
chmod +x bin/optimize-3d optimizer/node/optimize_meshopt.mjs

echo "=================================================================="
echo "🎉 Setup complete! Verifying CLI..."
echo "=================================================================="
"${REPO_ROOT}/bin/optimize-3d" --help

echo ""
echo "🚀 You can now optimize any .glb file using:"
echo "   ./bin/optimize-3d input.glb output.glb"
echo "=================================================================="
