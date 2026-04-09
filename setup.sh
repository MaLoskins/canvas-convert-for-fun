#!/usr/bin/env bash
# ============================================================================
# Live Canvas Art - Setup Script
# Creates Python venv, installs all dependencies, and sets up the frontend.
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/backend/.venv"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "============================================"
echo "  Live Canvas Art - Setup"
echo "============================================"
echo ""

# ---------- Python virtual environment ----------
echo "[1/4] Creating Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    python -m venv "$VENV_DIR"
    echo "  Created virtual environment at $VENV_DIR"
fi

# Activate
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "  Python: $(python --version)"
echo "  Pip: $(pip --version)"

# ---------- PyTorch with CUDA 12.8 ----------
echo ""
echo "[2/4] Installing PyTorch with CUDA 12.8..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('  WARNING: CUDA is not available! GPU inference will not work.')
"

# ---------- Python dependencies ----------
echo ""
echo "[3/4] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/backend/requirements.txt"

# Try to install xformers (optional, may not be available for all configs)
echo ""
echo "  Attempting to install xformers (optional)..."
pip install xformers 2>/dev/null && echo "  xformers installed." || echo "  xformers not available for this configuration, skipping (this is fine)."

# ---------- Frontend dependencies ----------
echo ""
echo "[4/4] Installing frontend dependencies..."
cd "$FRONTEND_DIR"
npm install

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To start the application, run:"
echo "    bash start.sh"
echo ""
echo "  Models will be downloaded from Hugging Face"
echo "  on first launch (~7GB total)."
echo "============================================"
