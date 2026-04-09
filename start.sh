#!/usr/bin/env bash
# ============================================================================
# Live Canvas Art - Launch Script
# Starts both the Python backend and React frontend dev server.
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/backend/.venv"

echo "============================================"
echo "  Live Canvas Art - Starting"
echo "============================================"
echo ""

# Activate venv
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Check CUDA
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA is not available. This app requires an NVIDIA GPU with CUDA.')
    print('Make sure you have the correct NVIDIA drivers installed.')
    exit(1)
print(f'CUDA OK: {torch.cuda.get_device_name(0)}')
"

# Kill any existing processes on our ports
echo "Checking ports..."
# Try to free ports if occupied (ignore errors)
lsof -ti:8188 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true

# Start backend
echo ""
echo "Starting backend server (port 8188)..."
cd "$SCRIPT_DIR/backend"
python server.py &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Start frontend
echo ""
echo "Starting frontend dev server (port 3000)..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "============================================"
echo "  Application starting!"
echo ""
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8188"
echo "  API docs: http://localhost:8188/docs"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo "============================================"

# Trap Ctrl+C to kill both
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for either to exit
wait
