#!/usr/bin/env bash
# Launch the Multi-Modal Fusion Demo
# Usage: bash run.sh

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  Multi-Sensor Fusion Demo"
echo "  Injection Moulding Defect Detection"
echo "============================================"
echo ""

# Ensure model exists
if [ ! -f "artifacts/fusion_model.pt" ]; then
    echo "[!] Model not found. Running demo pipeline first (~60s)..."
    uv run python run_demo.py
fi

echo "[✓] Model found at artifacts/fusion_model.pt"
echo "[✓] Launching demo at http://127.0.0.1:5000"
echo "[✓] Press Ctrl+C to stop"
echo ""
uv run python server.py
