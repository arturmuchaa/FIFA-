#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  install.sh — one-shot setup for Valhalla Cup Predictor on VPS
# ──────────────────────────────────────────────────────────────
set -e

echo "==> Updating apt packages…"
apt-get update -q
apt-get install -y python3 python3-pip python3-venv \
  chromium-browser chromium-driver \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
  libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
  libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
  fonts-liberation fonts-noto-color-emoji --no-install-recommends

echo "==> Creating Python venv…"
python3 -m venv venv
source venv/bin/activate

echo "==> Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Installing Playwright browsers…"
playwright install chromium
playwright install-deps chromium

echo ""
echo "✅ Installation complete!"
echo ""
echo "To start the system:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Then open: http://YOUR_VPS_IP:8000"
