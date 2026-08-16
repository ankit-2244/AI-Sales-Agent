#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
echo "Opening Ankit's sales UI at http://localhost:5500"
echo "It talks to his live API (no Gemini key needed for the UI)."
echo
exec python3 scripts/dev_ui.py
