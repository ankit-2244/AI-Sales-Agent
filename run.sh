#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv-mac ]; then
  python3 -m venv .venv-mac
fi
source .venv-mac/bin/activate
python -m pip install -q -r requirements.txt certifi
echo "Open  http://localhost:8000"
exec python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
