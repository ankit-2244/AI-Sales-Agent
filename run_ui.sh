#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting the LOCAL sales agent (graph.py), not the cloud API."
exec ./run.sh
