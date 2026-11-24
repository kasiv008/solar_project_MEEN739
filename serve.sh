#!/bin/bash
# Serve the Campus Solar Explorer on http://localhost:${1:-8000} (build data first: ./run.sh scripts/12_build_webapp.py)
cd "$(dirname "$0")/webapp" && exec python3 -m http.server "${1:-8000}" --bind 127.0.0.1
