#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  echo "Virtual environment not found. Please run ./scripts/setup_linux.sh first."
  exit 1
fi

source .venv/bin/activate
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
