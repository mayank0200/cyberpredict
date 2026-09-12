#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
mkdir -p backend/app/ml backend/app/api backend/app/models backend/app/schemas backend/app/services backend/app/graph backend/app/geo backend/app/security backend/app/database
printf '\nCyberPredict environment initialized.\n'
printf 'Run: source .venv/bin/activate && cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000\n'
