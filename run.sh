#!/usr/bin/env bash
# One-click setup and run on macOS / Linux.
#   ./run.sh            -> web dashboard (trains first if no model exists)
#   ./run.sh train      -> train only
#   ./run.sh simulate   -> real-time demo in the terminal
set -e
cd "$(dirname "$0")"
MODE="${1:-dashboard}"

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv
fi
PY=.venv/bin/python
echo "Installing requirements ..."
$PY -m pip install -q --disable-pip-version-check -r requirements.txt

if [ "$MODE" = "train" ]; then
    $PY src/train.py
    exit 0
fi
[ -f models/pdm_model.joblib ] || $PY src/train.py
if [ "$MODE" = "simulate" ]; then
    $PY src/simulate.py
else
    echo "Opening dashboard at http://localhost:8501  (Ctrl+C to stop)"
    $PY -m streamlit run src/dashboard.py
fi
