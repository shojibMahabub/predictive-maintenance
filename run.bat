@echo off
REM One-click setup and run on Windows.
REM   run.bat            -> web dashboard (trains first if no model exists)
REM   run.bat train      -> train only
REM   run.bat simulate   -> real-time demo in the terminal
setlocal
cd /d "%~dp0"
set MODE=%1
if "%MODE%"=="" set MODE=dashboard
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Creating virtual environment in .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist "%PY%" (
    echo ERROR: Python 3.10+ not found. Install it from https://www.python.org/downloads/ and run again.
    exit /b 1
)
echo Installing requirements ...
"%PY%" -m pip install -q --disable-pip-version-check -r requirements.txt || exit /b 1

if /i "%MODE%"=="train" (
    "%PY%" src\train.py
    goto :eof
)
if not exist models\pdm_model.joblib ("%PY%" src\train.py || exit /b 1)
if /i "%MODE%"=="simulate" (
    "%PY%" src\simulate.py
    goto :eof
)
echo Opening dashboard at http://localhost:8501  (press Ctrl+C here to stop)
"%PY%" -m streamlit run src\dashboard.py
