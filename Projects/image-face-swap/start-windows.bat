@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run first: py -3.11 -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py --open-browser %*
if errorlevel 1 pause
