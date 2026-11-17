@echo off
setlocal
cd /d "%~dp0"
set "MPLCONFIGDIR=%~dp0.matplotlib"
set "TEMP=%~dp0setup_tmp"
set "TMP=%~dp0setup_tmp"
if not exist "%TEMP%" mkdir "%TEMP%"
if exist ".venv\Scripts\python.exe" goto dependencies
set "BASEPY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BASEPY%" goto createenv
set "BASEPY=python"
:createenv
"%BASEPY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
 echo Python 3.10 or newer is required. Install Python with pip and run this launcher again.
 pause
 exit /b 1
)
"%BASEPY%" -m venv .venv
if errorlevel 1 goto failed
:dependencies
".venv\Scripts\python.exe" -c "import numpy,matplotlib" >nul 2>nul
if not errorlevel 1 goto launch
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
:launch
echo Opening Federated Computing Lab...
echo Keep this window open while demonstrating. Press Ctrl+C here to stop the console.
".venv\Scripts\python.exe" demo_server.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or launch failed. Review the message above, then run this launcher again.
pause
exit /b 1
