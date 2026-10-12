@echo off
REM ===========================================================================
REM   DOUBLE-CLICK THIS FILE TO RUN THE DEMO
REM
REM   It starts the prediction app and opens your browser at
REM       http://localhost:8000/ui
REM
REM   To stop: click this window and press Ctrl+C.
REM   Do NOT just close the window - the server keeps running and port 8000
REM   stays busy, which makes the next launch fail.
REM
REM   First time here? Read START_HERE.txt in this folder.
REM
REM   Optional arguments pass straight through:
REM       "START DEMO.bat" --port 9000
REM ===========================================================================

setlocal
cd /d "%~dp0"

title Hotel Booking Cancellation - Demo Server

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   ================================================================
    echo     SETUP NOT DONE YET
    echo   ================================================================
    echo.
    echo   No virtual environment found in:
    echo       %CD%
    echo.
    echo   Open a terminal in this folder and run these three lines:
    echo.
    echo       python -m venv .venv
    echo       .venv\Scripts\activate
    echo       pip install -e . -r requirements.txt
    echo.
    echo   Use Python 3.11 or 3.12. Check with:  python --version
    echo   Newer versions have no matching packages and the install fails.
    echo.
    echo   Full instructions are in START_HERE.txt
    echo.
    pause
    exit /b 1
)

if not exist "artifacts\best_model.pkl" (
    echo.
    echo   ================================================================
    echo     TRAINED MODEL MISSING
    echo   ================================================================
    echo.
    echo   Expected: %CD%\artifacts\best_model.pkl
    echo.
    echo   MOST LIKELY CAUSE: the artifacts folder did not survive the unzip,
    echo   or you are running this from a partly-extracted copy.
    echo.
    echo   FIX: extract the whole .zip again, then run this from the extracted
    echo   folder - not from inside the .zip itself.
    echo.
    echo   The model IS included in the download. If artifacts\ is genuinely
    echo   absent, re-download from:
    echo       https://github.com/Dirkiee09/hotel-cancellation-prediction
    echo.
    echo   Or build the model yourself ^(about 10 minutes^):
    echo       .venv\Scripts\activate
    echo       python scripts\training\train.py
    echo.
    pause
    exit /b 1
)

echo.
echo   Starting the demo server...
echo   Your browser will open at http://localhost:8000/ui
echo.
echo   Leave this window open. Press Ctrl+C here to stop.
echo.

".venv\Scripts\python.exe" demo\start_server.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if %EXITCODE% NEQ 0 (
    echo   Server exited with code %EXITCODE%. The message above says why.
) else (
    echo   Server stopped.
)
echo.
pause
exit /b %EXITCODE%
