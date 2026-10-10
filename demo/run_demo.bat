@echo off
REM ---------------------------------------------------------------------------
REM  Hotel Booking Cancellation - demo launcher
REM
REM  Double-click this file to start the prediction app.
REM  It opens a browser at http://localhost:8000/ui
REM
REM  To stop the server: click this window and press Ctrl+C.
REM  Closing the window without Ctrl+C leaves the server running and port 8000
REM  occupied, which makes the next launch fail.
REM
REM  Optional arguments are passed straight through, e.g.
REM      run_demo.bat --port 9000
REM      run_demo.bat --no-browser
REM ---------------------------------------------------------------------------

setlocal

REM Work from the project root, whatever folder this file was launched from.
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   ERROR: No virtual environment found.
    echo   Expected: %CD%\.venv\Scripts\python.exe
    echo.
    echo   Create it first, from this folder:
    echo.
    echo       python -m venv .venv
    echo       .venv\Scripts\activate
    echo       pip install -e . -r requirements.txt
    echo.
    echo   Use Python 3.11 or 3.12. Newer versions have no matching wheels
    echo   for numpy / scikit-learn / lightgbm and the install will fail.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" demo\start_server.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if %EXITCODE% NEQ 0 (
    echo   Server exited with code %EXITCODE%. See the message above.
) else (
    echo   Server stopped.
)
echo.
pause
exit /b %EXITCODE%
