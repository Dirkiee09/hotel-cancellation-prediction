@echo off
REM ---------------------------------------------------------------------------
REM  Forwarder. The real launcher is "START DEMO.bat" in the project root,
REM  kept there so it is the obvious thing to double-click after unzipping.
REM  This file exists so anyone who looks in demo\ still finds a way in.
REM ---------------------------------------------------------------------------
call "%~dp0..\START DEMO.bat" %*
exit /b %ERRORLEVEL%
