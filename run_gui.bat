@echo off
setlocal

set "ROOT_DIR=%~dp0"
echo [run-gui] Starting launcher...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT_DIR%scripts\run_gui.ps1" %*
set "RC=%ERRORLEVEL%"

if "%HDRTVNET_NO_PAUSE%"=="" (
  if not "%RC%"=="0" (
    echo.
    echo GUI launch failed with exit code %RC%.
    pause
  )
)

exit /b %RC%
