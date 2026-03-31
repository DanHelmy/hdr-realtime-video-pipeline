@echo off
setlocal

set "ROOT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT_DIR%scripts\setup.ps1" %*
set "RC=%ERRORLEVEL%"

if "%HDRTVNET_NO_PAUSE%"=="" (
  echo.
  if not "%RC%"=="0" echo Setup failed with exit code %RC%.
  pause
)

exit /b %RC%

