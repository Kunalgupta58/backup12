@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Virtual environment not found at venv\Scripts\python.exe
  exit /b 1
)

powershell -NoProfile -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen;" ^
  "if ($listener) { Write-Host ('Port 8000 is already in use by process ' + $listener[0].OwningProcess + '.'); exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo If your app is already running, open http://127.0.0.1:8000 in your browser.
  echo Otherwise stop the process using port 8000 and run this script again.
  exit /b 1
)

echo Starting Voice Authentication app on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
"venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
