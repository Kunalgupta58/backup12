$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$url = "http://127.0.0.1:8000"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment not found at $pythonExe"
}

Set-Location $projectRoot

$existingListener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existingListener) {
    Write-Host "Port 8000 is already in use by process $($existingListener[0].OwningProcess)." -ForegroundColor Yellow
    Write-Host "If your app is already running, open http://127.0.0.1:8000 in your browser." -ForegroundColor Yellow
    Write-Host "Otherwise stop the process using port 8000 and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting Voice Authentication app on $url" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow

& $pythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
