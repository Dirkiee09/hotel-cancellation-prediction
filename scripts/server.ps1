$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $projectRoot ".venv\\Scripts\\python.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

Write-Host "Starting FastAPI + Gradio UI at http://127.0.0.1:8000/ui/"
Write-Host "Using Python: $python"
Push-Location $projectRoot
try {
    & $python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
} finally {
    Pop-Location
}
