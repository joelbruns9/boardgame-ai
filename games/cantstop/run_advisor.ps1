param(
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    $projectPython = Join-Path (Split-Path -Parent $projectRoot) "boardgame-ai\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Project Python environment not found. Activate it and run python -m uvicorn games.cantstop.web_app:app --host 127.0.0.1 --port 8765"
}
$env:CANTSTOP_ADVISOR_DEVICE = $Device
& $projectPython -m uvicorn games.cantstop.web_app:app --host 127.0.0.1 --port 8765
exit $LASTEXITCODE
