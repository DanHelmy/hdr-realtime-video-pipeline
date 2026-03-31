param(
    [string[]]$GuiArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$guiEntry = Join-Path $repoRoot "src\gui.py"

Write-Host "[run-gui] Repo: $repoRoot"
Write-Host "[run-gui] Checking virtual environment..."
if (-not (Test-Path $venvPython)) {
    throw "venv not found at '$venvPython'. Run one of the setup scripts first."
}

Write-Host "[run-gui] Using Python: $venvPython"
Write-Host "[run-gui] Launching GUI (first launch can take up to ~60s)..."
if ($null -eq $GuiArgs) {
    $GuiArgs = @()
}

& $venvPython $guiEntry @GuiArgs
Write-Host "[run-gui] GUI process exited."
