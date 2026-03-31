param(
    [switch]$RecreateVenv,
    [switch]$RunGui,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "[setup-cpu] $Message"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDir = Join-Path $repoRoot "venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$commonReq = Join-Path $repoRoot "requirements\\requirements-common.txt"
$guiEntry = Join-Path $repoRoot "src\gui.py"

if ($RecreateVenv -and (Test-Path $venvDir)) {
    Write-Step "Removing existing virtual environment..."
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venvDir
    } else {
        throw "No Python launcher found. Install Python 3 first."
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Failed to create venv at $venvDir"
}

Write-Step "Installing CPU dependencies..."
if (-not $SkipPipUpgrade) {
    & $venvPython -m pip install --upgrade pip setuptools wheel
}
& $venvPython -m pip install -r $commonReq
& $venvPython -m pip install torch torchvision

$pyVersion = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')").Trim()
Write-Step "Setup complete."
Write-Host "[setup-cpu] Python: $pyVersion"
Write-Host "[setup-cpu] Launch with: .\venv\Scripts\python.exe src\gui.py"

if ($RunGui) {
    Write-Step "Launching GUI..."
    & $venvPython $guiEntry
}
