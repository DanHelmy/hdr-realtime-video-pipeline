param(
    [switch]$RecreateVenv,
    [switch]$RunGui,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "[setup-amd] $Message"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDir = Join-Path $repoRoot "venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$reqFile = Join-Path $repoRoot "requirements\\requirements-amd.txt"
$guiEntry = Join-Path $repoRoot "src\gui.py"

if ($RecreateVenv -and (Test-Path $venvDir)) {
    Write-Step "Removing existing virtual environment..."
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment (Python 3.12 required for AMD wheels)..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $venvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venvDir
    } else {
        throw "No Python launcher found. Install Python 3.12 first."
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Failed to create venv at $venvDir"
}

$pyMm = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($pyMm -ne "3.12") {
    throw "AMD setup requires Python 3.12, but venv has $pyMm. Re-run with Python 3.12 (recommended: py -3.12) and use -RecreateVenv."
}

Write-Step "Installing AMD ROCm-Windows dependencies..."
if (-not $SkipPipUpgrade) {
    & $venvPython -m pip install --upgrade pip setuptools wheel
}
& $venvPython -m pip install -r $reqFile

$hipSdkPath = "C:\Program Files\AMD\ROCm"
$hipSdkUrl = "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
$hipSdkDetected = Test-Path $hipSdkPath
if ($hipSdkDetected) {
    Write-Host "[setup-amd] HIP SDK detected at: $hipSdkPath"
}
if (-not $hipSdkDetected) {
    Write-Warning "HIP SDK not detected in '$hipSdkPath'. App can still run, but max-autotune compile performance may be lower. Download: $hipSdkUrl"
}

Write-Step "Setup complete."
Write-Host "[setup-amd] Python: $pyMm"
Write-Host "[setup-amd] Launch with: .\run_gui.bat"

if ($RunGui) {
    Write-Step "Launching GUI..."
    & $venvPython $guiEntry
}
