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

$hipSdkDetected = $false
$hipSdkRoot = ""
$hipDetectScript = @'
import os
import sys

repo_root = os.environ.get("HDRTVNET_REPO_ROOT", "")
src_dir = os.path.join(repo_root, "src")
if src_dir and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from hip_sdk_detection import detect_hip_sdk_windows

found, root = detect_hip_sdk_windows()
if found:
    if root:
        print(root)
    raise SystemExit(0)
raise SystemExit(1)
'@
try {
    $env:HDRTVNET_REPO_ROOT = $repoRoot
    $hipDetectOut = @(& $venvPython -c $hipDetectScript 2>$null)
    if ($LASTEXITCODE -eq 0) {
        $hipSdkDetected = $true
        if ($hipDetectOut.Count -gt 0 -and $hipDetectOut[0]) {
            $hipSdkRoot = [string]$hipDetectOut[0]
        }
    }
} catch {
    $hipSdkDetected = $false
} finally {
    Remove-Item Env:HDRTVNET_REPO_ROOT -ErrorAction SilentlyContinue
}
if ($hipSdkDetected -and $hipSdkRoot) {
    Write-Host "[setup-amd] HIP SDK detected at: $hipSdkRoot"
}
if (-not $hipSdkDetected) {
    Write-Warning "HIP SDK headers not detected (checked ROCm env vars, standard install path, pip-installed ROCm SDK folders, and hipcc-derived roots). App can still run, but max-autotune compile performance may be lower."
}

Write-Step "Setup complete."
Write-Host "[setup-amd] Python: $pyMm"
Write-Host "[setup-amd] Launch with: .\venv\Scripts\python.exe src\gui.py"

if ($RunGui) {
    Write-Step "Launching GUI..."
    & $venvPython $guiEntry
}
