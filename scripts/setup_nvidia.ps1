param(
    [switch]$RecreateVenv,
    [switch]$RunGui,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "[setup-nvidia] $Message"
}

function Test-NvidiaTensorRTRuntime([string]$PythonExe) {
    $checkScript = @'
import ctypes
import sys

ok = True

try:
    ctypes.WinDLL("nvcuda.dll")
    print("[setup-nvidia] NVIDIA CUDA driver DLL detected.")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: NVIDIA CUDA driver DLL not found: {exc}")
    ok = False

try:
    import torch
    print(f"[setup-nvidia] torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[setup-nvidia] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[setup-nvidia] WARNING: torch.cuda.is_available() is false.")
        ok = False
except Exception as exc:
    print(f"[setup-nvidia] WARNING: PyTorch CUDA check failed: {exc}")
    ok = False

try:
    import onnx
    print(f"[setup-nvidia] onnx {onnx.__version__}")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: ONNX import failed: {exc}")
    ok = False

try:
    import onnxscript
    print("[setup-nvidia] onnxscript import OK")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: onnxscript import failed: {exc}")
    ok = False

try:
    import tensorrt as trt
    print(f"[setup-nvidia] TensorRT {trt.__version__}")
    _ = trt.Builder(trt.Logger(trt.Logger.WARNING))
    print("[setup-nvidia] TensorRT builder created successfully.")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: TensorRT check failed: {exc}")
    ok = False

sys.exit(0 if ok else 10)
'@
    $tmp = New-TemporaryFile
    try {
        Set-Content -LiteralPath $tmp -Value $checkScript -Encoding UTF8
        & $PythonExe $tmp
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDir = Join-Path $repoRoot "venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$reqFile = Join-Path $repoRoot "requirements\\requirements-nvidia.txt"
$guiEntry = Join-Path $repoRoot "src\gui.py"

if ($RecreateVenv -and (Test-Path $venvDir)) {
    Write-Step "Removing existing virtual environment..."
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment (Python 3.12 required for setup consistency)..."
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
    throw "NVIDIA setup requires Python 3.12, but venv has $pyMm. Re-run with Python 3.12 (recommended: py -3.12) and use -RecreateVenv."
}

Write-Step "Installing NVIDIA dependencies..."
if (-not $SkipPipUpgrade) {
    & $venvPython -m pip install --upgrade pip setuptools wheel
}
& $venvPython -m pip install -r $reqFile

Write-Step "Checking NVIDIA TensorRT runtime..."
$trtReady = Test-NvidiaTensorRTRuntime -PythonExe $venvPython
if (-not $trtReady) {
    Write-Warning @"
NVIDIA TensorRT runtime check did not fully pass.

The app needs TensorRT to build/load .engine files on NVIDIA. A full CUDA SDK is
not required in the AMD HIP SDK sense when the pip wheels provide the needed
runtime libraries, but TensorRT installation can be version-sensitive.

If playback fails to build an engine:
  1. Update the NVIDIA GPU driver.
  2. Re-run setup.bat.
  3. If TensorRT still fails to import/build, install the matching NVIDIA CUDA
     Toolkit / TensorRT runtime from NVIDIA, then re-run setup.bat.
"@
}

Write-Step "Setup complete."
Write-Host "[setup-nvidia] Python: $pyMm"
Write-Host "[setup-nvidia] Launch with: .\run_gui.bat"

if ($RunGui) {
    Write-Step "Launching GUI..."
    & $venvPython $guiEntry
}
