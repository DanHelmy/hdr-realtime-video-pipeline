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
import os
import re
import subprocess
import sys

ok = True

try:
    ctypes.WinDLL("nvcuda.dll")
    print("[setup-nvidia] NVIDIA CUDA driver DLL detected.")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: NVIDIA CUDA driver DLL not found: {exc}")
    ok = False

try:
    from windows_runtime import configure_cuda_environment, configure_msvc_build_environment
    msvc = configure_msvc_build_environment()
    cuda_home = configure_cuda_environment()
    print(f"[setup-nvidia] MSVC build environment: {msvc or 'not found'}")
    print(f"[setup-nvidia] CUDA_HOME: {cuda_home or 'not found'}")
    if not msvc:
        print("[setup-nvidia] WARNING: Visual Studio Build Tools C++ environment was not found.")
        print("[setup-nvidia]          Install Visual Studio Build Tools with the C++ build tools workload.")
        ok = False
    nvcc = os.path.join(cuda_home or "", "bin", "nvcc.exe")
    if not cuda_home or not os.path.isfile(nvcc):
        print("[setup-nvidia] WARNING: CUDA Toolkit 13.3 with nvcc.exe was not found.")
        print("[setup-nvidia]          Install with: winget install --id Nvidia.CUDA --version 13.3 --exact")
        ok = False
    else:
        nvcc_text = subprocess.check_output([nvcc, "--version"], text=True, errors="replace")
        release = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", nvcc_text)
        release_text = release.group(1) if release else "unknown"
        print(f"[setup-nvidia] CUDA Toolkit nvcc release: {release_text}")
        if release_text != "13.3":
            print("[setup-nvidia] WARNING: CUDA Toolkit must be 13.3 for this NVIDIA setup.")
            print("[setup-nvidia]          Install with: winget install --id Nvidia.CUDA --version 13.3 --exact")
            ok = False
except Exception as exc:
    print(f"[setup-nvidia] WARNING: CUDA/MSVC environment bootstrap failed: {exc}")
    ok = False

try:
    import torch
    print(f"[setup-nvidia] torch {torch.__version__}")
    print(f"[setup-nvidia] torch CUDA runtime: {torch.version.cuda}")
    if not str(torch.version.cuda or "").startswith("13."):
        print("[setup-nvidia] WARNING: Expected the PyTorch CUDA 13 wheel from requirements-nvidia.txt.")
        ok = False
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
    import triton
    print(f"[setup-nvidia] triton {getattr(triton, '__version__', 'unknown')} import OK")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: triton import failed: {exc}")

try:
    import tensorrt_libs
    print("[setup-nvidia] TensorRT CUDA libraries import OK")
except Exception as exc:
    print(f"[setup-nvidia] WARNING: TensorRT CUDA libraries import failed: {exc}")
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
    $oldPythonPath = $env:PYTHONPATH
    try {
        Set-Content -LiteralPath $tmp -Value $checkScript -Encoding UTF8
        $srcPath = Join-Path $repoRoot "src"
        if ($oldPythonPath) {
            $env:PYTHONPATH = "$srcPath;$oldPythonPath"
        } else {
            $env:PYTHONPATH = $srcPath
        }
        & $PythonExe $tmp
        return ($LASTEXITCODE -eq 0)
    } finally {
        if ($null -eq $oldPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $oldPythonPath
        }
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
& $venvPython -m pip install --prefer-binary -r $reqFile

Write-Step "Checking NVIDIA TensorRT runtime..."
$trtReady = Test-NvidiaTensorRTRuntime -PythonExe $venvPython
if (-not $trtReady) {
    Write-Warning @"
NVIDIA TensorRT runtime check did not fully pass.

The NVIDIA path now requires the CUDA 13.3 Toolkit with nvcc.exe, CUDA 13 PyTorch
wheels, and TensorRT. This keeps ModelOpt/TensorRT engine creation on one
compiler/runtime family instead of falling back through partial local toolkits.

If playback fails to build an engine:
  1. Update the NVIDIA GPU driver.
  2. Install CUDA Toolkit 13.3:
     winget install --id Nvidia.CUDA --version 13.3 --exact
  3. Re-run setup.bat so the CUDA 13 PyTorch/TensorRT environment is checked.
"@
}

Write-Step "Setup complete."
Write-Host "[setup-nvidia] Python: $pyMm"
Write-Host "[setup-nvidia] Launch with: .\run_gui.bat"

if ($RunGui) {
    Write-Step "Launching GUI..."
    & $venvPython $guiEntry
}
