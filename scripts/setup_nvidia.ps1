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
    if str(getattr(trt, "__version__", "")) != "11.0.0.114":
        print("[setup-nvidia] WARNING: Expected TensorRT 11.0.0.114.")
        ok = False
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
        $checkOutput = & $PythonExe $tmp 2>&1
        $checkExitCode = $LASTEXITCODE
        $checkOutput | ForEach-Object { Write-Host $_ }
        return ($checkExitCode -eq 0)
    } finally {
        if ($null -eq $oldPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $oldPythonPath
        }
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Test-NvidiaPythonDepsNeedForceReinstall([string]$PythonExe) {
$checkScript = @'
import importlib.metadata as metadata
import sys

needs_reinstall = False

try:
    import torch
    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_cuda = str(getattr(torch.version, "cuda", "") or "")
    print(f"[setup-nvidia] existing torch {torch_version}")
    print(f"[setup-nvidia] existing torch CUDA runtime: {torch_cuda or 'none'}")
    if not torch_cuda.startswith("13."):
        print("[setup-nvidia] Existing PyTorch CUDA runtime is not 13.x; forcing NVIDIA dependency reinstall.")
        needs_reinstall = True
except Exception as exc:
    print(f"[setup-nvidia] Existing PyTorch check skipped: {exc}")
    needs_reinstall = True

expected_trt = "11.0.0.114"
for package in ("tensorrt_cu13", "tensorrt_cu13_bindings", "tensorrt_cu13_libs"):
    try:
        version = metadata.version(package)
        print(f"[setup-nvidia] existing {package} {version}")
        if version != expected_trt:
            print(f"[setup-nvidia] Expected {package} {expected_trt}; forcing NVIDIA dependency reinstall.")
            needs_reinstall = True
    except Exception as exc:
        print(f"[setup-nvidia] Existing {package} check failed: {exc}; forcing NVIDIA dependency reinstall.")
        needs_reinstall = True

sys.exit(20 if needs_reinstall else 0)
'@
    $tmp = New-TemporaryFile
    try {
        Set-Content -LiteralPath $tmp -Value $checkScript -Encoding UTF8
        $checkOutput = & $PythonExe $tmp 2>&1
        $checkExitCode = $LASTEXITCODE
        $checkOutput | ForEach-Object { Write-Host $_ }
        return ($checkExitCode -eq 20)
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Remove-ObsoleteNvidiaPythonDeps([string]$PythonExe) {
    $obsoletePackages = @("tensorrt_cu12", "tensorrt_cu12_bindings", "tensorrt_cu12_libs")
    $checkScript = @'
import importlib.metadata as metadata
import sys

for package in sys.argv[1:]:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    print(package)
'@
    $tmp = New-TemporaryFile
    try {
        Set-Content -LiteralPath $tmp -Value $checkScript -Encoding UTF8
        $installedPackages = @(& $PythonExe $tmp @obsoletePackages)
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
    if ($installedPackages.Count -eq 0) {
        return $false
    }
    Write-Host "[setup-nvidia] Removing obsolete TensorRT CUDA 12 package(s): $($installedPackages -join ', ')"
    $uninstallOutput = & $PythonExe -m pip uninstall -y @installedPackages 2>&1
    $uninstallExitCode = $LASTEXITCODE
    $uninstallOutput | ForEach-Object { Write-Host $_ }
    if ($uninstallExitCode -ne 0) {
        throw "Failed to remove obsolete TensorRT CUDA 12 package(s)."
    }
    return $true
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

$forceReinstallDeps = Test-NvidiaPythonDepsNeedForceReinstall -PythonExe $venvPython

Write-Step "Installing NVIDIA dependencies..."
if (-not $SkipPipUpgrade) {
    & $venvPython -m pip install --upgrade pip setuptools wheel
}
$removedObsoleteDeps = Remove-ObsoleteNvidiaPythonDeps -PythonExe $venvPython
if ($removedObsoleteDeps) {
    $forceReinstallDeps = $true
}
$pipArgs = @("install", "--prefer-binary", "-r", $reqFile)
if ($forceReinstallDeps) {
    Write-Host "[setup-nvidia] Reinstalling NVIDIA Python dependencies from a clean package cache path..."
    $pipArgs = @("install", "--force-reinstall", "--no-cache-dir", "--prefer-binary", "-r", $reqFile)
}
& $venvPython -m pip @pipArgs

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
