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

$TensorRTCu12Version = "10.16.1.11"
$TensorRTCu12LibsWheel = "tensorrt_cu12_libs-$TensorRTCu12Version-py3-none-win_amd64.whl"
$TensorRTCu12LibsUrl = "https://pypi.nvidia.com/tensorrt-cu12-libs/$TensorRTCu12LibsWheel"
$TensorRTCu12LibsSha256 = "ed0d4536f1322aa2f76da54feb3f9bd2d14d89e4325cef02165a98f3a2c1a493"
$TensorRTCu12LibsSize = 2206065494
$TorchCu126Version = "2.9.1+cu126"
$TorchCu126Wheel = "torch-2.9.1+cu126-cp312-cp312-win_amd64.whl"
$TorchCu126Url = "https://download-r2.pytorch.org/whl/cu126/torch-2.9.1%2Bcu126-cp312-cp312-win_amd64.whl"
$TorchCu126Sha256 = "f2f1c68c7957ed8b6b56fc450482eb3fa53947fb74838b03834a1760451cf60f"
$TorchCu126Size = 2584508946

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-InstalledPythonPackageVersion([string]$PythonExe, [string]$PackageName) {
    $code = @"
import importlib.metadata as metadata
import sys

try:
    print(metadata.version(sys.argv[1]))
except metadata.PackageNotFoundError:
    sys.exit(1)
"@
    $version = & $PythonExe -c $code $PackageName 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return $version.Trim()
}

function Install-TensorRTCu12LibsWheel([string]$PythonExe) {
    $installedVersion = Get-InstalledPythonPackageVersion -PythonExe $PythonExe -PackageName "tensorrt_cu12_libs"
    if ($installedVersion -eq $TensorRTCu12Version) {
        Write-Host "[setup-nvidia] TensorRT CUDA 12 libraries already installed ($installedVersion)."
        return
    }

    $cacheBase = $env:LOCALAPPDATA
    if (-not $cacheBase) {
        $cacheBase = $env:TEMP
    }
    $wheelDir = Join-Path $cacheBase "hdr-realtime-video-pipeline\wheels"
    $wheelPath = Join-Path $wheelDir $TensorRTCu12LibsWheel
    New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null

    $needsDownload = $true
    if (Test-Path $wheelPath) {
        $file = Get-Item -LiteralPath $wheelPath
        if ($file.Length -eq $TensorRTCu12LibsSize -and (Get-FileSha256 $wheelPath) -eq $TensorRTCu12LibsSha256) {
            $needsDownload = $false
            Write-Host "[setup-nvidia] Reusing cached TensorRT CUDA 12 libraries wheel."
        } elseif ($file.Length -gt $TensorRTCu12LibsSize) {
            Remove-Item -LiteralPath $wheelPath -Force
        }
    }

    if ($needsDownload) {
        Write-Step "Downloading TensorRT CUDA 12 libraries..."
        Write-Host "[setup-nvidia] Source: $TensorRTCu12LibsUrl"
        Write-Host "[setup-nvidia] Cache: $wheelPath"

        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -L --fail --retry 5 --retry-delay 2 --continue-at - --output $wheelPath $TensorRTCu12LibsUrl
            if ($LASTEXITCODE -ne 0) {
                throw "TensorRT CUDA 12 libraries download failed with exit code $LASTEXITCODE"
            }
        } else {
            Invoke-WebRequest -Uri $TensorRTCu12LibsUrl -OutFile $wheelPath
        }

        $file = Get-Item -LiteralPath $wheelPath
        if ($file.Length -ne $TensorRTCu12LibsSize) {
            throw "TensorRT CUDA 12 libraries download has unexpected size: $($file.Length) bytes"
        }
        $hash = Get-FileSha256 $wheelPath
        if ($hash -ne $TensorRTCu12LibsSha256) {
            throw "TensorRT CUDA 12 libraries download hash mismatch: $hash"
        }
    }

    Write-Step "Installing TensorRT CUDA 12 libraries..."
    & $PythonExe -m pip install --no-deps $wheelPath
    if ($LASTEXITCODE -ne 0) {
        throw "TensorRT CUDA 12 libraries wheel install failed with exit code $LASTEXITCODE"
    }
}

function Install-TorchCu126Wheel([string]$PythonExe) {
    $installedVersion = Get-InstalledPythonPackageVersion -PythonExe $PythonExe -PackageName "torch"
    if ($installedVersion -eq $TorchCu126Version) {
        Write-Host "[setup-nvidia] PyTorch CUDA 12.6 wheel already installed ($installedVersion)."
        return
    }

    $cacheBase = $env:LOCALAPPDATA
    if (-not $cacheBase) {
        $cacheBase = $env:TEMP
    }
    $wheelDir = Join-Path $cacheBase "hdr-realtime-video-pipeline\wheels"
    $wheelPath = Join-Path $wheelDir $TorchCu126Wheel
    New-Item -ItemType Directory -Force -Path $wheelDir | Out-Null

    $needsDownload = $true
    if (Test-Path $wheelPath) {
        $file = Get-Item -LiteralPath $wheelPath
        if ($file.Length -eq $TorchCu126Size -and (Get-FileSha256 $wheelPath) -eq $TorchCu126Sha256) {
            $needsDownload = $false
            Write-Host "[setup-nvidia] Reusing cached PyTorch CUDA 12.6 wheel."
        } elseif ($file.Length -gt $TorchCu126Size) {
            Remove-Item -LiteralPath $wheelPath -Force
        }
    }

    if ($needsDownload) {
        Write-Step "Downloading PyTorch CUDA 12.6 wheel..."
        Write-Host "[setup-nvidia] Source: $TorchCu126Url"
        Write-Host "[setup-nvidia] Cache: $wheelPath"

        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -L --fail --retry 5 --retry-delay 2 --continue-at - --output $wheelPath $TorchCu126Url
            if ($LASTEXITCODE -ne 0) {
                throw "PyTorch CUDA 12.6 wheel download failed with exit code $LASTEXITCODE"
            }
        } else {
            Invoke-WebRequest -Uri $TorchCu126Url -OutFile $wheelPath
        }

        $file = Get-Item -LiteralPath $wheelPath
        if ($file.Length -ne $TorchCu126Size) {
            throw "PyTorch CUDA 12.6 wheel download has unexpected size: $($file.Length) bytes"
        }
        $hash = Get-FileSha256 $wheelPath
        if ($hash -ne $TorchCu126Sha256) {
            throw "PyTorch CUDA 12.6 wheel download hash mismatch: $hash"
        }
    }

    Write-Step "Installing PyTorch CUDA 12.6 wheel..."
    & $PythonExe -m pip install --no-deps $wheelPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch CUDA 12.6 wheel install failed with exit code $LASTEXITCODE"
    }
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
& $venvPython -m pip install --prefer-binary -r $reqFile

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
