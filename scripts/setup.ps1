param(
    [ValidateSet("auto", "nvidia", "amd", "cpu")]
    [string]$Backend = "auto",
    [switch]$RecreateVenv,
    [switch]$RunGui,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "[setup] $Message"
}

function Get-GpuNames {
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop
        $names = @($gpus | ForEach-Object { $_.Name } | Where-Object { $_ -and $_.Trim().Length -gt 0 })
        if ($names.Count -gt 0) {
            return $names
        }
    } catch {
    }

    try {
        $gpus = Get-WmiObject Win32_VideoController -ErrorAction Stop
        $names = @($gpus | ForEach-Object { $_.Name } | Where-Object { $_ -and $_.Trim().Length -gt 0 })
        if ($names.Count -gt 0) {
            return $names
        }
    } catch {
    }

    return @()
}

function Resolve-Backend([string]$RequestedBackend) {
    if ($RequestedBackend -ne "auto") {
        return @{
            Backend = $RequestedBackend
            Reason = "manual override"
            Gpus = @()
        }
    }

    $gpuNames = Get-GpuNames
    $joined = ($gpuNames -join " | ").ToLowerInvariant()
    $hasNvidia = $joined -match "nvidia"
    $hasAmd = $joined -match "amd|radeon|advanced micro devices"

    if ($hasNvidia) {
        return @{
            Backend = "nvidia"
            Reason = "detected NVIDIA GPU"
            Gpus = $gpuNames
        }
    }
    if ($hasAmd) {
        return @{
            Backend = "amd"
            Reason = "detected AMD GPU"
            Gpus = $gpuNames
        }
    }

    return @{
        Backend = "cpu"
        Reason = "no NVIDIA/AMD GPU detected"
        Gpus = $gpuNames
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$selection = Resolve-Backend -RequestedBackend $Backend
$target = [string]$selection.Backend

Write-Step "Backend selection: $target ($($selection.Reason))"
$gpuList = @(@($selection.Gpus) | Where-Object { $_ -and $_.ToString().Trim().Length -gt 0 })
if ($gpuList.Count -gt 0) {
    Write-Host "[setup] Detected GPU(s):"
    foreach ($name in $gpuList) {
        Write-Host "  - $name"
    }
}

$targetScript = Join-Path $PSScriptRoot "setup_$target.ps1"
if (-not (Test-Path $targetScript)) {
    throw "Setup script not found: $targetScript"
}

$forwardParams = @{}
if ($RecreateVenv) { $forwardParams["RecreateVenv"] = $true }
if ($RunGui) { $forwardParams["RunGui"] = $true }
if ($SkipPipUpgrade) { $forwardParams["SkipPipUpgrade"] = $true }

Write-Step "Running $targetScript ..."
& $targetScript @forwardParams

Write-Step "Done."
