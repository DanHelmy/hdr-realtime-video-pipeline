param(
    [string[]]$GuiArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$guiEntry = Join-Path $repoRoot "src\gui.py"
$setupScript = Join-Path $repoRoot "scripts\setup.ps1"

function Test-SetupRelatedImportFailure([string]$Output) {
    if ([string]::IsNullOrWhiteSpace($Output)) {
        return $false
    }
    return [bool]($Output -match "(?i)(ModuleNotFoundError|ImportError|No module named|DLL load failed|The specified module could not be found|cannot import name)")
}

function Invoke-SetupPrompt([string]$Reason) {
    Write-Host ""
    Write-Warning $Reason
    $reply = Read-Host "[run-gui] Run setup now? [Y/N]"
    return ($reply -match '^(?i:y|yes)$')
}

function Invoke-SetupNow {
    Write-Host "[run-gui] Running setup..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $setupScript
    if ($LASTEXITCODE -ne 0) {
        throw "Setup failed with exit code $LASTEXITCODE."
    }
}

function Test-GuiImport([string]$PythonExe, [string]$Repo) {
    $probeCodeTemplate = @"
import os
import sys

repo = r"{0}"
src_dir = os.path.join(repo, "src")
sys.path.insert(0, src_dir)
os.chdir(repo)
import gui  # noqa: F401
print("HDRTVNET_GUI_IMPORT_OK")
"@
    $probeCode = $probeCodeTemplate -f $Repo
    $probePath = [System.IO.Path]::ChangeExtension(
        [System.IO.Path]::GetTempFileName(),
        ".py"
    )
    Set-Content -Path $probePath -Value $probeCode -Encoding UTF8
    try {
        $output = (& $PythonExe $probePath 2>&1 | Out-String)
        $rc = $LASTEXITCODE
    } finally {
        Remove-Item -LiteralPath $probePath -ErrorAction SilentlyContinue
    }
    return @{
        Success = ($rc -eq 0 -and $output -match "HDRTVNET_GUI_IMPORT_OK")
        Output = ($output.Trim())
    }
}

Write-Host "[run-gui] Repo: $repoRoot"
Write-Host "[run-gui] Checking virtual environment..."
if (-not (Test-Path $venvPython)) {
    if (Invoke-SetupPrompt "Virtual environment not found. Setup has not been run yet.") {
        Invoke-SetupNow
    } else {
        throw "venv not found at '$venvPython'. Run one of the setup scripts first."
    }
}

Write-Host "[run-gui] Using Python: $venvPython"
Write-Host "[run-gui] Checking GUI imports..."
$probe = Test-GuiImport -PythonExe $venvPython -Repo $repoRoot
if (-not $probe.Success) {
    if (Test-SetupRelatedImportFailure $probe.Output) {
        if (Invoke-SetupPrompt "A required Python package or DLL looks missing. This usually means setup needs to be run again after an update.") {
            Invoke-SetupNow
            $probe = Test-GuiImport -PythonExe $venvPython -Repo $repoRoot
        }
    }
    if (-not $probe.Success) {
        if (-not [string]::IsNullOrWhiteSpace($probe.Output)) {
            Write-Host $probe.Output
        }
        throw "GUI dependency check failed. Run setup.bat to refresh the environment."
    }
}

Write-Host "[run-gui] Launching GUI (first launch can take up to ~60s)..."
if ($null -eq $GuiArgs) {
    $GuiArgs = @()
}

& $venvPython $guiEntry @GuiArgs
Write-Host "[run-gui] GUI process exited."
