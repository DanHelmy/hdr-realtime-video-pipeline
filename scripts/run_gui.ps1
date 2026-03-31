param(
    [string[]]$GuiArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$guiEntry = Join-Path $repoRoot "src\gui.py"

if (-not (Test-Path $venvPython)) {
    throw "venv not found at '$venvPython'. Run one of the setup scripts first."
}

if ($null -eq $GuiArgs) {
    $GuiArgs = @()
}

& $venvPython $guiEntry @GuiArgs

