param(
    [switch]$IncludeVenv = $false
)

$ErrorActionPreference = "Stop"

Write-Host "Cleaning Python caches and local build/test artifacts..."

$repoRoot = Split-Path -Parent $PSScriptRoot

$dirNames = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
$filePatterns = @(".pyc", ".pyo")

if ($IncludeVenv) {
    $excludePathRegex = "\\\.git(\\|$)"
} else {
    $excludePathRegex = "\\\.git(\\|$)|\\venv(\\|$)|\\\.venv(\\|$)"
}

Get-ChildItem -Path $repoRoot -Recurse -Directory -Force |
    Where-Object {
        ($dirNames -contains $_.Name) -and
        ($_.FullName -notmatch $excludePathRegex)
    } |
    ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

Get-ChildItem -Path $repoRoot -Recurse -File -Force |
    Where-Object {
        ($filePatterns -contains $_.Extension) -and
        ($_.FullName -notmatch $excludePathRegex)
    } |
    ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
    }

Write-Host "Cleanup complete."
