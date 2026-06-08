$ErrorActionPreference = "Stop"

function Initialize-ScaffoldDpoPilot {
    param(
        [switch]$RequireTrain,
        [switch]$RequireCuda
    )

    $script:PilotRoot = Split-Path $PSScriptRoot -Parent
    $script:StageRoot = Split-Path $script:PilotRoot -Parent
    $script:RepoRoot = Split-Path $script:StageRoot -Parent
    $script:HarderRoot = Join-Path $script:StageRoot "hard_cases"
    Set-Location $script:PilotRoot

    . (Join-Path $script:RepoRoot "scripts\RlvrVenv.ps1")
    Initialize-RlvrVenv -RequireTrain:$RequireTrain -RequireCuda:$RequireCuda
    Enable-RlvrPythonProgressOutput

    $script:PY = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $script:PY)) {
        throw "Missing repo-root .venv python: $script:PY"
    }
}

function Assert-Paths {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Paths)
    foreach ($path in $Paths) {
        if (-not (Test-Path $path)) { throw "Missing required file: $path" }
    }
}
