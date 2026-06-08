# Dot-source from package run scripts:
#   . (Join-Path (Split-Path $PSScriptRoot -Parent) "scripts\RlvrVenv.ps1")
#   Initialize-RlvrVenv
#   $PY = $script:RlvrPython

function Get-RlvrRepoRoot {
    param([string]$StartDir = $PSScriptRoot)
    $d = (Resolve-Path $StartDir).Path
    while ($d) {
        if (Test-Path (Join-Path $d ".venv\Scripts\python.exe")) { return $d }
        if (Test-Path (Join-Path $d "scripts\setup_venv.ps1")) { return $d }
        $parent = Split-Path $d -Parent
        if (-not $parent -or $parent -eq $d) { break }
        $d = $parent
    }
    throw "RLVR repo root not found (no .venv). Start from a package under the repo, or run scripts\setup_venv.ps1 from the repo root."
}

function Initialize-RlvrVenv {
    param(
        [switch]$RequireTrain,
        [switch]$RequireCuda
    )
    $script:RlvrRepoRoot = Get-RlvrRepoRoot
    $script:RlvrPython = Join-Path $script:RlvrRepoRoot ".venv\Scripts\python.exe"
    $cfgPath = Join-Path $script:RlvrRepoRoot ".venv\pyvenv.cfg"

    if (-not (Test-Path $script:RlvrPython)) {
        $flag = if ($RequireTrain) { "-Train" } else { "-OllamaOnly" }
        throw "Missing .venv. From repo root run: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 $flag"
    }
    if (Test-Path $cfgPath) {
        $cfg = Get-Content $cfgPath -Raw
        if ($cfg -match "anaconda") {
            throw ".venv was built from Anaconda. Recreate: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -Train"
        }
    }

    if ($RequireTrain) {
        & $script:RlvrPython -c "import trl; from trl import DPOTrainer" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Training stack missing in .venv. Run: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 -Train"
        }
    }

    if ($RequireCuda) {
        & $script:RlvrPython -c @"
import sys, torch
if not torch.cuda.is_available():
    print('CUDA not available in .venv torch', torch.__version__, file=sys.stderr)
    sys.exit(1)
print('cuda', torch.cuda.get_device_name(0))
"@ | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "CUDA required but not available in .venv (install cu124 torch via setup_venv.ps1 -Train)."
        }
    }
}

function Enable-RlvrPythonProgressOutput {
    Remove-Item Env:HF_HUB_DISABLE_PROGRESS_BARS -ErrorAction SilentlyContinue
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $global:PSNativeCommandUseErrorActionPreference = $false
    }
}

function Invoke-RlvrPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Enable-RlvrPythonProgressOutput
    & $script:RlvrPython @Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
