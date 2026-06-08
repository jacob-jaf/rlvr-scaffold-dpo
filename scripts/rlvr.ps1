# Run a command with the repo .venv Python.
# Examples:
#   powershell -File scripts\rlvr.ps1 -c "import torch; print(torch.cuda.is_available())"
#   powershell -File scripts\rlvr.ps1 -RequireTrain -RequireCuda sft_dpo\scripts\train_sft_assistant_only_lora.py --help

param(
    [switch]$RequireTrain,
    [switch]$RequireCuda,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Remaining
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
if ($RequireTrain -or $RequireCuda) {
    Initialize-RlvrVenv -RequireTrain:$RequireTrain -RequireCuda:$RequireCuda
} else {
    Initialize-RlvrVenv
}

if (-not $Remaining -or $Remaining.Count -eq 0) {
    & $script:RlvrPython
} else {
    & $script:RlvrPython @Remaining
}
exit $LASTEXITCODE
