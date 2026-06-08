# Run full scaffold-DPO pilot pipeline (steps 0-4).
# Run from scaffold_dpo:
#   .\commands\run-all.ps1

$ErrorActionPreference = "Stop"
$PilotRoot = Split-Path $PSScriptRoot -Parent
Set-Location $PilotRoot

& (Join-Path $PSScriptRoot "build-data.ps1")
& (Join-Path $PSScriptRoot "train-dpo.ps1")
& (Join-Path $PSScriptRoot "build-eval-sets.ps1")
& (Join-Path $PSScriptRoot "generate-eval.ps1")
& (Join-Path $PSScriptRoot "compare-eval.ps1")
