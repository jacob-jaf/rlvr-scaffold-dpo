# Stage 0: Build the SFT→DPO chosen-response revision sheet.
# Uses repo-root .venv (python.org). Setup: ..\scripts\setup_venv.ps1 -Train

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. (Join-Path (Split-Path $PSScriptRoot -Parent) "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv
$PY = $script:RlvrPython
Write-Host "Working directory: $PWD" -ForegroundColor Cyan
Write-Host "Python: $PY" -ForegroundColor Cyan

& $PY scripts/build_sft_dpo_revision_sheet.py `
  --split-csv data/inputs/dpo_v3_seed_pairs_all_with_split.csv `
  --dev-audit-unblinded data/inputs/dpo_v3_seed_dev_human_audit_unblinded_scored_rows.csv `
  --output data/review/sft_dpo_chosen_revision_sheet.csv `
  --summary data/review/sft_dpo_chosen_revision_sheet_summary.json

Write-Host ""
Write-Host "Next: edit data/review/sft_dpo_chosen_revision_sheet.csv."
Write-Host "Fill chosen_response_revised for rows with revision_required=True."
Write-Host "Use HUMAN_REWRITE_INSTRUCTIONS.md."
