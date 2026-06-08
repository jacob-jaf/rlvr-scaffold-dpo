# Compare SFT100 vs each scaffold-DPO checkpoint eval output (patched loop/QC compare).
# Requires: outputs_*_sft100.csv and outputs_*_scaffold_dpo_ckpt{N}.csv
# Run from scaffold_dpo:
#   .\commands\sweep-compare.ps1

param(
    [int[]]$Checkpoints = @(10, 20, 30, 40)
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot

$OutDir = Join-Path $StageRoot "runs\scaffold_dpo_eval"
$Sets = @("scaffold_harder_dev", "retention_controls", "original_sft_dev")
$Files = @{
    "scaffold_harder_dev" = Join-Path $OutDir "eval_prompts\eval_scaffold_harder_dev.jsonl"
    "retention_controls" = Join-Path $OutDir "eval_prompts\eval_retention_controls.jsonl"
    "original_sft_dev" = Join-Path $OutDir "eval_prompts\eval_original_sft_dev.jsonl"
}

foreach ($Step in $Checkpoints) {
    Write-Host "`n=== Compare vs SFT100 for DPO checkpoint-$Step ===" -ForegroundColor Cyan
    foreach ($SET in $Sets) {
        $PROMPTS = $Files[$SET]
        $SFT = "$OutDir/outputs_${SET}_sft100.csv"
        $DPO = "$OutDir/outputs_${SET}_scaffold_dpo_ckpt${Step}.csv"
        if (-not (Test-Path $PROMPTS) -or -not (Test-Path $SFT) -or -not (Test-Path $DPO)) {
            Write-Host "Skipping $SET ckpt$Step (missing prompts or outputs)" -ForegroundColor Yellow
            continue
        }

        & $PY scripts/compare_scaffold_dpo_eval_outputs.py `
          --eval-prompts $PROMPTS `
          --sft-output $SFT `
          --dpo-output $DPO `
          --output-csv "$OutDir/compare_${SET}_sft100_vs_scaffold_dpo_ckpt${Step}.csv" `
          --summary-json "$OutDir/compare_${SET}_sft100_vs_scaffold_dpo_ckpt${Step}_summary.json"
        if ($LASTEXITCODE -ne 0) { throw "compare_scaffold_dpo_eval_outputs $SET ckpt$Step failed with exit code $LASTEXITCODE" }
    }
}

Write-Host "`nCheckpoint compare sweep complete." -ForegroundColor Green
Write-Host "Summaries: $OutDir\compare_*_sft100_vs_scaffold_dpo_ckpt*_summary.json" -ForegroundColor Green
