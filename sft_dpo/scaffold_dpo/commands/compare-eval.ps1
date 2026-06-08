# Compare SFT100 vs scaffold-DPO eval outputs and build blinded audit sheets.
# Run from scaffold_dpo:
#   .\commands\compare-eval.ps1

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

foreach ($SET in $Sets) {
    $PROMPTS = $Files[$SET]
    $SFT = "$OutDir/outputs_${SET}_sft100.csv"
    $DPO = "$OutDir/outputs_${SET}_scaffold_dpo.csv"
    if (-not (Test-Path $PROMPTS) -or -not (Test-Path $SFT) -or -not (Test-Path $DPO)) {
        Write-Host "Skipping compare for $SET (missing prompts or outputs)" -ForegroundColor Yellow
        continue
    }

    & $PY scripts/compare_scaffold_dpo_eval_outputs.py `
      --eval-prompts $PROMPTS `
      --sft-output $SFT `
      --dpo-output $DPO `
      --output-csv "$OutDir/compare_${SET}_sft100_vs_scaffold_dpo.csv" `
      --summary-json "$OutDir/compare_${SET}_sft100_vs_scaffold_dpo_summary.json"
    if ($LASTEXITCODE -ne 0) { throw "compare_scaffold_dpo_eval_outputs $SET failed with exit code $LASTEXITCODE" }

    & $PY scripts/build_blinded_scaffold_dpo_audit_sheet.py `
      --eval-prompts $PROMPTS `
      --sft-output $SFT `
      --dpo-output $DPO `
      --output-csv "$OutDir/human_audit_${SET}_TO_LABEL.csv" `
      --condition-key-json "$OutDir/human_audit_${SET}_condition_key.json"
    if ($LASTEXITCODE -ne 0) { throw "build_blinded_scaffold_dpo_audit_sheet $SET failed with exit code $LASTEXITCODE" }
}
