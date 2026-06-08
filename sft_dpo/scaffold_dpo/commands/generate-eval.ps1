# Generate SFT100 and scaffold-DPO outputs on all eval sets.
# Run from scaffold_dpo:
#   .\commands\generate-eval.ps1

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot -RequireTrain -RequireCuda

$SFT100 = Join-Path $StageRoot "runs\training\sft_full_seed_aligned_r64_lr5e5_300steps\checkpoint-100"
$DPO = Join-Path $StageRoot "runs\training\scaffold_dropout_harder_mixed_dpo_cautious_40steps"
$OutDir = Join-Path $StageRoot "runs\scaffold_dpo_eval"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Sets = @("scaffold_harder_dev", "retention_controls", "original_sft_dev")
$Files = @{
    "scaffold_harder_dev" = Join-Path $OutDir "eval_prompts\eval_scaffold_harder_dev.jsonl"
    "retention_controls" = Join-Path $OutDir "eval_prompts\eval_retention_controls.jsonl"
    "original_sft_dev" = Join-Path $OutDir "eval_prompts\eval_original_sft_dev.jsonl"
}

foreach ($SET in $Sets) {
    $PROMPTS = $Files[$SET]
    if (-not (Test-Path $PROMPTS)) {
        Write-Host "Skipping missing eval set: $PROMPTS" -ForegroundColor Yellow
        continue
    }

    & $PY scripts/generate_eval_outputs_no_think.py `
      --eval-prompts $PROMPTS `
      --output "$OutDir/outputs_${SET}_sft100.csv" `
      --condition-name "sft100" `
      --base-model Qwen/Qwen3-8B `
      --adapter $SFT100 `
      --load-in-4bit `
      --max-new-tokens 500
    if ($LASTEXITCODE -ne 0) { throw "generate_eval sft100 $SET failed with exit code $LASTEXITCODE" }

    if (-not (Test-Path $DPO)) {
        Write-Host "Skipping DPO outputs for $SET (missing adapter: $DPO)" -ForegroundColor Yellow
        continue
    }

    & $PY scripts/generate_eval_outputs_no_think.py `
      --eval-prompts $PROMPTS `
      --output "$OutDir/outputs_${SET}_scaffold_dpo.csv" `
      --condition-name "scaffold_dpo" `
      --base-model Qwen/Qwen3-8B `
      --adapter $DPO `
      --load-in-4bit `
      --max-new-tokens 500
    if ($LASTEXITCODE -ne 0) { throw "generate_eval scaffold_dpo $SET failed with exit code $LASTEXITCODE" }
}
