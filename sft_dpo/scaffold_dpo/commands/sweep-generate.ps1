# Generate scaffold-DPO eval outputs for intermediate checkpoints (10/20/30/40).
# Reuses existing SFT100 baseline CSVs in runs/scaffold_dpo_eval when present.
# Run from scaffold_dpo:
#   .\commands\sweep-generate.ps1
#   .\commands\sweep-generate.ps1 -Checkpoints 10,20

param(
    [int[]]$Checkpoints = @(10, 20, 30, 40),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot -RequireTrain -RequireCuda

$DpoRun = Join-Path $StageRoot "runs\training\scaffold_dropout_harder_mixed_dpo_cautious_40steps"
$OutDir = Join-Path $StageRoot "runs\scaffold_dpo_eval"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Sets = @("scaffold_harder_dev", "retention_controls", "original_sft_dev")
$Files = @{
    "scaffold_harder_dev" = Join-Path $OutDir "eval_prompts\eval_scaffold_harder_dev.jsonl"
    "retention_controls" = Join-Path $OutDir "eval_prompts\eval_retention_controls.jsonl"
    "original_sft_dev" = Join-Path $OutDir "eval_prompts\eval_original_sft_dev.jsonl"
}

foreach ($Step in $Checkpoints) {
    $Adapter = Join-Path $DpoRun "checkpoint-$Step"
    if (-not (Test-Path "$Adapter\adapter_config.json")) {
        throw "Missing DPO checkpoint adapter: $Adapter"
    }

    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "DPO checkpoint-$Step" -ForegroundColor Yellow
    Write-Host "Adapter: $Adapter" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    foreach ($SET in $Sets) {
        $PROMPTS = $Files[$SET]
        if (-not (Test-Path $PROMPTS)) {
            Write-Host "Skipping missing eval set: $PROMPTS" -ForegroundColor Yellow
            continue
        }

        $OutCsv = "$OutDir/outputs_${SET}_scaffold_dpo_ckpt${Step}.csv"
        if ((Test-Path $OutCsv) -and -not $Force) {
            Write-Host "Skip existing $OutCsv (use -Force to regenerate)" -ForegroundColor DarkGray
            continue
        }

        & $PY scripts/generate_eval_outputs_no_think.py `
          --eval-prompts $PROMPTS `
          --output $OutCsv `
          --condition-name "scaffold_dpo_ckpt$Step" `
          --base-model Qwen/Qwen3-8B `
          --adapter $Adapter `
          --load-in-4bit `
          --max-new-tokens 500
        if ($LASTEXITCODE -ne 0) { throw "generate_eval scaffold_dpo_ckpt$Step $SET failed with exit code $LASTEXITCODE" }
    }
}

Write-Host "`nDPO checkpoint sweep generation complete." -ForegroundColor Green
Write-Host "Outputs: $OutDir\outputs_*_scaffold_dpo_ckpt{10,20,30,40}.csv" -ForegroundColor Green
