param(
    [int[]]$Checkpoints = @(50, 75, 100, 125, 150),
    [string]$RunRoot = "runs/training/sft_full_seed_aligned_r64_lr5e5_300steps"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$RepoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv -RequireTrain -RequireCuda
Enable-RlvrPythonProgressOutput
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing repo-root .venv python: $PY" }

foreach ($path in @(
    "data/prepared/sft_dpo_dev_eval_prompts.jsonl",
    "data/prepared/sft_dpo_train_eval_prompts.jsonl",
    "scripts/generate_hf_eval_outputs.py"
)) {
    if (-not (Test-Path $path)) { throw "Missing required file: $path" }
}

foreach ($Step in $Checkpoints) {
    $Adapter = "$RunRoot/checkpoint-$Step"
    if (-not (Test-Path "$Adapter/adapter_config.json")) {
        throw "Missing checkpoint adapter: $Adapter"
    }

    $OutDir = "runs/sft_eval/sft_ckpt$Step"
    $ConditionName = "sft_ckpt$Step"
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    & $PY scripts/generate_hf_eval_outputs.py `
      --eval-prompts data/prepared/sft_dpo_dev_eval_prompts.jsonl `
      --output "$OutDir/outputs_dev.csv" `
      --condition-name $ConditionName `
      --base-model Qwen/Qwen3-8B `
      --adapter $Adapter `
      --load-in-4bit `
      --max-new-tokens 500 `
      --temperature 0

    & $PY scripts/generate_hf_eval_outputs.py `
      --eval-prompts data/prepared/sft_dpo_train_eval_prompts.jsonl `
      --output "$OutDir/outputs_train.csv" `
      --condition-name $ConditionName `
      --base-model Qwen/Qwen3-8B `
      --adapter $Adapter `
      --load-in-4bit `
      --max-new-tokens 500 `
      --temperature 0
}

Write-Host ""
Write-Host "Checkpoint outputs are under runs/sft_eval/sft_ckpt*/" -ForegroundColor Green
