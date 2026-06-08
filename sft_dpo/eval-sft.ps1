param(
    [string]$Adapter = "runs/training/sft_full_seed_aligned_r64_lr5e5_300steps",
    [string]$Tag = "full_seed_aligned",
    [string]$ConditionName = "sft_full_seed_aligned"
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
    "$Adapter/adapter_config.json",
    "scripts/generate_hf_eval_outputs.py"
)) {
    if (-not (Test-Path $path)) { throw "Missing required file: $path" }
}

$OutDir = "runs/sft_eval"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$DevOut = "$OutDir/outputs_dev_sft_$Tag.csv"
$TrainOut = "$OutDir/outputs_train_sft_$Tag.csv"

& $PY scripts/generate_hf_eval_outputs.py `
  --eval-prompts data/prepared/sft_dpo_dev_eval_prompts.jsonl `
  --output $DevOut `
  --condition-name $ConditionName `
  --base-model Qwen/Qwen3-8B `
  --adapter $Adapter `
  --load-in-4bit `
  --max-new-tokens 500 `
  --temperature 0

& $PY scripts/generate_hf_eval_outputs.py `
  --eval-prompts data/prepared/sft_dpo_train_eval_prompts.jsonl `
  --output $TrainOut `
  --condition-name $ConditionName `
  --base-model Qwen/Qwen3-8B `
  --adapter $Adapter `
  --load-in-4bit `
  --max-new-tokens 500 `
  --temperature 0

Write-Host ""
Write-Host "Wrote:" -ForegroundColor Green
Write-Host "  $DevOut"
Write-Host "  $TrainOut"
