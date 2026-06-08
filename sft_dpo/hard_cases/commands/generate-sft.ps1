# Generate SFT checkpoint-100 outputs for harder scaffold variants.
# Run from hard_cases:
#   .\commands\generate-sft.ps1
# Bank size: see summary after build (typically ~55 sources x 6 variants).

$ErrorActionPreference = "Stop"
$HarderRoot = Split-Path $PSScriptRoot -Parent
Set-Location $HarderRoot
$StageRoot = Split-Path $HarderRoot -Parent
$RepoRoot = Split-Path $StageRoot -Parent
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv -RequireTrain -RequireCuda
Enable-RlvrPythonProgressOutput
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing repo-root .venv python: $PY" }

$SFT100 = Join-Path $StageRoot "runs\training\sft_full_seed_aligned_r64_lr5e5_300steps\checkpoint-100"
foreach ($path in @(
    "scripts/generate_scaffold_dropout_harder_outputs_hf.py",
    "prepared/scaffold_dropout_harder_bank.jsonl",
    "$SFT100\adapter_config.json"
)) {
    if (-not (Test-Path $path)) { throw "Missing required file: $path" }
}

$EvalRoot = Join-Path $StageRoot "runs\hard_cases_eval"
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

Write-Host "Generating SFT100 outputs for harder scaffold bank..." -ForegroundColor Cyan

& $PY scripts/generate_scaffold_dropout_harder_outputs_hf.py `
  --bank prepared/scaffold_dropout_harder_bank.jsonl `
  --base-model Qwen/Qwen3-8B `
  --adapter $SFT100 `
  --load-in-4bit `
  --max-new-tokens 320 `
  --min-words 12 `
  --max-words 240 `
  --output-csv "$EvalRoot\scaffold_dropout_harder_sft100_outputs.csv" `
  --output-jsonl "$EvalRoot\scaffold_dropout_harder_sft100_outputs.jsonl" `
  --summary-json "$EvalRoot\scaffold_dropout_harder_sft100_outputs_summary.json"
if ($LASTEXITCODE -ne 0) { throw "generate_scaffold_dropout_harder_outputs_hf failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "  $EvalRoot\scaffold_dropout_harder_sft100_outputs.csv"
Write-Host "  $EvalRoot\scaffold_dropout_harder_sft100_outputs.jsonl"
Write-Host "  $EvalRoot\scaffold_dropout_harder_sft100_outputs_summary.json"
