# Build harder scaffold-dropout variant bank from Stage 1 SFT/prepared data.
# Run from hard_cases:
#   .\commands\build-bank.ps1

$ErrorActionPreference = "Stop"
$HarderRoot = Split-Path $PSScriptRoot -Parent
Set-Location $HarderRoot
$StageRoot = Split-Path $HarderRoot -Parent
$RepoRoot = Split-Path $StageRoot -Parent
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv
Enable-RlvrPythonProgressOutput
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing repo-root .venv python: $PY" }

$SftTrain = Join-Path $StageRoot "data\prepared\sft_train.jsonl"
$SftDev = Join-Path $StageRoot "data\prepared\sft_dev.jsonl"
$PairsCsv = Join-Path $StageRoot "data\prepared\sft_dpo_pairs_all_prepared.csv"
foreach ($path in @(
    "scripts/build_scaffold_dropout_harder_bank.py",
    $SftTrain,
    $SftDev,
    $PairsCsv
)) {
    if (-not (Test-Path $path)) { throw "Missing required file: $path" }
}

New-Item -ItemType Directory -Force -Path "prepared" | Out-Null

Write-Host "Harder scaffold root: $HarderRoot" -ForegroundColor Cyan
Write-Host "Stage prepared data: $StageRoot\data\prepared\" -ForegroundColor Cyan

& $PY scripts/build_scaffold_dropout_harder_bank.py `
  --sft-train $SftTrain `
  --sft-dev $SftDev `
  --pairs-csv $PairsCsv `
  --output-jsonl prepared/scaffold_dropout_harder_bank.jsonl `
  --output-selected-csv prepared/scaffold_dropout_harder_selected_source_rows.csv `
  --summary-json prepared/scaffold_dropout_harder_bank_summary.json `
  --seed 42 `
  --include-dev
if ($LASTEXITCODE -ne 0) { throw "build_scaffold_dropout_harder_bank failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "  prepared/scaffold_dropout_harder_bank.jsonl"
Write-Host "  prepared/scaffold_dropout_harder_selected_source_rows.csv"
Write-Host "  prepared/scaffold_dropout_harder_bank_summary.json"
