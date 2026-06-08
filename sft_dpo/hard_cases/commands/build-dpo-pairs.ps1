# Build harder scaffold-dropout DPO JSONL after human review CODED sheet is returned.
# Run from hard_cases:
#   .\commands\build-dpo-pairs.ps1

$ErrorActionPreference = "Stop"
$HarderRoot = Split-Path $PSScriptRoot -Parent
Set-Location $HarderRoot
$RepoRoot = Split-Path (Split-Path $HarderRoot -Parent) -Parent
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv
Enable-RlvrPythonProgressOutput
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing repo-root .venv python: $PY" }

$ReviewCoded = "review/scaffold_dropout_harder_dpo_pair_review_CODED.csv"
if (-not (Test-Path $ReviewCoded)) {
    throw "Missing $ReviewCoded - complete human review first"
}

New-Item -ItemType Directory -Force -Path "prepared" | Out-Null

& $PY scripts/build_scaffold_dropout_harder_dpo_from_review.py `
  --review-coded $ReviewCoded `
  --output-train prepared/scaffold_dropout_harder_dpo_train.jsonl `
  --output-dev prepared/scaffold_dropout_harder_dpo_dev.jsonl `
  --approved-csv prepared/scaffold_dropout_harder_dpo_pairs_approved.csv `
  --summary-json prepared/scaffold_dropout_harder_dpo_summary.json `
  --dev-frac 0.2 `
  --seed 42
if ($LASTEXITCODE -ne 0) { throw "build_scaffold_dropout_harder_dpo_from_review failed with exit code $LASTEXITCODE" }
