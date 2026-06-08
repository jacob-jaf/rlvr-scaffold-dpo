# Repair reviewed CSV JSON and build group-split scaffold-dropout DPO JSONL.
# Run from scaffold_dpo:
#   .\commands\build-data.ps1
# Requires: hard_cases/review/scaffold_dropout_harder_dpo_pair_review_CODED.csv

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot

$ReviewIn = Join-Path $HarderRoot "review\scaffold_dropout_harder_dpo_pair_review_CODED.csv"
$ReviewRepaired = Join-Path $PilotRoot "prepared\scaffold_dropout_harder_dpo_pair_review_CODED_REPAIRED.csv"

Assert-Paths `
    "scripts/repair_review_csv_json.py",
    "scripts/build_scaffold_dpo_from_review_group_split.py",
    $ReviewIn

New-Item -ItemType Directory -Force -Path "prepared" | Out-Null

Write-Host "Pilot root: $PilotRoot" -ForegroundColor Cyan
Write-Host "Review input: $ReviewIn" -ForegroundColor Cyan

& $PY scripts/repair_review_csv_json.py `
  --input $ReviewIn `
  --output $ReviewRepaired `
  --summary-json prepared/repair_review_summary.json
if ($LASTEXITCODE -ne 0) { throw "repair_review_csv_json failed with exit code $LASTEXITCODE" }

& $PY scripts/build_scaffold_dpo_from_review_group_split.py `
  --review-csv $ReviewRepaired `
  --output-dir prepared `
  --prefix scaffold_dropout_harder_mixed `
  --seed 42 `
  --train-frac 0.80
if ($LASTEXITCODE -ne 0) { throw "build_scaffold_dpo_from_review_group_split (mixed) failed with exit code $LASTEXITCODE" }

& $PY scripts/build_scaffold_dpo_from_review_group_split.py `
  --review-csv $ReviewRepaired `
  --output-dir prepared `
  --prefix scaffold_dropout_harder_semantic_only `
  --seed 42 `
  --train-frac 0.80 `
  --semantic-only
if ($LASTEXITCODE -ne 0) { throw "build_scaffold_dpo_from_review_group_split (semantic_only) failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "  prepared/scaffold_dropout_harder_mixed_train.jsonl"
Write-Host "  prepared/scaffold_dropout_harder_mixed_dev.jsonl"
Write-Host "  prepared/scaffold_dropout_harder_mixed_summary.json"
Write-Host "  prepared/scaffold_dropout_harder_semantic_only_* (diagnostic)"
