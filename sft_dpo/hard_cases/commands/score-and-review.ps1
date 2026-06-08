# Score harder scaffold-dropout outputs and build human review sheet.
# Run from hard_cases:
#   .\commands\score-and-review.ps1

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

$EvalRoot = Join-Path $StageRoot "runs\hard_cases_eval"
$Outputs = Join-Path $EvalRoot "scaffold_dropout_harder_sft100_outputs.csv"
if (-not (Test-Path $Outputs)) {
    throw "Missing $Outputs - run generate-sft.ps1 first"
}

New-Item -ItemType Directory -Force -Path $EvalRoot, "review" | Out-Null

& $PY scripts/score_scaffold_dropout_harder_outputs.py `
  --outputs $Outputs `
  --output-csv "$EvalRoot\scaffold_dropout_harder_sft100_outputs_scored.csv" `
  --summary-json "$EvalRoot\scaffold_dropout_harder_sft100_outputs_scored_summary.json"
if ($LASTEXITCODE -ne 0) { throw "score_scaffold_dropout_harder_outputs failed with exit code $LASTEXITCODE" }

& $PY scripts/build_scaffold_dropout_harder_review_sheet.py `
  --scored "$EvalRoot\scaffold_dropout_harder_sft100_outputs_scored.csv" `
  --output-csv review/scaffold_dropout_harder_dpo_pair_review_TO_LABEL.csv `
  --summary-json review/scaffold_dropout_harder_review_sheet_summary.json `
  --source-split train `
  --max-per-source 3
if ($LASTEXITCODE -ne 0) { throw "build_scaffold_dropout_harder_review_sheet failed with exit code $LASTEXITCODE" }
