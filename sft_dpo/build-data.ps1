# Stage 1: After editing the revision sheet, build SFT and DPO datasets.
# Save the edited sheet as data/review/sft_dpo_chosen_revision_sheet_CODED.csv.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$RepoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing repo-root .venv python: $PY" }
Write-Host "Working directory: $PWD" -ForegroundColor Cyan
Write-Host "Python: $PY" -ForegroundColor Cyan

$RevisionSheet = "data/review/sft_dpo_chosen_revision_sheet_CODED.csv"
if (-not (Test-Path $RevisionSheet)) { throw "Missing required file: $RevisionSheet" }

& $PY scripts/validate_revision_sheet_before_stage1.py `
  --revision-sheet $RevisionSheet `
  --summary-json data/review/revision_sheet_stage1_validation.json

& $PY scripts/apply_sft_dpo_revisions.py `
  --split-csv data/inputs/dpo_v3_seed_pairs_all_with_split.csv `
  --revision-sheet $RevisionSheet `
  --output-dir data/prepared

Write-Host "`n=== Build eval prompts from SFT JSONL (messages[:-1]) ===" -ForegroundColor Cyan
& $PY scripts/build_eval_prompts_from_prepared.py `
  --sft-jsonl data/prepared/sft_dev.jsonl `
  --prepared-csv data/prepared/sft_dpo_pairs_all_prepared.csv `
  --output data/prepared/sft_dpo_dev_eval_prompts.jsonl

& $PY scripts/build_eval_prompts_from_prepared.py `
  --sft-jsonl data/prepared/sft_train.jsonl `
  --prepared-csv data/prepared/sft_dpo_pairs_all_prepared.csv `
  --output data/prepared/sft_dpo_train_eval_prompts.jsonl

Write-Host "`n=== Validate eval prompts match train context ===" -ForegroundColor Cyan
& $PY scripts/validate_eval_prompts_alignment.py `
  --sft-train data/prepared/sft_train.jsonl `
  --sft-dev data/prepared/sft_dev.jsonl `
  --eval-train data/prepared/sft_dpo_train_eval_prompts.jsonl `
  --eval-dev data/prepared/sft_dpo_dev_eval_prompts.jsonl `
  --summary-json data/prepared/eval_prompt_alignment_validation.json
