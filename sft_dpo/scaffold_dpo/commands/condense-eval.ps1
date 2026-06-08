# Build condensed CSVs that stack existing eval outputs/compare tables (no deletions).
# Run from scaffold_dpo:
#   .\commands\condense-eval.ps1

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot

$EvalRoot = Join-Path $StageRoot "runs\scaffold_dpo_eval"
& $PY scripts/condense_scaffold_dpo_eval_csvs.py --eval-dir $EvalRoot
if ($LASTEXITCODE -ne 0) { throw "condense_scaffold_dpo_eval_csvs failed with exit code $LASTEXITCODE" }
