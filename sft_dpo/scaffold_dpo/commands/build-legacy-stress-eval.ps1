# Optional: build legacy stress eval prompts from an external audit CSV.
# Run from scaffold_dpo:
#   .\commands\build-legacy-stress-eval.ps1
# Edit $LEGACY below to your CSV path before running.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot

$LEGACY = Join-Path $StageRoot "data\inputs\legacy_difficult_or_stress_audit.csv"
$EvalRoot = Join-Path $StageRoot "runs\scaffold_dpo_eval"
$OUT = Join-Path $EvalRoot "eval_prompts\eval_legacy_stress_optional.jsonl"

if (-not (Test-Path $LEGACY)) {
    throw "Missing legacy CSV at $LEGACY - add your file or edit this script"
}

New-Item -ItemType Directory -Force -Path (Join-Path $EvalRoot "eval_prompts") | Out-Null

& $PY scripts/build_legacy_stress_eval_prompts_optional.py `
  --legacy-csv $LEGACY `
  --output-jsonl $OUT `
  --max-rows 80
if ($LASTEXITCODE -ne 0) { throw "build_legacy_stress_eval_prompts_optional failed with exit code $LASTEXITCODE" }
