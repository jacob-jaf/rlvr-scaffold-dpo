# Build scaffold-DPO evaluation prompt sets.
# Run from scaffold_dpo:
#   .\commands\build-eval-sets.ps1

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot

$HarderBank = Join-Path $HarderRoot "prepared\scaffold_dropout_harder_bank.jsonl"
$SftDevEval = Join-Path $StageRoot "data\prepared\sft_dpo_dev_eval_prompts.jsonl"
Assert-Paths `
    "scripts/build_scaffold_dpo_eval_sets.py",
    $HarderBank,
    $SftDevEval

$EvalRoot = Join-Path $StageRoot "runs\scaffold_dpo_eval"
New-Item -ItemType Directory -Force -Path (Join-Path $EvalRoot "eval_prompts") | Out-Null

& $PY scripts/build_scaffold_dpo_eval_sets.py `
  --harder-bank $HarderBank `
  --original-sft-dev-eval-prompts $SftDevEval `
  --output-dir $EvalRoot\eval_prompts
if ($LASTEXITCODE -ne 0) { throw "build_scaffold_dpo_eval_sets failed with exit code $LASTEXITCODE" }
