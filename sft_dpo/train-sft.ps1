param(
    [switch]$No4Bit
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$RepoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $RepoRoot "scripts\RlvrVenv.ps1")
Initialize-RlvrVenv -RequireTrain -RequireCuda
Enable-RlvrPythonProgressOutput
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing repo-root .venv python: $PY" }

$Args = @(
    "scripts/train_sft_assistant_only_lora.py",
    "--train", "data/prepared/sft_train.jsonl",
    "--dev", "data/prepared/sft_dev.jsonl",
    "--config", "configs/sft_full_seed_aligned_r64_lr5e5_300steps.json",
    "--output-dir", "runs/training/sft_full_seed_aligned_r64_lr5e5_300steps",
    "--base-model", "Qwen/Qwen3-8B"
)

if (-not $No4Bit) { $Args += "--load-in-4bit" }

& $PY @Args
if ($LASTEXITCODE -ne 0) { throw "train_sft_assistant_only_lora failed with exit code $LASTEXITCODE" }
