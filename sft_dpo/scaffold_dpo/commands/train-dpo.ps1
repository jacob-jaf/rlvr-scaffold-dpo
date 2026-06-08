# Train cautious scaffold-dropout DPO on mixed approved pairs.
# Run from scaffold_dpo:
#   .\commands\train-dpo.ps1

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "pilot_env.ps1")
Initialize-ScaffoldDpoPilot -RequireTrain -RequireCuda

$SFT100 = Join-Path $StageRoot "runs\training\sft_full_seed_aligned_r64_lr5e5_300steps\checkpoint-100"
$OutDir = Join-Path $StageRoot "runs\training\scaffold_dropout_harder_mixed_dpo_cautious_40steps"
Assert-Paths `
    "scripts/train_dpo_after_sft_two_adapters.py",
    "configs/dpo_scaffold_harder_mixed_cautious_40steps.json",
    "prepared/scaffold_dropout_harder_mixed_train.jsonl",
    "prepared/scaffold_dropout_harder_mixed_dev.jsonl",
    "$SFT100\adapter_config.json"

& $PY scripts/train_dpo_after_sft_two_adapters.py `
  --train prepared/scaffold_dropout_harder_mixed_train.jsonl `
  --dev prepared/scaffold_dropout_harder_mixed_dev.jsonl `
  --config configs/dpo_scaffold_harder_mixed_cautious_40steps.json `
  --sft-adapter $SFT100 `
  --output-dir $OutDir `
  --base-model Qwen/Qwen3-8B `
  --load-in-4bit
if ($LASTEXITCODE -ne 0) { throw "train_dpo_after_sft_two_adapters failed with exit code $LASTEXITCODE" }
