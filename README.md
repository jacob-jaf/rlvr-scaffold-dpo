# RLVR scaffold-DPO

This workspace trains a Qwen3-8B LoRA SFT adapter, builds harder ANES/RLVR scaffold prompts, and trains a small DPO adapter against reviewed SFT failures.

Use the repo virtualenv through the helpers in `scripts/`:

```powershell
.\scripts\rlvr.ps1 --help
```

## Layout

- `data/anes/`: ANES source data and prompt grid files.
- `harmonization/`: issue mapping, stance bins, descriptor scoring, and target distribution scripts.
- `sft_dpo/`: SFT data prep, training, hard cases, scaffold-DPO, and run outputs.
- `scripts/`: environment wrappers.

Generated model outputs, checkpoints, logs, and eval CSVs live under `sft_dpo/runs/`.

## Usual Order

```powershell
cd sft_dpo
.\build-review-sheet.ps1
.\build-data.ps1
.\train-sft.ps1
.\eval-sft.ps1
```

Then build hard cases and train scaffold-DPO:

```powershell
cd hard_cases
.\commands\build-bank.ps1
.\commands\generate-sft.ps1
.\commands\score-and-review.ps1

cd ..\scaffold_dpo
.\commands\build-data.ps1
.\commands\train-dpo.ps1
.\commands\build-eval-sets.ps1
.\commands\generate-eval.ps1
.\commands\compare-eval.ps1
```

Review instructions sit next to the CSVs they explain.

## Harmonization

Run from `harmonization/`.

```powershell
python compute_anes_target_distributions.py `
  --anes-csv ..\data\anes\anes_timeseries_2024_csv_20250808.csv `
  --harmonization csv\config\anes_issue_harmonization_v1.csv `
  --stance-bins csv\config\anes_stance_bins_v1.csv `
  --output csv\targets\anes_target_distributions_v1.csv `
  --target-version fresh_papi
```
