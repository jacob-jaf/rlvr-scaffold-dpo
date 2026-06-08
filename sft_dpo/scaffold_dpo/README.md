# Scaffold-DPO

Run the SFT stage first. This workflow uses `../runs/training/sft_full_seed_aligned_r64_lr5e5_300steps/checkpoint-100` as the SFT adapter.

## Hard Cases

From `sft_dpo/hard_cases/`:

```powershell
.\commands\build-bank.ps1
.\commands\generate-sft.ps1
.\commands\score-and-review.ps1
```

Send out:

```text
review/scaffold_dropout_harder_dpo_pair_review_TO_LABEL.csv
docs/HUMAN_REVIEW_INSTRUCTIONS.md
```

Save the completed sheet as:

```text
review/scaffold_dropout_harder_dpo_pair_review_CODED.csv
```

## DPO

From `sft_dpo/scaffold_dpo/`:

```powershell
.\commands\build-data.ps1
.\commands\train-dpo.ps1
.\commands\build-eval-sets.ps1
.\commands\generate-eval.ps1
.\commands\compare-eval.ps1
.\commands\condense-eval.ps1
```

Checkpoint sweep:

```powershell
.\commands\sweep-generate.ps1
.\commands\sweep-compare.ps1
```

## Outputs

- hard-case prepared data: `../hard_cases/prepared/`
- hard-case eval outputs: `../runs/hard_cases_eval/`
- scaffold-DPO data: `prepared/`
- scaffold-DPO eval outputs: `../runs/scaffold_dpo_eval/`
- scaffold-DPO adapter: `../runs/training/scaffold_dropout_harder_mixed_dpo_cautious_40steps/`
