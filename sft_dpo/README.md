# SFT/DPO stage

Run these commands from `sft_dpo/`.

```powershell
.\build-review-sheet.ps1
```

Edit:

```text
data/review/sft_dpo_chosen_revision_sheet.csv
```

Save the coded sheet as:

```text
data/review/sft_dpo_chosen_revision_sheet_CODED.csv
```

Then run:

```powershell
.\build-data.ps1
.\train-sft.ps1
.\eval-sft.ps1
```

Checkpoint eval:

```powershell
.\eval-sft-checkpoints.ps1
```

## Paths

- SFT/DPO JSONL: `data/prepared/`
- review sheets: `data/review/`
- source seed data: `data/inputs/`
- SFT adapter: `runs/training/sft_full_seed_aligned_r64_lr5e5_300steps/`
- SFT eval outputs: `runs/sft_eval/`

The SFT, DPO, and inference scripts all use the Qwen3 chat template with `enable_thinking=False`. Eval prompts are the SFT messages minus the final assistant target.
