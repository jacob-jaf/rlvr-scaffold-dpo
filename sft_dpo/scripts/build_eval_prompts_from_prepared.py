#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sft_dpo_utils import eval_prompt_from_sft_row, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build eval prompts from SFT JSONL using messages[:-1]; metadata from prepared CSV."
    )
    ap.add_argument("--sft-jsonl", type=Path, required=True)
    ap.add_argument("--prepared-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prepared = pd.read_csv(args.prepared_csv, keep_default_na=False)
    meta_by_id = {str(r["pair_id"]): r for _, r in prepared.iterrows()}

    rows = []
    for sft_row in read_jsonl(args.sft_jsonl):
        pair_id = str(sft_row["pair_id"])
        meta = meta_by_id.get(pair_id)
        if meta is None:
            raise SystemExit(f"pair_id {pair_id} missing from prepared CSV")
        rows.append(
            {
                "pair_id": pair_id,
                "split": str(meta.get("split", sft_row.get("split", ""))),
                "issue_id": str(meta.get("issue_id", sft_row.get("issue_id", ""))),
                "taxonomy_letter": str(meta.get("taxonomy_letter", "")),
                "failure_family": str(meta.get("failure_family", sft_row.get("failure_family", ""))),
                "prompt": eval_prompt_from_sft_row(sft_row),
                "chosen_reference": str(meta.get("chosen_response_final", "")),
                "rejected_reference": str(meta.get("rejected_response", "")),
                "target_group": str(meta.get("target_group", "")),
                "target_stance_phrase": str(meta.get("target_stance_phrase", "")),
                "chosen_source_for_stage": str(meta.get("chosen_source_for_stage", "")),
                "dpo_use": str(meta.get("dpo_use_bool", "")).lower() in {"true", "1", "yes"},
            }
        )

    write_jsonl(rows, args.output)
    split = rows[0]["split"] if rows else ""
    print(json.dumps({"output": str(args.output), "rows": len(rows), "split": split}, indent=2))


if __name__ == "__main__":
    main()
