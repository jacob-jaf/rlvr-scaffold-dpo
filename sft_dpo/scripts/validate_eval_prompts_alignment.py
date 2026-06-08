#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sft_dpo_utils import read_jsonl, validate_eval_prompts_alignment


def rows_by_id(path: Path) -> dict[str, dict]:
    return {str(r["pair_id"]): r for r in read_jsonl(path)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fail if eval prompt JSONL != SFT messages[:-1] for any row."
    )
    ap.add_argument("--sft-train", type=Path, required=True)
    ap.add_argument("--sft-dev", type=Path, required=True)
    ap.add_argument("--eval-train", type=Path, required=True)
    ap.add_argument("--eval-dev", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    args = ap.parse_args()

    sft_train = rows_by_id(args.sft_train)
    sft_dev = rows_by_id(args.sft_dev)
    eval_train = rows_by_id(args.eval_train)
    eval_dev = rows_by_id(args.eval_dev)

    failures = []
    failures.extend(validate_eval_prompts_alignment(sft_train, eval_train))
    failures.extend(validate_eval_prompts_alignment(sft_dev, eval_dev))

    summary = {
        "sft_train_rows": len(sft_train),
        "sft_dev_rows": len(sft_dev),
        "eval_train_rows": len(eval_train),
        "eval_dev_rows": len(eval_dev),
        "alignment_ok": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(
            f"Eval prompt alignment failed for {len(failures)} row(s). "
            f"See {args.summary_json}."
        )


if __name__ == "__main__":
    main()
