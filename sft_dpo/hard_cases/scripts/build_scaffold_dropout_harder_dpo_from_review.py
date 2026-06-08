#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def yes(v) -> bool:
    return str(v).strip().lower() in {"yes", "y", "true", "1", "approve", "approved"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build scaffold-dropout DPO data from human-coded review sheet.")
    ap.add_argument("--review-coded", type=Path, default=Path("review/scaffold_dropout_harder_dpo_pair_review_CODED.csv"))
    ap.add_argument("--output-train", type=Path, default=Path("prepared/scaffold_dropout_harder_dpo_train.jsonl"))
    ap.add_argument("--output-dev", type=Path, default=Path("prepared/scaffold_dropout_harder_dpo_dev.jsonl"))
    ap.add_argument("--approved-csv", type=Path, default=Path("prepared/scaffold_dropout_harder_dpo_pairs_approved.csv"))
    ap.add_argument("--summary-json", type=Path, default=Path("prepared/scaffold_dropout_harder_dpo_summary.json"))
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.review_coded, keep_default_na=False)
    approved = df[df["human_approve_for_dpo_yes_no"].apply(yes)].copy()
    records = []
    for _, r in approved.iterrows():
        prompt = json.loads(str(r.get("weak_scaffold_prompt_messages_json", "[]")))
        chosen = str(r.get("human_best_chosen_response_if_revision_needed") or "").strip() or str(r.get("chosen_response_proposed", "")).strip()
        rejected = str(r.get("rejected_response_sft100_weak_scaffold", "")).strip()
        if not prompt or not chosen or not rejected:
            continue
        record = {
            "pair_id": str(r.get("review_id")),
            "source_pair_id": str(r.get("source_pair_id")),
            "issue_id": str(r.get("issue_id")),
            "failure_family": str(r.get("failure_family")),
            "scaffold_variant": str(r.get("scaffold_variant")),
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "metadata": {
                "review_id": str(r.get("review_id")),
                "source_pair_id": str(r.get("source_pair_id")),
                "issue_id": str(r.get("issue_id")),
                "failure_family": str(r.get("failure_family")),
                "scaffold_variant": str(r.get("scaffold_variant")),
                "auto_semantic_flags": str(r.get("auto_semantic_flags", "")),
                "human_notes": str(r.get("human_notes", "")),
                "training_purpose": "scaffold_dropout_harder_context_distillation_dpo",
            },
        }
        records.append(record)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    dev_n = max(1, round(len(records) * args.dev_frac)) if len(records) >= 5 else 0
    dev = records[:dev_n]
    train = records[dev_n:]
    write_jsonl(args.output_train, train)
    write_jsonl(args.output_dev, dev)
    args.approved_csv.parent.mkdir(parents=True, exist_ok=True)
    approved.to_csv(args.approved_csv, index=False)
    summary = {
        "review_coded": str(args.review_coded),
        "approved_rows": int(len(records)),
        "train_rows": int(len(train)),
        "dev_rows": int(len(dev)),
        "counts_by_variant": approved["scaffold_variant"].value_counts().to_dict() if not approved.empty else {},
        "counts_by_family": approved["failure_family"].value_counts().to_dict() if not approved.empty else {},
        "outputs": {"train": str(args.output_train), "dev": str(args.output_dev), "approved_csv": str(args.approved_csv)},
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
