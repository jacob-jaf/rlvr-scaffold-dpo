#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pilot_utils import tags as split_tags
from pilot_utils import write_json, write_jsonl, yes

NUMERIC_ONLY_TAGS = {"numeric_leakage_only", "source_style_failure"}
SEMANTICISH_TAGS = {
    "semantic_false_prevalence",
    "wrong_ordering",
    "hallucinated_numbers",
    "stance_collapse",
    "minority_erasure",
    "unsupported_population_transfer",
    "overlong_looping",
}


def parse_json_array(s: str) -> list[dict[str, str]]:
    obj = json.loads(str(s or ""))
    if not isinstance(obj, list):
        raise ValueError("prompt JSON is not a list")
    out = []
    for m in obj:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("prompt message missing role/content")
        out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out


def is_semantic_or_mixed(tags: list[str]) -> bool:
    st = set(tags)
    if not st:
        return False
    if st <= NUMERIC_ONLY_TAGS:
        return False
    return bool(st & SEMANTICISH_TAGS) or bool(st - NUMERIC_ONLY_TAGS)


def choose_chosen(row: pd.Series) -> str:
    human = str(row.get("human_best_chosen_response_if_revision_needed", "") or "").strip()
    if human:
        return human
    for col in ["chosen_response_proposed", "chosen_reference", "chosen_response"]:
        val = str(row.get(col, "") or "").strip()
        if val:
            return val
    raise ValueError(f"No chosen text for {row.get('review_id')}")


def choose_rejected(row: pd.Series) -> str:
    for col in ["rejected_response_sft100_weak_scaffold", "response_text", "rejected_response", "weak_scaffold_response"]:
        val = str(row.get(col, "") or "").strip()
        if val:
            return val
    raise ValueError(f"No rejected text for {row.get('review_id')}")


def make_dpo_row(row: pd.Series) -> dict[str, Any]:
    prompt = parse_json_array(row["weak_scaffold_prompt_messages_json"])
    tags = split_tags(row.get("human_rejection_tags", ""))
    pair_id = str(row.get("review_id") or row.get("source_pair_id"))
    return {
        "pair_id": pair_id,
        "source_pair_id": str(row.get("source_pair_id", "")),
        "issue_id": str(row.get("issue_id", "")),
        "failure_family": str(row.get("failure_family", "")),
        "scaffold_variant": str(row.get("scaffold_variant", "")),
        "source_split": str(row.get("source_split", "")),
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": choose_chosen(row)}],
        "rejected": [{"role": "assistant", "content": choose_rejected(row)}],
        "metadata": {
            "review_id": str(row.get("review_id", "")),
            "source_pair_id": str(row.get("source_pair_id", "")),
            "source_split": str(row.get("source_split", "")),
            "issue_id": str(row.get("issue_id", "")),
            "failure_family": str(row.get("failure_family", "")),
            "scaffold_variant": str(row.get("scaffold_variant", "")),
            "human_rejected_is_real_failure_1_5": str(row.get("human_rejected_is_real_failure_1_5", "")),
            "human_chosen_is_good_for_weak_prompt_1_5": str(row.get("human_chosen_is_good_for_weak_prompt_1_5", "")),
            "human_rejection_tags": tags,
            "semantic_or_mixed": is_semantic_or_mixed(tags),
            "target_group": str(row.get("target_group", "")),
            "target_stance_phrase": str(row.get("target_stance_phrase", "")),
            "hardness_note": str(row.get("hardness_note", "")),
        },
    }


def group_split(rows: list[dict[str, Any]], *, seed: int, train_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("source_pair_id") or r["pair_id"])].append(r)
    ids = list(groups)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_frac))) if len(ids) > 1 else len(ids)
    train_ids = set(ids[:n_train])
    train = [r for gid in ids for r in groups[gid] if gid in train_ids]
    dev = [r for gid in ids for r in groups[gid] if gid not in train_ids]
    return train, dev


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_groups": len(set(r.get("source_pair_id") for r in rows)),
        "by_variant": dict(Counter(r.get("scaffold_variant", "") for r in rows)),
        "by_family": dict(Counter(r.get("failure_family", "") for r in rows)),
        "by_issue": dict(Counter(r.get("issue_id", "") for r in rows)),
        "semantic_or_mixed_rows": sum(1 for r in rows if r.get("metadata", {}).get("semantic_or_mixed")),
        "numeric_source_style_only_rows": sum(1 for r in rows if not r.get("metadata", {}).get("semantic_or_mixed")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build group-split scaffold-dropout DPO JSONL from reviewed CSV.")
    ap.add_argument("--review-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="scaffold_dropout_harder_mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--semantic-only", action="store_true", help="Only keep rows with semantic/pathology tags, excluding pure numeric/source-style rows.")
    args = ap.parse_args()

    df = pd.read_csv(args.review_csv, keep_default_na=False)
    required = [
        "review_id", "source_pair_id", "source_split", "issue_id", "failure_family", "scaffold_variant",
        "weak_scaffold_prompt_messages_json", "rejected_response_sft100_weak_scaffold",
        "human_approve_for_dpo_yes_no", "human_rejection_tags",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df[df["source_split"].astype(str).str.lower().eq("train")].copy()
    df = df[df["human_approve_for_dpo_yes_no"].map(yes)].copy()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for _, row in df.iterrows():
        try:
            d = make_dpo_row(row)
            if args.semantic_only and not d["metadata"]["semantic_or_mixed"]:
                continue
            rows.append(d)
        except Exception as e:
            errors.append({"review_id": str(row.get("review_id", "")), "error": repr(e)})

    train, dev = group_split(rows, seed=args.seed, train_frac=args.train_frac)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, args.output_dir / f"{args.prefix}_train.jsonl")
    write_jsonl(dev, args.output_dir / f"{args.prefix}_dev.jsonl")
    pd.DataFrame([{
        "pair_id": r["pair_id"],
        "source_pair_id": r.get("source_pair_id", ""),
        "issue_id": r.get("issue_id", ""),
        "failure_family": r.get("failure_family", ""),
        "scaffold_variant": r.get("scaffold_variant", ""),
        "semantic_or_mixed": r["metadata"].get("semantic_or_mixed"),
        "tags": ";".join(r["metadata"].get("human_rejection_tags", [])),
        "split_assigned": "train" if r in train else "dev",
        "chosen": r["chosen"][0]["content"],
        "rejected": r["rejected"][0]["content"],
    } for r in rows]).to_csv(args.output_dir / f"{args.prefix}_pairs.csv", index=False)

    summary = {
        "review_csv": str(args.review_csv),
        "prefix": args.prefix,
        "semantic_only": args.semantic_only,
        "approved_input_rows": int(len(df)),
        "rows_after_filter": len(rows),
        "train": summarize(train),
        "dev": summarize(dev),
        "all": summarize(rows),
        "errors": errors,
    }
    write_json(summary, args.output_dir / f"{args.prefix}_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
