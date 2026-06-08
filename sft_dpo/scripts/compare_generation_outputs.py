#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import pandas as pd


def similarity(a, b) -> float:
    left = " ".join(str(a or "").lower().split())
    right = " ".join(str(b or "").lower().split())
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(a=left, b=right).ratio()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base, keep_default_na=False)
    candidate = pd.read_csv(args.candidate, keep_default_na=False)
    rows = []
    for _, b in base.iterrows():
        match = candidate[candidate["pair_id"].astype(str) == str(b["pair_id"])]
        if match.empty:
            rows.append({"pair_id": b["pair_id"], "comparison_status": "missing_candidate"})
            continue
        c = match.iloc[0]
        base_text = str(b.get("response_text", ""))
        candidate_text = str(c.get("response_text", ""))
        chosen = str(b.get("chosen_reference", ""))
        rejected = str(b.get("rejected_reference", ""))
        rows.append({
            "pair_id": b["pair_id"],
            "issue_id": b.get("issue_id", ""),
            "failure_family": b.get("failure_family", ""),
            "base_text": base_text,
            f"{args.candidate_name}_text": candidate_text,
            "exact_same": base_text.strip() == candidate_text.strip(),
            "base_candidate_similarity": similarity(base_text, candidate_text),
            "base_to_chosen_similarity": similarity(base_text, chosen),
            "candidate_to_chosen_similarity": similarity(candidate_text, chosen),
            "base_to_rejected_similarity": similarity(base_text, rejected),
            "candidate_to_rejected_similarity": similarity(candidate_text, rejected),
            "delta_similarity_to_chosen": similarity(candidate_text, chosen) - similarity(base_text, chosen),
            "delta_similarity_to_rejected": similarity(candidate_text, rejected) - similarity(base_text, rejected),
            "base_word_count": len(base_text.split()),
            "candidate_word_count": len(candidate_text.split()),
            "base_qc_flags": b.get("qc_flags", ""),
            "candidate_qc_flags": c.get("qc_flags", ""),
        })
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    by_family = {}
    for family, group in out.groupby("failure_family"):
        by_family[family] = {
            "rows": int(len(group)),
            "exact_same": int(group["exact_same"].sum()) if "exact_same" in group else 0,
            "mean_delta_similarity_to_chosen": float(group["delta_similarity_to_chosen"].mean()),
            "mean_delta_similarity_to_rejected": float(group["delta_similarity_to_rejected"].mean()),
        }
    summary = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "candidate_name": args.candidate_name,
        "rows": int(len(out)),
        "exact_same_count": int(out["exact_same"].sum()) if len(out) else 0,
        "changed_count": int((~out["exact_same"]).sum()) if len(out) else 0,
        "mean_base_candidate_similarity": float(out["base_candidate_similarity"].mean()) if len(out) else None,
        "mean_delta_similarity_to_chosen": float(out["delta_similarity_to_chosen"].mean()) if len(out) else None,
        "mean_delta_similarity_to_rejected": float(out["delta_similarity_to_rejected"].mean()) if len(out) else None,
        "by_family": by_family,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
