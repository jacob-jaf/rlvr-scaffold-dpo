#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VARIANT_PRIORITY = {
    "H8_raw_numeric_adversarial_pressure": 0,
    "H7_no_prior_anchor_ordered_options": 1,
    "H6_raw_numeric_distribution": 2,
    "H5_ordered_options_only": 3,
    "H0_full_original_control": 99,
    "H9_insufficient_prevalence_control": 99,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build human review sheet for harder scaffold-dropout DPO candidates.")
    ap.add_argument("--scored", type=Path, default=Path("../runs/hard_cases_eval/scaffold_dropout_harder_sft100_outputs_scored.csv"))
    ap.add_argument("--output-csv", type=Path, default=Path("review/scaffold_dropout_harder_dpo_pair_review_TO_LABEL.csv"))
    ap.add_argument("--summary-json", type=Path, default=Path("review/scaffold_dropout_harder_review_sheet_summary.json"))
    ap.add_argument("--source-split", default="train")
    ap.add_argument("--max-per-source", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.scored, keep_default_na=False)
    cand = df[(df["source_split"].astype(str) == args.source_split) & (df["auto_candidate_for_review"].astype(str).str.lower().isin(["true", "1", "yes"]))].copy()
    if cand.empty:
        out = pd.DataFrame()
    else:
        cand["variant_priority"] = cand["scaffold_variant"].map(VARIANT_PRIORITY).fillna(50).astype(int)
        rows = []
        for pair_id, g in cand.sort_values(["source_pair_id", "variant_priority"]).groupby("source_pair_id", sort=False):
            rows.extend(g.head(args.max_per_source).to_dict("records"))
        out_rows = []
        for i, r in enumerate(rows, start=1):
            prompt_messages_json = str(r.get("prompt_messages_json", "[]"))
            try:
                prompt_msgs = json.loads(prompt_messages_json)
                prompt_render = "\n\n".join(f"{m.get('role','').upper()}: {m.get('content','')}" for m in prompt_msgs)
            except Exception:
                prompt_render = prompt_messages_json
            review_id = f"scaffold_harder_review_{i:05d}"
            out_rows.append(
                {
                    "review_id": review_id,
                    "source_pair_id": r.get("source_pair_id", ""),
                    "source_split": r.get("source_split", ""),
                    "issue_id": r.get("issue_id", ""),
                    "failure_family": r.get("failure_family", ""),
                    "scaffold_variant": r.get("scaffold_variant", ""),
                    "hardness_note": r.get("hardness_note", ""),
                    "target_group": r.get("target_group", ""),
                    "target_stance_phrase": r.get("target_stance_phrase", ""),
                    "user_prompt_or_conversation": r.get("user_prompt_or_conversation", ""),
                    "weak_scaffold_prompt_rendered": prompt_render,
                    "weak_scaffold_prompt_messages_json": prompt_messages_json,
                    "positions_text_for_reviewer": r.get("positions_text", ""),
                    "chosen_response_proposed": r.get("chosen_reference", ""),
                    "rejected_response_sft100_weak_scaffold": r.get("response_text", ""),
                    "seed_rejected_reference": r.get("rejected_reference", ""),
                    "auto_semantic_flags": r.get("auto_semantic_flags", ""),
                    "qc_flags": r.get("qc_flags", ""),
                    "response_to_chosen_similarity": r.get("response_to_chosen_similarity", ""),
                    "response_to_rejected_similarity": r.get("response_to_rejected_similarity", ""),
                    "human_rejected_is_real_failure_1_5": "",
                    "human_chosen_is_good_for_weak_prompt_1_5": "",
                    "human_best_chosen_response_if_revision_needed": "",
                    "human_approve_for_dpo_yes_no": "",
                    "human_rejection_tags": "",
                    "human_notes": "",
                }
            )
        out = pd.DataFrame(out_rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = {
        "source": str(args.scored),
        "rows_in_review_sheet": int(len(out)),
        "source_split": args.source_split,
        "max_per_source": args.max_per_source,
        "counts_by_variant": out["scaffold_variant"].value_counts().to_dict() if not out.empty else {},
        "counts_by_family": out["failure_family"].value_counts().to_dict() if not out.empty else {},
        "output_csv": str(args.output_csv),
        "note": "Rows are auto-flagged candidates only. Reviewer should approve semantic or severe public-facing failures, not merely mild style differences.",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
