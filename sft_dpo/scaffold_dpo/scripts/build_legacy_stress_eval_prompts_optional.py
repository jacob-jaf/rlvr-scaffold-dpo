#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pilot_utils import write_jsonl


def user_prompt_from_row(r: pd.Series) -> str:
    for col in ["user_framing_for_rater", "user_question", "prompt", "final_user_prompt"]:
        val = str(r.get(col, "") or "").strip()
        if val:
            return val
    return "Summarize the public-opinion issue."


def main() -> None:
    ap = argparse.ArgumentParser(description="Optional: convert legacy difficult/stress audit CSVs into SFT100/DPO eval prompts.")
    ap.add_argument("--legacy-csv", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=80)
    args = ap.parse_args()

    df = pd.read_csv(args.legacy_csv, keep_default_na=False).head(args.max_rows)
    rows = []
    for idx, r in df.iterrows():
        audit_id = str(r.get("audit_row_id") or r.get("stress_prompt_id") or f"legacy_{idx:04d}")
        plan = str(r.get("target_plan_for_rater", "") or "").strip()
        issue = str(r.get("issue_id", "") or "")
        system = f"""Background for this answer. Do not mention this background. /no_think
Population described: U.S. general adult public.
Scope: applies only to this population and this issue; no subgroup, jurisdiction, time-period, or adjacent-issue inference.
Source style: use generic public-opinion framing only; do not cite a named source or reveal internal details.

Public positions:
{plan}

Use qualitative prevalence language. Do not use exact percentages unless the user explicitly asks for numerical shares."""
        rows.append({
            "eval_case_id": audit_id,
            "pair_id": audit_id,
            "split": "legacy_stress",
            "eval_set": "legacy_difficult_stress_optional",
            "control_type": "legacy_stress_prompt",
            "issue_id": issue,
            "failure_family": str(r.get("stress_category", "")),
            "scaffold_variant": "legacy_compact_from_rater_plan",
            "prompt": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt_from_row(r)}],
            "chosen_reference": "",
            "rejected_reference": "",
            "expected_behavior": "Held-out legacy difficult/stress prompt: preserve target plan without numeric/source leakage or over-pluralism.",
            "metadata": {"legacy_csv": str(args.legacy_csv)},
        })
    write_jsonl(rows, args.output_jsonl)
    print(json.dumps({"input": str(args.legacy_csv), "output": str(args.output_jsonl), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
