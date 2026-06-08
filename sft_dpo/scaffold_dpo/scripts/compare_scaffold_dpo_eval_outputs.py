#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pilot_utils import add_stage_scripts_to_path, read_jsonl, row_id, write_json

add_stage_scripts_to_path(__file__)
from response_generation_qc import (
    LOOP_OVERLONG_FLAGS,
    eval_compare_flags,
    has_any_loop_or_overlong,
    length_and_loop_flags,
    loop_flags_only,
    loop_summary_counts,
    merge_flags,
)


def read_outputs(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if "pair_id" not in df.columns:
        raise SystemExit(f"{path} lacks pair_id")
    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    df[f"{label}_text"] = df["response_text"].astype(str)
    df[f"{label}_qc_flags"] = df.get("qc_flags", "").astype(str)
    df[f"{label}_word_count"] = df.get("word_count", 0)
    keep = ["pair_id", f"{label}_text", f"{label}_qc_flags", f"{label}_word_count"]
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare SFT100 vs scaffold-dropout DPO outputs on eval sets.")
    ap.add_argument("--eval-prompts", type=Path, required=True)
    ap.add_argument("--sft-output", type=Path, required=True)
    ap.add_argument("--dpo-output", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    args = ap.parse_args()

    refs = []
    for r in read_jsonl(args.eval_prompts):
        pair_id = row_id(r)
        refs.append({
            "pair_id": pair_id,
            "eval_case_id": str(r.get("eval_case_id") or pair_id),
            "eval_set": r.get("eval_set", ""),
            "control_type": r.get("control_type", ""),
            "issue_id": r.get("issue_id", ""),
            "failure_family": r.get("failure_family", ""),
            "scaffold_variant": r.get("scaffold_variant", ""),
            "expected_behavior": r.get("expected_behavior", ""),
            "chosen_reference": r.get("chosen_reference", ""),
            "metadata_json": json.dumps(r.get("metadata", {}), ensure_ascii=False),
        })
    ref = pd.DataFrame(refs)
    sft = read_outputs(args.sft_output, "sft")
    dpo = read_outputs(args.dpo_output, "dpo")
    out = ref.merge(sft, on="pair_id", how="left").merge(dpo, on="pair_id", how="left")

    rows = []
    for _, r in out.iterrows():
        sflags = merge_flags(str(r.get("sft_qc_flags", "")).split(";"))
        dflags = merge_flags(str(r.get("dpo_qc_flags", "")).split(";"))
        s_eval = eval_compare_flags(r.get("sft_text", ""), control_type=r.get("control_type", ""))
        d_eval = eval_compare_flags(r.get("dpo_text", ""), control_type=r.get("control_type", ""))
        s_loop = length_and_loop_flags(r.get("sft_text", ""))
        d_loop = length_and_loop_flags(r.get("dpo_text", ""))
        s_all = merge_flags(sflags, s_eval, s_loop)
        d_all = merge_flags(dflags, d_eval, d_loop)
        row = r.to_dict()
        row["sft_eval_flags"] = ";".join(s_eval)
        row["dpo_eval_flags"] = ";".join(d_eval)
        row["sft_loop_flags"] = ";".join(loop_flags_only(s_loop))
        row["dpo_loop_flags"] = ";".join(loop_flags_only(d_loop))
        row["sft_any_flags"] = ";".join(s_all)
        row["dpo_any_flags"] = ";".join(d_all)
        row["dpo_flag_count_delta_vs_sft"] = len(d_all) - len(s_all)
        row["dpo_reduced_numeric_leakage"] = int(
            "numeric_or_approx_numeric_leakage" in s_all
            and "numeric_or_approx_numeric_leakage" not in d_all
        )
        row["dpo_introduced_numeric_leakage"] = int(
            "numeric_or_approx_numeric_leakage" not in s_all
            and "numeric_or_approx_numeric_leakage" in d_all
        )
        row["dpo_introduced_loop_or_overlong"] = int(
            not has_any_loop_or_overlong(s_all) and has_any_loop_or_overlong(d_all)
        )
        row["dpo_reduced_loop_or_overlong"] = int(
            has_any_loop_or_overlong(s_all) and not has_any_loop_or_overlong(d_all)
        )
        row["dpo_possible_regression"] = int(len(d_all) > len(s_all))
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    def count_contains(series: pd.Series, term: str) -> int:
        return int(series.astype(str).str.contains(term, regex=False).sum())

    summary: dict[str, Any] = {
        "eval_prompts": str(args.eval_prompts),
        "sft_output": str(args.sft_output),
        "dpo_output": str(args.dpo_output),
        "rows": int(len(frame)),
        "by_variant": frame.groupby("scaffold_variant").size().to_dict() if "scaffold_variant" in frame.columns else {},
        "by_control_type": frame.groupby("control_type").size().to_dict() if "control_type" in frame.columns else {},
        "sft_qc_flagged_rows": int(frame["sft_any_flags"].astype(str).ne("").sum()),
        "dpo_qc_flagged_rows": int(frame["dpo_any_flags"].astype(str).ne("").sum()),
        "sft_numeric_leakage_rows": count_contains(frame["sft_any_flags"], "numeric_or_approx_numeric_leakage"),
        "dpo_numeric_leakage_rows": count_contains(frame["dpo_any_flags"], "numeric_or_approx_numeric_leakage"),
        "dpo_reduced_numeric_leakage_rows": int(frame["dpo_reduced_numeric_leakage"].sum()),
        "dpo_introduced_numeric_leakage_rows": int(frame["dpo_introduced_numeric_leakage"].sum()),
        "dpo_possible_regression_rows": int(frame["dpo_possible_regression"].sum()),
        "sft_loop_or_overlong_qc": loop_summary_counts(frame, "sft_any_flags"),
        "dpo_loop_or_overlong_qc": loop_summary_counts(frame, "dpo_any_flags"),
        "dpo_introduced_loop_or_overlong_rows": int(frame["dpo_introduced_loop_or_overlong"].sum()),
        "dpo_reduced_loop_or_overlong_rows": int(frame["dpo_reduced_loop_or_overlong"].sum()),
        "loop_overlong_flag_names": sorted(LOOP_OVERLONG_FLAGS),
    }
    write_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
