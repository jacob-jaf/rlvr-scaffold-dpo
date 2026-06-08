#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_json_dict(x: Any) -> dict[str, float]:
    if isinstance(x, dict):
        return {str(k): float(v) for k, v in x.items()}
    try:
        obj = json.loads(str(x))
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def top_key(d: dict[str, float]) -> str:
    if not d:
        return ""
    return max(d.items(), key=lambda kv: kv[1])[0]


def entropy(d: dict[str, float]) -> float:
    vals = [v for v in d.values() if v > 0]
    return float(-sum(v * math.log(v) for v in vals))


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate scored ANES model outputs.")
    p.add_argument("--scored", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=None, help="Optional prompt split CSV to merge prompt metadata.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--low-coverage-threshold", type=float, default=0.50)
    p.add_argument("--high-js-threshold", type=float, default=0.15)
    args = p.parse_args()

    df = pd.read_csv(args.scored)
    if args.prompts is not None and args.prompts.exists():
        prompts = pd.read_csv(args.prompts)
        keep_cols = [
            c for c in [
                "prompt_id", "template_id", "template_family", "topic_polarity",
                "priority", "feasibility", "recommended_project_use",
                "main_distributional_score_candidate", "leading_or_pressure_prompt",
                "scope_class_expected"
            ]
            if c in prompts.columns
        ]
        if "prompt_id" in keep_cols:
            meta = prompts[keep_cols].drop_duplicates("prompt_id")
            df = df.merge(meta, on="prompt_id", how="left", suffixes=("", "_prompt"))

    required = {"issue_id", "predicted_distribution_json", "target_distribution_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing scored columns: {sorted(missing)}")

    df["predicted_distribution"] = df["predicted_distribution_json"].map(parse_json_dict)
    df["target_distribution"] = df["target_distribution_json"].map(parse_json_dict)
    df["top_predicted_bin_calc"] = df["predicted_distribution"].map(top_key)
    df["top_target_bin"] = df["target_distribution"].map(top_key)
    if "top_predicted_bin" not in df.columns:
        df["top_predicted_bin"] = df["top_predicted_bin_calc"]
    df["top_bin_matches_target"] = df["top_predicted_bin"].astype(str).eq(df["top_target_bin"].astype(str))
    df["predicted_entropy"] = df["predicted_distribution"].map(entropy)
    df["target_entropy"] = df["target_distribution"].map(entropy)

    for col in ["distribution_fidelity_l1", "l1_distance", "js_divergence", "coverage_share", "max_descriptor_similarity", "max_phrase_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "score_confidence" not in df.columns:
        low_mask = (df.get("coverage_share", pd.Series([0.0] * len(df))).fillna(0) == 0) & (
            df.get("max_phrase_score", pd.Series([0.0] * len(df))).fillna(0) == 0
        )
        df["score_confidence"] = low_mask.map(lambda is_low: "low" if is_low else "medium")
    df["score_confidence"] = df["score_confidence"].astype(str).str.lower()
    scoreable_mask = ~df["score_confidence"].eq("low")
    df["is_scoreable"] = scoreable_mask

    # Row flags.
    df["flag_low_coverage"] = df.get("coverage_share", pd.Series([float("nan")] * len(df))).fillna(0) < args.low_coverage_threshold
    df["flag_low_evidence"] = df["score_confidence"].eq("low")
    df["flag_high_js"] = df.get("js_divergence", pd.Series([float("nan")] * len(df))).fillna(0) > args.high_js_threshold
    df["flag_top_bin_mismatch"] = ~df["top_bin_matches_target"]
    df["flag_scoring_review"] = df["flag_low_coverage"] | df["flag_low_evidence"] | df["flag_high_js"] | df["flag_top_bin_mismatch"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Overall summary.
    overall = {
        "rows": int(len(df)),
        "issues": sorted(df["issue_id"].astype(str).unique().tolist()),
        "score_status_counts": df["score_status"].astype(str).value_counts().to_dict() if "score_status" in df.columns else {},
        "mean_distribution_fidelity_l1": float(df["distribution_fidelity_l1"].mean()) if "distribution_fidelity_l1" in df.columns else None,
        "mean_l1_distance": float(df["l1_distance"].mean()) if "l1_distance" in df.columns else None,
        "mean_js_divergence": float(df["js_divergence"].mean()) if "js_divergence" in df.columns else None,
        "mean_coverage_share": float(df["coverage_share"].mean()) if "coverage_share" in df.columns else None,
        "score_confidence_counts": df["score_confidence"].value_counts().to_dict(),
        "scoreable_rows": int(scoreable_mask.sum()),
        "scoreable_share": float(scoreable_mask.mean()) if len(df) else None,
        "mean_distribution_fidelity_l1_scoreable_only": float(df.loc[scoreable_mask, "distribution_fidelity_l1"].mean()) if "distribution_fidelity_l1" in df.columns and scoreable_mask.any() else None,
        "mean_l1_distance_scoreable_only": float(df.loc[scoreable_mask, "l1_distance"].mean()) if "l1_distance" in df.columns and scoreable_mask.any() else None,
        "mean_js_divergence_scoreable_only": float(df.loc[scoreable_mask, "js_divergence"].mean()) if "js_divergence" in df.columns and scoreable_mask.any() else None,
        "top_bin_match_rate": float(df["top_bin_matches_target"].mean()) if len(df) else None,
        "top_bin_match_rate_scoreable_only": float(df.loc[scoreable_mask, "top_bin_matches_target"].mean()) if scoreable_mask.any() else None,
        "flagged_for_review_rows": int(df["flag_scoring_review"].sum()),
    }
    (args.output_dir / "scored_overall_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    # By issue.
    agg_cols = {}
    for col in ["distribution_fidelity_l1", "l1_distance", "js_divergence", "coverage_share", "max_descriptor_similarity", "max_phrase_score"]:
        if col in df.columns:
            agg_cols[col] = ["mean", "median", "min", "max"]
    by_issue = df.groupby("issue_id").agg(
        rows=("prompt_id", "count"),
        scoreable_rate=("is_scoreable", "mean"),
        top_bin_match_rate=("top_bin_matches_target", "mean"),
        review_flag_rate=("flag_scoring_review", "mean"),
        **{f"{col}_{stat}": (col, stat) for col, stats in agg_cols.items() for stat in stats}
    ).reset_index()
    by_issue.to_csv(args.output_dir / "scored_summary_by_issue.csv", index=False)

    # By template family if present.
    if "template_family" in df.columns:
        by_family = df.groupby("template_family").agg(
            rows=("prompt_id", "count"),
            fidelity_mean=("distribution_fidelity_l1", "mean") if "distribution_fidelity_l1" in df.columns else ("prompt_id", "count"),
            coverage_mean=("coverage_share", "mean") if "coverage_share" in df.columns else ("prompt_id", "count"),
            js_mean=("js_divergence", "mean") if "js_divergence" in df.columns else ("prompt_id", "count"),
            review_flag_rate=("flag_scoring_review", "mean"),
        ).reset_index()
        by_family.to_csv(args.output_dir / "scored_summary_by_template_family.csv", index=False)

    # Average predicted vs target by issue/bin.
    dist_rows = []
    for issue_id, g in df.groupby("issue_id"):
        bins = sorted(set().union(*[set(x.keys()) for x in g["target_distribution"]]))
        for b in bins:
            pred_vals = [d.get(b, 0.0) for d in g["predicted_distribution"]]
            targ_vals = [d.get(b, 0.0) for d in g["target_distribution"]]
            dist_rows.append({
                "issue_id": issue_id,
                "stance_bin_id": b,
                "mean_predicted_share": sum(pred_vals) / len(pred_vals) if pred_vals else None,
                "target_share": sum(targ_vals) / len(targ_vals) if targ_vals else None,
                "mean_minus_target": (sum(pred_vals) / len(pred_vals) - sum(targ_vals) / len(targ_vals)) if pred_vals else None,
                "n_prompts": len(g),
            })
    pd.DataFrame(dist_rows).to_csv(args.output_dir / "scored_mean_predicted_vs_target_by_issue_bin.csv", index=False)

    df.to_csv(args.output_dir / "scored_rows_with_flags.csv", index=False)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
