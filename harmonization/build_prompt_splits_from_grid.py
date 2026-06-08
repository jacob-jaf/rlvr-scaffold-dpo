#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build_splits(prompt_grid: Path, targets_csv: Path, output_dir: Path) -> None:
    prompts = pd.read_csv(prompt_grid)
    targets = pd.read_csv(targets_csv)
    issue_versions = targets[["issue_id", "target_version"]].drop_duplicates()
    if issue_versions.groupby("issue_id").size().max() != 1:
        raise ValueError("Each issue_id should have exactly one target_version in targets file.")

    prompts = prompts.merge(issue_versions, on="issue_id", how="left")
    if prompts["target_version"].isna().any():
        missing = sorted(prompts.loc[prompts["target_version"].isna(), "issue_id"].unique())
        raise ValueError(f"Prompt grid has issue_ids missing from targets: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    main = prompts[
        (prompts["main_distributional_score_candidate"].astype(str).str.lower() == "yes")
        & (prompts["topic_polarity"].astype(str).str.lower() == "neutral")
        & (prompts["feasibility"].isin(["recommended", "usable_with_caution"]))
    ].copy()
    main["scope_class_expected"] = "core_eval"

    leading = prompts[
        (prompts["leading_or_pressure_prompt"].astype(str).str.lower() == "yes")
        & (prompts["topic_polarity"].isin(["pro", "con"]))
    ].copy()
    leading["scope_class_expected"] = "core_eval_leading_pressure"

    adjacent = prompts[
        prompts["feasibility"].isin(["adjacent_only", "stress_only", "usable_with_caution"])
        & (prompts["main_distributional_score_candidate"].astype(str).str.lower() != "yes")
    ].copy()
    adjacent["scope_class_expected"] = "adjacent_or_stress"

    main.to_csv(output_dir / "prompts_main_distributional_v1.csv", index=False)
    leading.to_csv(output_dir / "prompts_leading_pressure_v1.csv", index=False)
    adjacent.to_csv(output_dir / "prompts_adjacent_or_stress_v1.csv", index=False)

    summary = pd.DataFrame([
        {"split": "main_distributional", "rows": len(main)},
        {"split": "leading_pressure", "rows": len(leading)},
        {"split": "adjacent_or_stress", "rows": len(adjacent)},
    ])
    summary.to_csv(output_dir / "prompt_split_summary_v1.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Create v1 prompt splits from the IssueBench-style prompt grid.")
    p.add_argument("--prompt-grid", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    build_splits(args.prompt_grid, args.targets, args.output_dir)


if __name__ == "__main__":
    main()
