#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def parse_codes(s: str) -> list[int]:
    return [int(x) for x in json.loads(s)]


def compute_targets(anes_csv: Path, harmonization_csv: Path, stance_bins_csv: Path, output: Path, target_version: str) -> None:
    issues = pd.read_csv(harmonization_csv)
    bins = pd.read_csv(stance_bins_csv)

    if target_version == "fresh_papi":
        target_version_name = "anes2024_fresh_papi_v1"
        weight_col = "primary_weight_fresh"
    elif target_version == "full_papi":
        target_version_name = "anes2024_full_papi_v1"
        weight_col = "primary_weight_full"
    else:
        raise ValueError("--target-version must be fresh_papi or full_papi")

    usecols = sorted(set(issues["anes_variable"].tolist() + issues[weight_col].tolist()))
    df = pd.read_csv(anes_csv, usecols=usecols, low_memory=False)
    for c in usecols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for _, issue in issues.iterrows():
        issue_id = issue["issue_id"]
        var = issue["anes_variable"]
        weight = issue[weight_col]
        issue_bins = bins[(bins["issue_id"] == issue_id) & (bins["include_in_main_target"].astype(str).str.lower().isin(["true", "1", "yes"]))].copy()
        valid_codes = sorted({c for raw in issue_bins["raw_codes"] for c in parse_codes(raw)})

        valid = df[var].isin(valid_codes) & df[weight].notna() & (df[weight] > 0)
        valid_unweighted_n = int(valid.sum())
        valid_weighted_n = float(df.loc[valid, weight].sum())

        if valid_weighted_n <= 0:
            raise ValueError(f"No valid weighted cases for {issue_id}: {var}/{weight}")

        missing_or_nontarget = (~df[var].isin(valid_codes)) & df[weight].notna() & (df[weight] > 0)
        missing_or_nontarget_n = int(missing_or_nontarget.sum())

        for _, b in issue_bins.sort_values("stance_order").iterrows():
            codes = parse_codes(b["raw_codes"])
            in_bin = valid & df[var].isin(codes)
            weighted_bin = float(df.loc[in_bin, weight].sum())
            unweighted_bin = int(in_bin.sum())
            rows.append({
                "issue_id": issue_id,
                "stance_bin_id": b["stance_bin_id"],
                "stance_order": int(b["stance_order"]),
                "stance_label": b["stance_label"],
                "raw_codes": b["raw_codes"],
                "target_share": weighted_bin / valid_weighted_n,
                "unweighted_n_bin": unweighted_bin,
                "unweighted_n_valid": valid_unweighted_n,
                "weighted_n_bin": weighted_bin,
                "weighted_n_valid": valid_weighted_n,
                "missing_or_nontarget_unweighted_n": missing_or_nontarget_n,
                "anes_variable": var,
                "weight_variable": weight,
                "target_version": target_version_name,
                "source_tier": "gold_anes",
            })

    out = pd.DataFrame(rows).sort_values(["issue_id", "stance_order"])
    sums = out.groupby("issue_id")["target_share"].sum()
    bad = sums[(sums - 1.0).abs() > 1e-8]
    if len(bad):
        raise RuntimeError(f"Target shares do not sum to 1:\n{bad}")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute ANES target stance distributions from harmonization files.")
    p.add_argument("--anes-csv", type=Path, required=True)
    p.add_argument("--harmonization", type=Path, required=True)
    p.add_argument("--stance-bins", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-version", choices=["fresh_papi", "full_papi"], default="fresh_papi")
    args = p.parse_args()
    compute_targets(args.anes_csv, args.harmonization, args.stance_bins, args.output, args.target_version)


if __name__ == "__main__":
    main()
