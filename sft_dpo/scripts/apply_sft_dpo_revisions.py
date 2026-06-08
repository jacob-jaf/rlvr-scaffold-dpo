#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED_REVISION_VALUES = {"true", "1", "yes"}
TRUE_VALUES = {"true", "1", "yes"}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_messages(raw: str) -> list[dict]:
    try:
        obj = json.loads(str(raw))
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    raise ValueError("prompt_messages_json/user_messages_json did not parse as list")


def merge_prompt_messages(row: pd.Series) -> list[dict]:
    raw = str(row.get("prompt_messages_json", "") or "")
    if raw.strip():
        return parse_messages(raw)
    # Revision sheet has user_messages_json and compact_plan_text but not necessarily prompt_messages_json.
    user_msgs = parse_messages(str(row.get("user_messages_json", "[]") or "[]"))
    compact = str(row.get("compact_plan_text", "") or "").strip()
    if not compact:
        raise ValueError(f"No prompt_messages_json or compact_plan_text for {row.get('pair_id')}")
    return [{"role": "system", "content": compact}] + user_msgs


def final_chosen(row: pd.Series, *, fail_on_missing_revision: bool) -> tuple[str, str, list[str]]:
    revised = str(row.get("chosen_response_revised", "") or "").strip()
    original = str(row.get("chosen_response_original", "") or "").strip()
    revision_required = str(row.get("revision_required", "")).strip().lower() in REQUIRED_REVISION_VALUES
    sft_ok_unrevised = str(row.get("sft_use_if_unrevised", "")).strip().lower() in TRUE_VALUES
    flags = []
    if revised:
        return revised, "human_revised", flags
    if revision_required and not sft_ok_unrevised:
        flags.append("missing_required_revision")
        if fail_on_missing_revision:
            return "", "missing", flags
    return original, "original_unrevised", flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", type=Path, required=False, help="Optional original split CSV. Not required if revision sheet has prompt/user/plan fields.")
    ap.add_argument("--revision-sheet", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--allow-missing-required-revisions", action="store_true", help="Build with original chosen responses even when revision_required=True. Not recommended.")
    args = ap.parse_args()

    rev = read_csv(args.revision_sheet)
    required_cols = ["pair_id", "split", "issue_id", "failure_family", "chosen_response_original", "chosen_response_revised", "rejected_response", "dpo_use"]
    missing = [c for c in required_cols if c not in rev.columns]
    if missing:
        raise SystemExit(f"Revision sheet missing required columns: {missing}")

    rows = []
    errors = []
    for _, r in rev.iterrows():
        chosen, chosen_source, flags = final_chosen(r, fail_on_missing_revision=not args.allow_missing_required_revisions)
        if flags and not args.allow_missing_required_revisions:
            errors.append({"pair_id": r.get("pair_id", ""), "flags": ";".join(flags), "revision_reasons": r.get("revision_reasons", "")})
            continue
        rejected = str(r.get("rejected_response", "") or "").strip()
        if not chosen:
            errors.append({"pair_id": r.get("pair_id", ""), "flags": "empty_final_chosen", "revision_reasons": r.get("revision_reasons", "")})
            continue
        if not rejected:
            errors.append({"pair_id": r.get("pair_id", ""), "flags": "empty_rejected", "revision_reasons": r.get("revision_reasons", "")})
            continue
        try:
            prompt_messages = merge_prompt_messages(r)
        except Exception as e:
            errors.append({"pair_id": r.get("pair_id", ""), "flags": f"prompt_parse_error:{e}", "revision_reasons": r.get("revision_reasons", "")})
            continue
        dpo_use = str(r.get("dpo_use", "")).strip().lower() in TRUE_VALUES
        split = str(r.get("split", "train") or "train")
        rec = {
            **r.to_dict(),
            "chosen_response_final": chosen,
            "chosen_source_for_stage": chosen_source,
            "prompt_messages_json_final": json.dumps(prompt_messages, ensure_ascii=False),
            "final_chosen_word_count": len(chosen.split()),
            "final_rejected_word_count": len(rejected.split()),
            "final_length_ratio": len(chosen.split()) / max(1, len(rejected.split())),
            "dpo_use_bool": dpo_use,
            "split": split,
        }
        rows.append(rec)

    if errors:
        errdf = pd.DataFrame(errors)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        errdf.to_csv(args.output_dir / "sft_dpo_revision_errors.csv", index=False)
        raise SystemExit(
            f"Found {len(errors)} blocking revision errors. See {args.output_dir/'sft_dpo_revision_errors.csv'}. "
            "Fill chosen_response_revised for revision_required rows or rerun with --allow-missing-required-revisions for debugging only."
        )

    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "sft_dpo_pairs_all_prepared.csv", index=False)

    sft_train=[]; sft_dev=[]; dpo_train=[]; dpo_dev=[]
    for _, r in out.iterrows():
        prompt = json.loads(str(r["prompt_messages_json_final"]))
        chosen = str(r["chosen_response_final"])
        rejected = str(r["rejected_response"])
        split = str(r["split"])
        sft_row = {
            "pair_id": r["pair_id"],
            "issue_id": r["issue_id"],
            "failure_family": r["failure_family"],
            "messages": prompt + [{"role":"assistant", "content": chosen}],
            "metadata": {"chosen_source_for_stage": r["chosen_source_for_stage"], "dpo_use": bool(r["dpo_use_bool"])}
        }
        dpo_row = {
            "pair_id": r["pair_id"],
            "issue_id": r["issue_id"],
            "failure_family": r["failure_family"],
            "prompt": prompt,
            "chosen": [{"role":"assistant", "content": chosen}],
            "rejected": [{"role":"assistant", "content": rejected}],
            "metadata": {"chosen_source_for_stage": r["chosen_source_for_stage"], "dpo_drop_reason": r.get("dpo_drop_reason", "")}
        }
        (sft_dev if split == "dev" else sft_train).append(sft_row)
        if bool(r["dpo_use_bool"]):
            (dpo_dev if split == "dev" else dpo_train).append(dpo_row)

    write_jsonl(sft_train, args.output_dir / "sft_train.jsonl")
    write_jsonl(sft_dev, args.output_dir / "sft_dev.jsonl")
    write_jsonl(dpo_train, args.output_dir / "dpo_after_sft_train.jsonl")
    write_jsonl(dpo_dev, args.output_dir / "dpo_after_sft_dev.jsonl")

    summary = {
        "revision_sheet": str(args.revision_sheet),
        "rows_total": int(len(out)),
        "sft_train_rows": len(sft_train),
        "sft_dev_rows": len(sft_dev),
        "dpo_train_rows": len(dpo_train),
        "dpo_dev_rows": len(dpo_dev),
        "dpo_excluded_rows": int((~out["dpo_use_bool"].astype(bool)).sum()),
        "chosen_source_for_stage_counts": out["chosen_source_for_stage"].value_counts().to_dict(),
        "family_counts_total": out["failure_family"].value_counts().to_dict(),
        "family_counts_dpo": out[out["dpo_use_bool"].astype(bool)]["failure_family"].value_counts().to_dict(),
        "mean_final_length_ratio": float(out["final_length_ratio"].mean()),
        "length_ratio_outside_0p75_1p50": int(((out["final_length_ratio"] < 0.75) | (out["final_length_ratio"] > 1.5)).sum()),
    }
    (args.output_dir / "sft_dpo_preparation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
