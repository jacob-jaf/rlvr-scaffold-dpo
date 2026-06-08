#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

WORD_RE = re.compile(r"\b[\w'-]+\b")
BAD_ENDING_RE = re.compile(
    r"(,\s*$|:\s*$|;\s*$|\b(and|or|but|because|while|although|including|such as)\s*$)",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(str(text or "")))


def looks_truncated(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if BAD_ENDING_RE.search(s):
        return True
    # Responses should usually end with sentence punctuation.
    if s[-1] not in ".!?\"')”’":
        return True
    # Dangling list/header indicators.
    tail = s[-120:].lower()
    if re.search(r"\b(arguments?|perspectives?|concerns?|counterarguments?)\s*:\s*$", tail):
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Validate generated model responses before scoring.")
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--output-flags", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--min-words", type=int, default=120)
    p.add_argument("--max-words", type=int, default=500)
    p.add_argument("--response-column", default="response_text")
    args = p.parse_args()

    df = pd.read_csv(args.responses)
    required = {"prompt_id", "issue_id", args.response_column, "generation_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required response columns: {sorted(missing)}")

    flags = df.copy()
    flags["response_word_count"] = flags[args.response_column].map(word_count)
    flags["response_nonempty"] = flags[args.response_column].astype(str).str.strip().ne("")
    flags["status_ok"] = flags["generation_status"].astype(str).eq("ok")
    flags["too_short"] = flags["response_word_count"] < args.min_words
    flags["too_long"] = flags["response_word_count"] > args.max_words
    flags["looks_truncated"] = flags[args.response_column].map(looks_truncated)
    flags["duplicate_prompt_id"] = flags["prompt_id"].duplicated(keep=False)

    flags["passes_generation_validation"] = (
        flags["status_ok"]
        & flags["response_nonempty"]
        & ~flags["too_short"]
        & ~flags["too_long"]
        & ~flags["looks_truncated"]
        & ~flags["duplicate_prompt_id"]
    )

    summary = {
        "rows": int(len(flags)),
        "issues": sorted(flags["issue_id"].astype(str).unique().tolist()),
        "rows_by_issue": flags.groupby("issue_id").size().to_dict(),
        "status_counts": flags["generation_status"].astype(str).value_counts().to_dict(),
        "nonempty_rows": int(flags["response_nonempty"].sum()),
        "too_short_rows": int(flags["too_short"].sum()),
        "too_long_rows": int(flags["too_long"].sum()),
        "looks_truncated_rows": int(flags["looks_truncated"].sum()),
        "duplicate_prompt_id_rows": int(flags["duplicate_prompt_id"].sum()),
        "passed_rows": int(flags["passes_generation_validation"].sum()),
        "pass_rate": float(flags["passes_generation_validation"].mean()) if len(flags) else None,
        "word_count_min": int(flags["response_word_count"].min()) if len(flags) else None,
        "word_count_median": float(flags["response_word_count"].median()) if len(flags) else None,
        "word_count_max": int(flags["response_word_count"].max()) if len(flags) else None,
    }

    args.output_flags.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(args.output_flags, index=False)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
