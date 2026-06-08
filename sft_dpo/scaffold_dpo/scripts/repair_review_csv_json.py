#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pilot_utils import write_json


def repair_jsonish_array(value: str) -> str:
    """Escape raw control characters inside JSON strings."""
    s = str(value or "")
    out: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = not in_str
            continue
        if in_str and ch == "\n":
            out.append("\\n")
        elif in_str and ch == "\r":
            out.append("\\r")
        elif in_str and ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    repaired = "".join(out)
    json.loads(repaired)
    return repaired


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair weak_scaffold_prompt_messages_json after spreadsheet editing.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--json-column", default="weak_scaffold_prompt_messages_json")
    ap.add_argument("--summary-json", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)
    if args.json_column not in df.columns:
        raise SystemExit(f"Missing required JSON column: {args.json_column}")

    before_bad = 0
    after_bad = 0
    repaired_count = 0
    repaired_values = []
    for value in df[args.json_column].astype(str):
        try:
            json.loads(value)
            repaired_values.append(value)
        except Exception:
            before_bad += 1
            repaired = repair_jsonish_array(value)
            repaired_values.append(repaired)
            repaired_count += 1
        try:
            json.loads(repaired_values[-1])
        except Exception:
            after_bad += 1

    df[args.json_column] = repaired_values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(df)),
        "json_column": args.json_column,
        "before_bad_json_rows": int(before_bad),
        "after_bad_json_rows": int(after_bad),
        "repaired_rows": int(repaired_count),
    }
    if args.summary_json:
        write_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
