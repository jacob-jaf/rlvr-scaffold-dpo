#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pilot_utils import write_json

COMPARE_TEXT_COLS = ("sft_text", "dpo_text", "chosen_reference", "metadata_json")
GEN_TEXT_COLS = ("response_text", "chosen_reference", "rejected_reference", "expected_behavior", "metadata_json")


def _parse_output_meta(path: Path) -> dict[str, Any]:
    name = path.stem
    m_ckpt = re.search(r"_scaffold_dpo_ckpt(\d+)$", name)
    m_set = re.match(r"outputs_(.+?)_(?:sft100|scaffold_dpo)", name)
    eval_set = m_set.group(1) if m_set else ""
    if name.endswith("_sft100"):
        return {"eval_set": eval_set, "condition": "sft100", "dpo_checkpoint_step": ""}
    if m_ckpt:
        return {
            "eval_set": eval_set,
            "condition": f"scaffold_dpo_ckpt{m_ckpt.group(1)}",
            "dpo_checkpoint_step": int(m_ckpt.group(1)),
        }
    if name.endswith("_scaffold_dpo"):
        return {"eval_set": eval_set, "condition": "scaffold_dpo_final", "dpo_checkpoint_step": 40}
    return {"eval_set": eval_set, "condition": name, "dpo_checkpoint_step": ""}


def _parse_compare_meta(path: Path) -> dict[str, Any]:
    name = path.stem
    m = re.match(r"compare_(.+?)_sft100_vs_scaffold_dpo(?:_ckpt(\d+))?$", name)
    if not m:
        return {"eval_set": "", "dpo_checkpoint_step": ""}
    ckpt = m.group(2)
    return {
        "eval_set": m.group(1),
        "dpo_checkpoint_step": int(ckpt) if ckpt else 40,
        "compare_variant": f"ckpt{ckpt}" if ckpt else "final",
    }


def _flatten_summary(summary: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "eval_set": meta.get("eval_set", ""),
        "dpo_checkpoint_step": meta.get("dpo_checkpoint_step", ""),
        "compare_variant": meta.get("compare_variant", ""),
        "rows": summary.get("rows"),
        "sft_qc_flagged_rows": summary.get("sft_qc_flagged_rows"),
        "dpo_qc_flagged_rows": summary.get("dpo_qc_flagged_rows"),
        "sft_numeric_leakage_rows": summary.get("sft_numeric_leakage_rows"),
        "dpo_numeric_leakage_rows": summary.get("dpo_numeric_leakage_rows"),
        "dpo_reduced_numeric_leakage_rows": summary.get("dpo_reduced_numeric_leakage_rows"),
        "dpo_introduced_numeric_leakage_rows": summary.get("dpo_introduced_numeric_leakage_rows"),
        "dpo_possible_regression_rows": summary.get("dpo_possible_regression_rows"),
        "dpo_introduced_loop_or_overlong_rows": summary.get("dpo_introduced_loop_or_overlong_rows"),
        "dpo_reduced_loop_or_overlong_rows": summary.get("dpo_reduced_loop_or_overlong_rows"),
        "sft_output": summary.get("sft_output", ""),
        "dpo_output": summary.get("dpo_output", ""),
    }
    for side in ("sft", "dpo"):
        block = summary.get(f"{side}_loop_or_overlong_qc") or {}
        if isinstance(block, dict):
            for k, v in block.items():
                row[f"{side}_{k}"] = v
    return row


def stack_outputs(eval_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_dir.glob(pattern)):
        df = pd.read_csv(path, keep_default_na=False)
        meta = _parse_output_meta(path)
        df.insert(0, "source_file", path.name)
        df.insert(1, "dpo_checkpoint_step", meta["dpo_checkpoint_step"])
        df.insert(2, "stack_condition", meta["condition"])
        if not df.get("eval_set", pd.Series(dtype=str)).astype(str).str.len().any():
            df["eval_set"] = meta["eval_set"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def stack_compares(eval_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_dir.glob(pattern)):
        df = pd.read_csv(path, keep_default_na=False)
        meta = _parse_compare_meta(path)
        df.insert(0, "source_file", path.name)
        df.insert(1, "dpo_checkpoint_step", meta["dpo_checkpoint_step"])
        df.insert(2, "compare_variant", meta["compare_variant"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Condense scaffold-DPO eval CSVs into stacked summary files.")
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("eval"),
        help="Directory containing outputs_* and compare_* CSV/JSON files.",
    )
    args = ap.parse_args()
    eval_dir = args.eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_gen = stack_outputs(eval_dir, "outputs_*.csv")
    out_gen_slim = out_gen.drop(columns=[c for c in GEN_TEXT_COLS if c in out_gen.columns], errors="ignore")

    out_cmp = stack_compares(eval_dir, "compare_*_sft100_vs_scaffold_dpo*.csv")
    out_cmp_slim = out_cmp.drop(columns=[c for c in COMPARE_TEXT_COLS if c in out_cmp.columns], errors="ignore")

    summary_rows = []
    for path in sorted(eval_dir.glob("compare_*_summary.json")):
        if "_sft100_vs_scaffold_dpo" not in path.name:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        meta = _parse_compare_meta(path.with_suffix("").with_suffix(".csv"))
        meta["source_summary"] = path.name
        summary_rows.append(_flatten_summary(summary, meta))
    out_summaries = pd.DataFrame(summary_rows)

    paths = {
        "condensed_generation_outputs.csv": out_gen,
        "condensed_generation_outputs_slim.csv": out_gen_slim,
        "condensed_compare_rows.csv": out_cmp,
        "condensed_compare_rows_slim.csv": out_cmp_slim,
        "condensed_compare_summary_metrics.csv": out_summaries,
    }
    manifest: dict[str, Any] = {"eval_dir": str(eval_dir), "files": {}}
    for name, frame in paths.items():
        out_path = eval_dir / name
        frame.to_csv(out_path, index=False)
        manifest["files"][name] = {
            "path": str(out_path),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
        print(f"[condense] wrote {out_path} rows={len(frame)}")

    manifest_path = eval_dir / "condensed_manifest.json"
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
