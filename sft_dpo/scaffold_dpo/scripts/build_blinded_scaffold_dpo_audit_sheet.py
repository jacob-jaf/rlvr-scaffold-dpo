#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from pilot_utils import prompt_messages, read_jsonl, row_id, write_json

RATING_COLUMNS = [
    "human_preserves_all_major_stances_1_5",
    "human_preserves_minority_stances_1_5",
    "human_preserves_qualitative_prevalence_1_5",
    "human_avoids_numeric_leakage_1_5",
    "human_avoids_hallucinated_numbers_1_5",
    "human_avoids_wrong_ordering_1_5",
    "human_avoids_wrong_population_or_scope_transfer_1_5",
    "human_resists_user_framing_1_5",
    "human_handles_prompt_intent_1_5",
    "human_no_scaffold_intrusion_1_5",
    "human_naturalness_1_5",
    "human_non_robotic_1_5",
    "human_overall_usefulness_1_5",
    "human_pairwise_preference_notes",
    "human_notes",
]


def load_outputs(path: Path, condition: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["pair_id"] = df["pair_id"].astype(str)
    df["condition_true"] = condition
    return df


def read_prompts(path: Path) -> pd.DataFrame:
    rows = []
    for r in read_jsonl(path):
        pair_id = row_id(r)
        visible_turns = [
            f"{m['role'].upper()}: {m['content']}"
            for m in prompt_messages(r)
            if m["role"] != "system"
        ]
        rows.append({
            "pair_id": pair_id,
            "eval_case_id": str(r.get("eval_case_id") or pair_id),
            "eval_set": r.get("eval_set", ""),
            "control_type": r.get("control_type", ""),
            "issue_id": r.get("issue_id", ""),
            "failure_family": r.get("failure_family", ""),
            "scaffold_variant": r.get("scaffold_variant", ""),
            "user_prompt_or_conversation": "\n\n".join(visible_turns),
            "expected_behavior": r.get("expected_behavior", ""),
            "chosen_reference": r.get("chosen_reference", ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build blinded human audit sheet comparing SFT100 and scaffold-DPO outputs.")
    ap.add_argument("--eval-prompts", type=Path, required=True)
    ap.add_argument("--sft-output", type=Path, required=True)
    ap.add_argument("--dpo-output", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--condition-key-json", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    prompt_df = read_prompts(args.eval_prompts)
    sft = load_outputs(args.sft_output, "sft_ckpt100")
    dpo = load_outputs(args.dpo_output, "scaffold_dpo")
    all_rows = []
    rng = random.Random(args.seed)
    condition_names = ["condition_A", "condition_B"]
    key = {}
    shuffled_true = ["sft_ckpt100", "scaffold_dpo"]
    rng.shuffle(shuffled_true)
    for c, true in zip(condition_names, shuffled_true):
        key[true] = c

    for _, p in prompt_df.iterrows():
        pair_id = str(p["pair_id"])
        for df in [sft, dpo]:
            m = df[df["pair_id"] == pair_id]
            if m.empty:
                continue
            r = m.iloc[0]
            row = p.to_dict()
            row["audit_row_id"] = f"audit_{len(all_rows)+1:05d}"
            row["condition_blinded"] = key[str(r["condition_true"])]
            row["answer_text"] = str(r.get("response_text", ""))
            row["word_count"] = r.get("word_count", "")
            row["qc_flags"] = r.get("qc_flags", "")
            for col in RATING_COLUMNS:
                row[col] = ""
            all_rows.append(row)
    rng.shuffle(all_rows)
    out = pd.DataFrame(all_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    condition_key = {v: k for k, v in key.items()}
    write_json(condition_key, args.condition_key_json)
    print(json.dumps({"rows": len(out), "condition_key": condition_key, "output": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
