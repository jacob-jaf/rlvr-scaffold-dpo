#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

SCORE_COLS = [
    "human_preserves_all_major_stances_1_5",
    "human_preserves_minority_stances_1_5",
    "human_preserves_qualitative_prevalence_1_5",
    "human_avoids_flattening_1_5",
    "human_resists_user_framing_1_5",
    "human_handles_prompt_intent_1_5",
    "human_avoids_numeric_leakage_1_5",
    "human_avoids_plan_or_data_leakage_1_5",
    "human_avoids_wrong_population_or_scope_transfer_1_5",
    "human_no_scaffold_intrusion_1_5",
    "human_naturalness_1_5",
    "human_non_robotic_1_5",
    "human_neutrality_1_5",
    "human_scope_discipline_1_5",
    "human_overall_usefulness_1_5",
]

INVALID_DPO_NEGATIVE_PAIR_IDS = {"dpo_v3_nat_000071"}


def read_csv_safe(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(str(text or "").split())


def strip_thinking(text: str) -> str:
    raw = str(text or "")
    for open_tag, close_tag in (
        ("<" + "think" + ">", "</" + "think" + ">"),
        ("<" + "redacted_thinking" + ">", "</" + "redacted_thinking" + ">"),
    ):
        while True:
            start = raw.lower().find(open_tag)
            if start == -1:
                break
            end = raw.lower().find(close_tag, start)
            if end == -1:
                break
            raw = raw[:start] + raw[end + len(close_tag):]
    return raw.strip()


_NO_THINK_TEMPLATE_ERROR = (
    "Tokenizer does not accept enable_thinking=False. Do not run generation or scoring with a "
    "mismatched Qwen thinking template. Upgrade transformers/Qwen tokenizer support or pass "
    "allow_template_fallback=True only after verifying alignment."
)


def apply_chat_template_no_thinking(
    tokenizer,
    messages,
    *,
    allow_template_fallback: bool = False,
    **kwargs,
):
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError as e:
        if not allow_template_fallback:
            raise RuntimeError(_NO_THINK_TEMPLATE_ERROR) from e
        return tokenizer.apply_chat_template(messages, **kwargs)


def eval_prompt_from_sft_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT row must end with final assistant message.")
    return list(messages[:-1])


def eval_prompt_from_sft_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    return eval_prompt_from_sft_messages(row["messages"])


def validate_eval_prompts_alignment(
    sft_rows_by_id: dict[str, dict[str, Any]],
    eval_rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    missing = sorted(set(sft_rows_by_id) - set(eval_rows_by_id))
    for pair_id in missing:
        failures.append({"pair_id": pair_id, "issue": "missing_eval_prompt"})
    for pair_id, sft_row in sorted(sft_rows_by_id.items()):
        eval_row = eval_rows_by_id.get(pair_id)
        if eval_row is None:
            continue
        expected = eval_prompt_from_sft_row(sft_row)
        got = eval_row.get("prompt", [])
        if got != expected:
            failures.append(
                {
                    "pair_id": pair_id,
                    "issue": "prompt_mismatch",
                    "expected_roles": ",".join(m.get("role", "") for m in expected),
                    "got_roles": ",".join(m.get("role", "") for m in got),
                }
            )
        target = sft_row["messages"][-1]
        target_content = str(target.get("content", ""))
        if any(str(m.get("content", "")) == target_content for m in got):
            failures.append({"pair_id": pair_id, "issue": "target_leaked_into_eval_prompt"})
    return failures


def qc_text(
    text: str,
    *,
    max_new_tokens: int | None = None,
    generated_tokens: int | None = None,
) -> list[str]:
    from response_generation_qc import generation_qc_flags

    return generation_qc_flags(
        text,
        max_new_tokens=max_new_tokens,
        generated_tokens=generated_tokens,
    )


def safe_json_loads(raw: str, default=None):
    try:
        return json.loads(str(raw))
    except Exception:
        return default


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_sft_messages_from_row(row: pd.Series, chosen_col: str = "chosen_response_sft_dpo_final") -> list[dict[str, str]]:
    prompt = safe_json_loads(row.get("prompt_messages_json", "[]"), [])
    chosen = str(row.get(chosen_col, "") or "").strip()
    return list(prompt) + [{"role": "assistant", "content": chosen}]


def make_dpo_row_from_row(row: pd.Series, chosen_col: str = "chosen_response_sft_dpo_final") -> dict[str, Any]:
    prompt = safe_json_loads(row.get("prompt_messages_json", "[]"), [])
    return {
        "pair_id": str(row.get("pair_id", "")),
        "issue_id": str(row.get("issue_id", "")),
        "taxonomy_letter": str(row.get("taxonomy_letter", "")),
        "failure_family": str(row.get("failure_family", "")),
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": str(row.get(chosen_col, "") or "").strip()}],
        "rejected": [{"role": "assistant", "content": str(row.get("response_text", "") or "").strip()}],
        "metadata": {
            "candidate_id": str(row.get("candidate_id", "")),
            "sample_id": str(row.get("sample_id", "")),
            "chosen_source_final": str(row.get("chosen_source_final", "")),
            "sft_dpo_revision_status": str(row.get("revision_status", "")),
            "drop_from_dpo_reason": str(row.get("drop_from_dpo_reason", "")),
        }
    }
