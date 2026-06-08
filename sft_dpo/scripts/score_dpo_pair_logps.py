#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from sft_dpo_utils import apply_chat_template_no_thinking


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_completion_messages(value: Any) -> List[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    raise TypeError(f"Unsupported completion format: {type(value)}")


def token_ids_no_thinking(tokenizer, messages: List[dict], *, add_generation_prompt: bool) -> List[int]:
    ids = apply_chat_template_no_thinking(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def prefix_len_for_completion(tokenizer, prompt_messages: List[dict], full_messages: List[dict]) -> Tuple[int, str]:
    prompt_ids = token_ids_no_thinking(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = token_ids_no_thinking(tokenizer, full_messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids), "exact_prompt_prefix"

    # Fallback: longest common prefix. This is diagnostic; exact matching is expected.
    n = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        n += 1
    return n, f"lcp_prefix_not_exact_prompt_len={len(prompt_ids)}"


def sequence_completion_logp(
    model,
    tokenizer,
    prompt_messages: List[dict],
    completion_messages: List[dict],
    *,
    device: torch.device,
    max_length: int | None,
) -> Dict[str, Any]:
    full_messages = prompt_messages + completion_messages
    full_ids_list = token_ids_no_thinking(tokenizer, full_messages, add_generation_prompt=False)
    prefix_len, prefix_status = prefix_len_for_completion(tokenizer, prompt_messages, full_messages)

    if max_length is not None and len(full_ids_list) > max_length:
        raise ValueError(f"Sequence length {len(full_ids_list)} exceeds max_length={max_length}; increase max_length.")
    if prefix_len >= len(full_ids_list):
        raise ValueError(f"No completion tokens found; prefix_len={prefix_len}, full_len={len(full_ids_list)}, status={prefix_status}")

    input_ids = torch.tensor([full_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # labels[:, i] corresponds to full_ids original token index i+1.
    original_indices = torch.arange(1, input_ids.shape[1], device=device)
    mask = original_indices >= prefix_len
    selected = token_logps[0, mask]
    completion_tokens = int(mask.sum().item())
    sum_logp = float(selected.sum().item())
    mean_logp = float(selected.mean().item()) if completion_tokens else float("nan")

    return {
        "sum_logp": sum_logp,
        "mean_token_logp": mean_logp,
        "completion_tokens": completion_tokens,
        "full_tokens": int(input_ids.shape[1]),
        "prompt_prefix_tokens": int(prefix_len),
        "prefix_status": prefix_status,
    }


def load_model_and_tokenizer(base_model: str, adapter: str | None, *, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant,
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def summarize(df: pd.DataFrame, split_name: str, model_label: str) -> dict:
    ok = df[df["score_status"] == "ok"].copy()
    out: dict[str, Any] = {
        "split": split_name,
        "model_label": model_label,
        "rows": int(len(df)),
        "ok_rows": int(len(ok)),
        "error_rows": int((df["score_status"] != "ok").sum()),
        "note": "sum_logp is sequence-level and length-sensitive; mean_token_logp is length-normalized diagnostic. DPO itself uses sequence log-probabilities, so keep length controls in mind.",
    }
    if len(ok):
        out.update({
            "sum_logp_accuracy_chosen_gt_rejected": float((ok["chosen_sum_logp"] > ok["rejected_sum_logp"]).mean()),
            "mean_token_logp_accuracy_chosen_gt_rejected": float((ok["chosen_mean_token_logp"] > ok["rejected_mean_token_logp"]).mean()),
            "mean_sum_logp_margin_chosen_minus_rejected": float(ok["sum_logp_margin"].mean()),
            "median_sum_logp_margin_chosen_minus_rejected": float(ok["sum_logp_margin"].median()),
            "mean_token_logp_margin_chosen_minus_rejected": float(ok["mean_token_logp_margin"].mean()),
            "median_token_logp_margin_chosen_minus_rejected": float(ok["mean_token_logp_margin"].median()),
            "mean_chosen_completion_tokens": float(ok["chosen_completion_tokens"].mean()),
            "mean_rejected_completion_tokens": float(ok["rejected_completion_tokens"].mean()),
        })
        by_fam = {}
        if "failure_family" in ok.columns:
            for fam, g in ok.groupby("failure_family"):
                by_fam[str(fam)] = {
                    "rows": int(len(g)),
                    "sum_logp_accuracy": float((g["chosen_sum_logp"] > g["rejected_sum_logp"]).mean()),
                    "mean_token_logp_accuracy": float((g["chosen_mean_token_logp"] > g["rejected_mean_token_logp"]).mean()),
                    "mean_sum_logp_margin": float(g["sum_logp_margin"].mean()),
                    "mean_token_logp_margin": float(g["mean_token_logp_margin"].mean()),
                }
            out["by_family"] = by_fam
        prefix_counts = pd.concat([ok["chosen_prefix_status"], ok["rejected_prefix_status"]]).value_counts().to_dict()
        out["prefix_status_counts"] = {str(k): int(v) for k, v in prefix_counts.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Static log-prob scoring for DPO chosen/rejected pairs under a model/adapter.")
    ap.add_argument("--pairs", type=Path, required=True, help="DPO JSONL with prompt/chosen/rejected fields.")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter path, e.g. SFT checkpoint-100.")
    ap.add_argument("--model-label", default="model")
    ap.add_argument("--split", default="")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.pairs)
    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.adapter, load_in_4bit=args.load_in_4bit)
    out_rows: List[dict] = []
    for i, row in enumerate(rows, start=1):
        pair_id = str(row.get("pair_id", f"row_{i}"))
        prompt = row.get("prompt")
        if prompt is None and "messages" in row:
            prompt = row["messages"][:-1]
        chosen_msgs = normalize_completion_messages(row.get("chosen"))
        rejected_msgs = normalize_completion_messages(row.get("rejected"))
        meta = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
        base_info = {
            "pair_id": pair_id,
            "issue_id": row.get("issue_id", meta.get("issue_id", "")),
            "failure_family": row.get("failure_family", meta.get("failure_family", "")),
            "chosen_text": chosen_msgs[-1].get("content", "") if chosen_msgs else "",
            "rejected_text": rejected_msgs[-1].get("content", "") if rejected_msgs else "",
        }
        try:
            chosen_score = sequence_completion_logp(model, tokenizer, prompt, chosen_msgs, device=device, max_length=args.max_length)
            rejected_score = sequence_completion_logp(model, tokenizer, prompt, rejected_msgs, device=device, max_length=args.max_length)
            out_rows.append({
                **base_info,
                "score_status": "ok",
                "chosen_sum_logp": chosen_score["sum_logp"],
                "rejected_sum_logp": rejected_score["sum_logp"],
                "sum_logp_margin": chosen_score["sum_logp"] - rejected_score["sum_logp"],
                "chosen_mean_token_logp": chosen_score["mean_token_logp"],
                "rejected_mean_token_logp": rejected_score["mean_token_logp"],
                "mean_token_logp_margin": chosen_score["mean_token_logp"] - rejected_score["mean_token_logp"],
                "chosen_completion_tokens": chosen_score["completion_tokens"],
                "rejected_completion_tokens": rejected_score["completion_tokens"],
                "chosen_full_tokens": chosen_score["full_tokens"],
                "rejected_full_tokens": rejected_score["full_tokens"],
                "chosen_prefix_status": chosen_score["prefix_status"],
                "rejected_prefix_status": rejected_score["prefix_status"],
            })
        except Exception as e:
            out_rows.append({**base_info, "score_status": "error", "error": repr(e)})
        if i % 5 == 0:
            print(f"[score] {i}/{len(rows)}")

    df = pd.DataFrame(out_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary = summarize(df, args.split or args.pairs.stem, args.model_label)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
