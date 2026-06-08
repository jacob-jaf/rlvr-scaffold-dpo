#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_STAGE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_STAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_STAGE_SCRIPTS))
from response_generation_qc import generation_qc_flags_raw_and_stripped, strip_redacted_thinking  # noqa: E402
from sft_dpo_utils import apply_chat_template_no_thinking  # noqa: E402
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(base_model: str, adapter: str | None, *, load_in_4bit: bool, bf16: bool):
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if bf16 else None,
        device_map="auto",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tok, model


def generate_one(
    tokenizer,
    model,
    messages,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    allow_template_fallback: bool = False,
) -> tuple[str, dict[str, Any]]:
    input_ids = apply_chat_template_no_thinking(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        allow_template_fallback=allow_template_fallback,
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(temperature=None, top_p=None))
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    new_tokens = out[0, input_ids.shape[-1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = strip_redacted_thinking(text)
    return text.strip(), {
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": int(new_tokens.shape[-1]),
        "raw_response_text": raw,
        "raw_new_text_preview": raw[:500],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SFT100 outputs for harder scaffold-dropout bank.")
    ap.add_argument("--bank", type=Path, default=Path("prepared/scaffold_dropout_harder_bank.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="../runs/training/sft_full_seed_aligned_r64_lr5e5_300steps/checkpoint-100")
    ap.add_argument("--output-csv", type=Path, default=Path("../runs/hard_cases_eval/scaffold_dropout_harder_sft100_outputs.csv"))
    ap.add_argument("--output-jsonl", type=Path, default=Path("../runs/hard_cases_eval/scaffold_dropout_harder_sft100_outputs.jsonl"))
    ap.add_argument("--summary-json", type=Path, default=Path("../runs/hard_cases_eval/scaffold_dropout_harder_sft100_outputs_summary.json"))
    ap.add_argument("--variants", nargs="*", default=None, help="Optional subset of scaffold variants to generate.")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--max-new-tokens", type=int, default=280)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--min-words", type=int, default=12)
    ap.add_argument("--max-words", type=int, default=220)
    ap.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Opt-in only: if enable_thinking=False is unsupported, use plain apply_chat_template.",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.bank)
    if args.variants:
        keep = set(args.variants)
        rows = [r for r in rows if r.get("scaffold_variant") in keep]
    tok, model = load_model(args.base_model, args.adapter, load_in_4bit=args.load_in_4bit, bf16=args.bf16)

    out_rows = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, start=1):
        text = ""
        error = ""
        meta: dict[str, Any] = {}
        try:
            text, meta = generate_one(
                tok,
                model,
                row["prompt_messages"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                allow_template_fallback=args.allow_template_fallback,
            )
        except Exception as e:
            error = repr(e)
        gen_tok = int(meta.get("generated_tokens") or 0) if meta else 0
        raw = str(meta.get("raw_response_text") or "")
        flags = generation_qc_flags_raw_and_stripped(
            raw,
            text,
            min_words=args.min_words,
            max_words_cap=args.max_words,
            max_new_tokens=args.max_new_tokens,
            generated_tokens=gen_tok or None,
        )
        out = {
            "scaffold_case_id": row.get("scaffold_case_id", ""),
            "source_pair_id": row.get("source_pair_id", ""),
            "source_split": row.get("source_split", ""),
            "issue_id": row.get("issue_id", ""),
            "failure_family": row.get("failure_family", ""),
            "target_group": row.get("target_group", ""),
            "target_stance_phrase": row.get("target_stance_phrase", ""),
            "scaffold_variant": row.get("scaffold_variant", ""),
            "variant_rank": row.get("variant_rank", ""),
            "dpo_candidate_eligible": row.get("dpo_candidate_eligible", ""),
            "hardness_note": row.get("hardness_note", ""),
            "response_text": text,
            "word_count": len(text.split()),
            "qc_flags": ";".join(flags),
            "generation_status": "error" if error else "ok",
            "error": error,
            "chosen_reference": row.get("chosen_reference", ""),
            "rejected_reference": row.get("rejected_reference", ""),
            "population": row.get("population", ""),
            "positions_text": row.get("positions_text", ""),
            "user_prompt_or_conversation": row.get("user_prompt_or_conversation", ""),
            "prompt_messages_json": json.dumps(row.get("prompt_messages", []), ensure_ascii=False),
            "generation_meta_json": json.dumps(meta, ensure_ascii=False),
        }
        out_rows.append(out)
        print(f"[gen] {i}/{len(rows)} {out['scaffold_case_id']} {out['scaffold_variant']} wc={out['word_count']} flags={out['qc_flags']}")

    df = pd.DataFrame(out_rows)
    df.to_csv(args.output_csv, index=False)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(out_rows),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "output_csv": str(args.output_csv),
        "chat_template": (
            "template_fallback_allowed"
            if args.allow_template_fallback
            else "enable_thinking_false_strict"
        ),
        "allow_template_fallback": bool(args.allow_template_fallback),
        "qc_flagged_rows": int(sum(bool(r["qc_flags"]) for r in out_rows)),
        "counts_by_variant": df["scaffold_variant"].value_counts().to_dict() if not df.empty else {},
        "counts_by_family": df["failure_family"].value_counts().to_dict() if not df.empty else {},
        "word_count_min": int(df["word_count"].min()) if not df.empty else 0,
        "word_count_median": float(df["word_count"].median()) if not df.empty else 0,
        "word_count_max": int(df["word_count"].max()) if not df.empty else 0,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
