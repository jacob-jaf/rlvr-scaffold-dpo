#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pilot_utils import add_stage_scripts_to_path, prompt_messages, read_jsonl, row_id, write_json

add_stage_scripts_to_path(__file__)
from response_generation_qc import (
    generation_loop_summary_counts,
    generation_qc_flags_raw_and_stripped,
    strip_redacted_thinking,
)
from sft_dpo_utils import apply_chat_template_no_thinking


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate eval outputs with the Qwen no-thinking template.")
    ap.add_argument("--eval-prompts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--condition-name", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Use plain apply_chat_template if this tokenizer does not support enable_thinking=False.",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise SystemExit("BitsAndBytesConfig unavailable.")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quant,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = []
    for r in read_jsonl(args.eval_prompts):
        messages = prompt_messages(r)
        ptxt = apply_chat_template_no_thinking(
            tok,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            allow_template_fallback=args.allow_template_fallback,
        )
        inp = tok(ptxt, return_tensors="pt").to(model.device)
        do_sample = args.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        else:
            gen_kwargs.update(temperature=None, top_p=None)
        with torch.no_grad():
            out = model.generate(**inp, **gen_kwargs)
        new = out[0][inp["input_ids"].shape[1]:]
        raw = tok.decode(new, skip_special_tokens=True)
        text = strip_redacted_thinking(raw)
        gen_tok = int(new.shape[-1])
        flags = generation_qc_flags_raw_and_stripped(
            raw,
            text,
            max_new_tokens=args.max_new_tokens,
            generated_tokens=gen_tok,
        )
        pair_id = row_id(r)
        rows.append({
            "pair_id": pair_id,
            "eval_case_id": str(r.get("eval_case_id") or pair_id),
            "split": r.get("split", r.get("source_split", "")),
            "condition_name": args.condition_name,
            "eval_set": r.get("eval_set", ""),
            "control_type": r.get("control_type", ""),
            "issue_id": r.get("issue_id", ""),
            "taxonomy_letter": r.get("taxonomy_letter", ""),
            "failure_family": r.get("failure_family", ""),
            "scaffold_variant": r.get("scaffold_variant", ""),
            "response_text": text,
            "word_count": len(text.split()),
            "generated_tokens": gen_tok,
            "qc_flags": ";".join(flags),
            "base_model": args.base_model,
            "adapter": args.adapter,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "chosen_reference": r.get("chosen_reference", ""),
            "rejected_reference": r.get("rejected_reference", ""),
            "expected_behavior": r.get("expected_behavior", ""),
            "metadata_json": json.dumps(r.get("metadata", {}), ensure_ascii=False),
        })
        print(f"[gen] {pair_id} {args.condition_name} wc={len(text.split())} flags={';'.join(flags)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    summary = {
        "output": str(args.output),
        "eval_prompts": str(args.eval_prompts),
        "rows": len(rows),
        "condition": args.condition_name,
        "chat_template": "fallback" if args.allow_template_fallback else "qwen_no_thinking",
        "allow_template_fallback": bool(args.allow_template_fallback),
        **generation_loop_summary_counts(rows),
    }
    write_json(summary, args.output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
