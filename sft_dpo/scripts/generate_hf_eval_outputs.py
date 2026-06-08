#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from response_generation_qc import generation_qc_flags_raw_and_stripped
from sft_dpo_utils import apply_chat_template_no_thinking, read_jsonl, strip_thinking, word_count


def load_model_and_tokenizer(base_model: str, adapter: str | None, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise SystemExit("BitsAndBytesConfig unavailable. Install bitsandbytes or omit --load-in-4bit.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if adapter:
        from peft import PeftModel
        adapter_path = Path(adapter).resolve()
        if not (adapter_path / "adapter_config.json").exists():
            raise SystemExit(f"Missing adapter at {adapter_path} (adapter_config.json not found).")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate HF outputs with Qwen enable_thinking=False (matches SFT/DPO training)."
    )
    p.add_argument("--eval-prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--condition-name", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter", default="")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Opt-in only: if enable_thinking=False is unsupported, use plain apply_chat_template (matches DPO allow_template_fallback=true).",
    )
    args = p.parse_args()

    rows = read_jsonl(args.eval_prompts)
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter or None, args.load_in_4bit)

    out_rows = []
    for r in rows:
        prompt_text = apply_chat_template_no_thinking(
            tokenizer,
            r["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            allow_template_fallback=args.allow_template_fallback,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        do_sample = args.temperature > 0
        with __import__("torch").no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        gen_tok = int(new_tokens.shape[-1])
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = strip_thinking(raw)
        flags = generation_qc_flags_raw_and_stripped(
            raw,
            text,
            max_new_tokens=args.max_new_tokens,
            generated_tokens=gen_tok,
        )
        out_rows.append({
            "pair_id": r["pair_id"],
            "split": r.get("split", ""),
            "condition_name": args.condition_name,
            "issue_id": r.get("issue_id", ""),
            "taxonomy_letter": r.get("taxonomy_letter", ""),
            "failure_family": r.get("failure_family", ""),
            "response_text": text,
            "raw_response_text": raw,
            "word_count": word_count(text),
            "generated_tokens": gen_tok,
            "qc_flags": ";".join(flags),
            "base_model": args.base_model,
            "adapter": args.adapter,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "chosen_reference": r.get("chosen_reference", ""),
            "rejected_reference": r.get("rejected_reference", ""),
        })
        print(f"[gen] {r['pair_id']} {args.condition_name} wc={word_count(text)} flags={out_rows[-1]['qc_flags']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.output, index=False)
    summary = {
        "output": str(args.output),
        "rows": len(out_rows),
        "condition": args.condition_name,
        "chat_template": (
            "template_fallback_allowed"
            if args.allow_template_fallback
            else "enable_thinking_false_strict"
        ),
        "allow_template_fallback": bool(args.allow_template_fallback),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
