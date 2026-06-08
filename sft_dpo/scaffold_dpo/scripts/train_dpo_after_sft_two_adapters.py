#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from pilot_utils import read_jsonl, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO from one base model with train and reference adapters.")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sft-adapter", type=Path, required=True, help="Selected SFT adapter checkpoint, e.g. checkpoint-100")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base_model = args.base_model or cfg.get("base_model", "Qwen/Qwen3-8B")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    quant = None
    if args.load_in_4bit or cfg.get("load_in_4bit", False):
        if BitsAndBytesConfig is None:
            raise SystemExit("BitsAndBytesConfig unavailable; install bitsandbytes or omit --load-in-4bit.")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        quantization_config=quant,
    )
    base.config.use_cache = False

    model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True, adapter_name="default")
    model.load_adapter(str(args.sft_adapter), adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    model.train()

    for name, p in model.named_parameters():
        if "reference" in name:
            p.requires_grad_(False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def render_prompt(messages: list[dict[str, str]]) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as e:
            if cfg.get("allow_template_fallback", False):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raise RuntimeError(
                "Tokenizer does not support enable_thinking=False; update the tokenizer or set allow_template_fallback."
            ) from e

    max_length = int(cfg.get("max_length", 6144))
    beta = float(cfg.get("beta", 0.05))

    def encode_prompt_completion(prompt_messages: list[dict[str, str]], completion: str) -> dict[str, torch.Tensor]:
        prompt_text = render_prompt(prompt_messages)
        target_text = str(completion or '').strip()
        if tokenizer.eos_token and not target_text.endswith(tokenizer.eos_token):
            target_text += tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1]

        max_prompt = max_length - len(target_ids)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        input_ids = prompt_ids + target_ids
        device = model.device
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device),
            "labels": torch.tensor([[-100] * len(prompt_ids) + target_ids], dtype=torch.long, device=device),
            "target_tokens": torch.tensor(len(target_ids), dtype=torch.long),
            "prompt_tokens": torch.tensor(len(prompt_ids), dtype=torch.long),
        }

    def sequence_logprob(batch: dict[str, torch.Tensor], adapter_name: str, requires_grad: bool) -> torch.Tensor:
        model.set_adapter(adapter_name)
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1, :]
            labels = batch["labels"][:, 1:]
            mask = labels.ne(-100)
            safe_labels = labels.masked_fill(~mask, 0)
            log_probs = F.log_softmax(logits, dim=-1)
            tok_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            return (tok_log_probs * mask).sum(dim=-1).squeeze(0)

    def row_logps(row: dict[str, Any], *, policy_grad: bool) -> dict[str, torch.Tensor]:
        prompt = row["prompt"]
        chosen = row["chosen"][0]["content"]
        rejected = row["rejected"][0]["content"]
        chosen_batch = encode_prompt_completion(prompt, chosen)
        rejected_batch = encode_prompt_completion(prompt, rejected)
        pi_c = sequence_logprob(chosen_batch, "default", requires_grad=policy_grad)
        pi_r = sequence_logprob(rejected_batch, "default", requires_grad=policy_grad)
        ref_c = sequence_logprob(chosen_batch, "reference", requires_grad=False)
        ref_r = sequence_logprob(rejected_batch, "reference", requires_grad=False)
        model.set_adapter("default")
        return {"pi_c": pi_c, "pi_r": pi_r, "ref_c": ref_c, "ref_r": ref_r}

    def scalar(x: torch.Tensor) -> float:
        return float(x.detach().cpu())

    def dpo_loss_from_logps(logps: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        pi_logratio = logps["pi_c"] - logps["pi_r"]
        ref_logratio = logps["ref_c"] - logps["ref_r"]
        margin = pi_logratio - ref_logratio
        loss = -F.logsigmoid(beta * margin)
        with torch.no_grad():
            chosen_reward = beta * (logps["pi_c"] - logps["ref_c"])
            rejected_reward = beta * (logps["pi_r"] - logps["ref_r"])
            out = {
                "loss": scalar(loss),
                "margin": scalar(margin),
                "accuracy": scalar((margin > 0).float()),
                "chosen_reward": scalar(chosen_reward),
                "rejected_reward": scalar(rejected_reward),
                "pi_chosen_logp": scalar(logps["pi_c"]),
                "pi_rejected_logp": scalar(logps["pi_r"]),
                "ref_chosen_logp": scalar(logps["ref_c"]),
                "ref_rejected_logp": scalar(logps["ref_r"]),
            }
        return loss, out

    @torch.no_grad()
    def evaluate(rows: list[dict[str, Any]], split: str, step: int) -> dict[str, Any]:
        model.eval()
        vals: list[dict[str, float]] = []
        for row in rows:
            logps = row_logps(row, policy_grad=False)
            _, stats = dpo_loss_from_logps(logps)
            vals.append(stats)
        model.train()
        keys = vals[0].keys() if vals else []
        mean = {k: (sum(v[k] for v in vals) / len(vals) if vals else None) for k in keys}
        return {"step": step, "split": split, "n": len(rows), **{f"eval_{k}": v for k, v in mean.items()}}

    def save_adapter(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.set_adapter("default")
        try:
            model.save_pretrained(str(path), selected_adapters=["default"])
        except TypeError:
            model.save_pretrained(str(path))
        tokenizer.save_pretrained(str(path))

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)

    lr = float(cfg.get("learning_rate", 2e-6))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))
    max_steps = int(cfg.get("max_steps", 60))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    eval_steps = int(cfg.get("eval_steps", 10))
    save_steps = int(cfg.get("save_steps", 10))
    logging_steps = int(cfg.get("logging_steps", 1))
    seed = int(cfg.get("seed", 42))

    random.seed(seed)
    torch.manual_seed(seed)

    logs: list[dict[str, Any]] = []
    start = time.time()
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    shuffled = train_rows.copy()
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        if micro_step % len(shuffled) == 0:
            random.shuffle(shuffled)
        row = shuffled[micro_step % len(shuffled)]
        micro_step += 1
        logps = row_logps(row, policy_grad=True)
        loss, stats = dpo_loss_from_logps(logps)
        (loss / grad_accum).backward()
        accum_loss += float(loss.detach().cpu())

        if micro_step % grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            record = {
                "step": global_step,
                "pair_id": row.get("pair_id", ""),
                "failure_family": row.get("failure_family", ""),
                "issue_id": row.get("issue_id", ""),
                "loss": accum_loss / grad_accum,
                "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
                "learning_rate": lr,
                **stats,
            }
            accum_loss = 0.0
            if global_step % logging_steps == 0:
                print(json.dumps(record, ensure_ascii=False))
            logs.append(record)

            if global_step % eval_steps == 0 or global_step == max_steps:
                ev_train = evaluate(train_rows, "train", global_step)
                ev_dev = evaluate(dev_rows, "dev", global_step)
                print(json.dumps(ev_train, ensure_ascii=False))
                print(json.dumps(ev_dev, ensure_ascii=False))
                logs.extend([ev_train, ev_dev])

            if global_step % save_steps == 0 or global_step == max_steps:
                save_adapter(args.output_dir / f'checkpoint-{global_step}')

    save_adapter(args.output_dir)

    summary = {
        "base_model": base_model,
        "sft_adapter_reference_and_init": str(args.sft_adapter),
        "output_dir": str(args.output_dir),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "global_steps": global_step,
        "micro_steps": micro_step,
        "runtime_seconds": time.time() - start,
        "config": cfg,
    }
    write_json(summary, args.output_dir / "dpo_after_sft_two_adapter_training_summary.json")
    write_json(logs, args.output_dir / "dpo_after_sft_log_history.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
