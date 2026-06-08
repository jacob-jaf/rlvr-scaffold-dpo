#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--dev', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--sft-adapter', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--base-model', default=None)
    ap.add_argument('--load-in-4bit', action='store_true')
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None
    try:
        from trl import DPOTrainer, DPOConfig
    except Exception as e:
        raise SystemExit("Install compatible trl/transformers/peft before running DPO training.") from e

    from sft_dpo_utils import apply_chat_template_no_thinking

    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    base_model = args.base_model or cfg.get('base_model','Qwen/Qwen3-8B')
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    quant = None
    if args.load_in_4bit or cfg.get('load_in_4bit', False):
        if BitsAndBytesConfig is None:
            raise SystemExit('BitsAndBytesConfig unavailable; install bitsandbytes or omit 4bit.')
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            quantization_config=quant,
        )

    adapter_path = Path(args.sft_adapter).resolve()
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"Missing SFT adapter at {adapter_path}")

    base = load_base()
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    model.config.use_cache = False
    # One 4-bit base load only. TRL uses adapter-disabled forwards for the reference policy.
    ref_model = None

    def to_dpo_row(row):
        prompt_text = apply_chat_template_no_thinking(
            tok, row['prompt'], tokenize=False, add_generation_prompt=True
        )
        return {'prompt': prompt_text, 'chosen': row['chosen'][0]['content'], 'rejected': row['rejected'][0]['content'], 'pair_id': row.get('pair_id',''), 'failure_family': row.get('failure_family',''), 'issue_id': row.get('issue_id','')}

    train_ds = Dataset.from_list([to_dpo_row(r) for r in load_jsonl(args.train)])
    dev_ds = Dataset.from_list([to_dpo_row(r) for r in load_jsonl(args.dev)])
    dpo_args = DPOConfig(
        output_dir=str(args.output_dir), beta=float(cfg.get('beta',0.05)), loss_type=cfg.get('loss_type','sigmoid'), learning_rate=float(cfg.get('learning_rate',5e-6)), num_train_epochs=float(cfg.get('num_train_epochs',1.0)),
        per_device_train_batch_size=int(cfg.get('per_device_train_batch_size',1)), per_device_eval_batch_size=int(cfg.get('per_device_eval_batch_size',1)), gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps',8)),
        max_prompt_length=int(cfg.get('max_prompt_length',4096)), max_length=int(cfg.get('max_length',6144)), eval_strategy=cfg.get('eval_strategy','steps'), eval_steps=int(cfg.get('eval_steps',5)), save_steps=int(cfg.get('save_steps',5)), logging_steps=int(cfg.get('logging_steps',1)),
        warmup_ratio=float(cfg.get('warmup_ratio',0.03)), lr_scheduler_type=cfg.get('lr_scheduler_type','cosine'), bf16=bool(cfg.get('bf16',True)) and torch.cuda.is_available(), gradient_checkpointing=bool(cfg.get('gradient_checkpointing',True)), remove_unused_columns=False, report_to=cfg.get('report_to','none'), seed=int(cfg.get('seed',42))
    )
    try:
        trainer = DPOTrainer(model=model, ref_model=ref_model, args=dpo_args, train_dataset=train_ds, eval_dataset=dev_ds, processing_class=tok)
    except TypeError:
        trainer = DPOTrainer(model=model, ref_model=ref_model, args=dpo_args, train_dataset=train_ds, eval_dataset=dev_ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    summary = {'base_model': base_model, 'sft_adapter': str(args.sft_adapter), 'train_rows': len(train_ds), 'eval_rows': len(dev_ds), 'output_dir': str(args.output_dir), 'chat_template': 'enable_thinking_false', 'config': cfg}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'dpo_after_sft_training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
