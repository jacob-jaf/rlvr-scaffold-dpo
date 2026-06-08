#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from sft_dpo_utils import apply_chat_template_no_thinking

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

class AssistantOnlySFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_length: int):
        self.rows=rows
        self.tokenizer=tokenizer
        self.max_seq_length=max_seq_length
        self.items=[self.encode(r) for r in rows]

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

    def encode(self, row: dict[str, Any]) -> dict[str, Any]:
        messages=list(row['messages'])
        if not messages or messages[-1].get('role')!='assistant':
            raise ValueError(f"Row {row.get('pair_id')} must end with final assistant message.")
        context=messages[:-1]
        target=str(messages[-1].get('content','')).strip()
        if not target:
            raise ValueError(f"Row {row.get('pair_id')} has empty final assistant target.")
        # Match generation: Qwen3 no-thinking chat template.
        prompt_text = apply_chat_template_no_thinking(
            self.tokenizer, context, tokenize=False, add_generation_prompt=True
        )
        full_text=prompt_text + target + (self.tokenizer.eos_token or '')
        prompt_ids=self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        full_ids=self.tokenizer(full_text, add_special_tokens=False)['input_ids']
        if len(full_ids) > self.max_seq_length:
            # Prefer not to silently break. Capacity-test rows should fit.
            raise ValueError(f"Row {row.get('pair_id')} has {len(full_ids)} tokens > max_seq_length={self.max_seq_length}.")
        labels=[-100]*min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        if len(labels)!=len(full_ids):
            raise AssertionError('labels/input length mismatch')
        if all(x == -100 for x in labels):
            raise ValueError(f"Row {row.get('pair_id')} has no supervised assistant tokens.")
        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(full_ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pair_id': row.get('pair_id',''),
            'failure_family': row.get('failure_family',''),
            'issue_id': row.get('issue_id',''),
            'supervised_tokens': sum(1 for x in labels if x!=-100),
            'total_tokens': len(full_ids),
        }

class DataCollator:
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids=pad_sequence([f['input_ids'] for f in features], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask=pad_sequence([f['attention_mask'] for f in features], batch_first=True, padding_value=0)
        labels=pad_sequence([f['labels'] for f in features], batch_first=True, padding_value=-100)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main() -> None:
    ap=argparse.ArgumentParser(description='Assistant-final-response-only SFT LoRA/QLoRA trainer.')
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--dev', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--base-model', default=None)
    ap.add_argument('--load-in-4bit', action='store_true')
    ap.add_argument('--summary-json', default='sft_training_summary.json')
    args=ap.parse_args()

    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    base_model=args.base_model or cfg.get('base_model','Qwen/Qwen3-8B')
    tokenizer=AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'

    quant=None
    use_4bit=args.load_in_4bit or bool(cfg.get('load_in_4bit', False))
    if use_4bit:
        if BitsAndBytesConfig is None:
            raise SystemExit('BitsAndBytesConfig unavailable; install bitsandbytes or omit --load-in-4bit')
        quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model=AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        quantization_config=quant,
    )
    model.config.use_cache=False
    if bool(cfg.get('gradient_checkpointing', True)):
        model.gradient_checkpointing_enable()
    if use_4bit:
        model=prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(cfg.get('gradient_checkpointing', True)))

    peft_config=LoraConfig(
        r=int(cfg.get('lora_r',64)),
        lora_alpha=int(cfg.get('lora_alpha',128)),
        lora_dropout=float(cfg.get('lora_dropout',0.0)),
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=cfg.get('target_modules',["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
    )
    model=get_peft_model(model, peft_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    max_seq=int(cfg.get('max_seq_length',6144))
    train_rows=load_jsonl(args.train)
    dev_rows=load_jsonl(args.dev)
    train_ds=AssistantOnlySFTDataset(train_rows, tokenizer, max_seq)
    dev_ds=AssistantOnlySFTDataset(dev_rows, tokenizer, max_seq)
    supervised=sum(x['supervised_tokens'] for x in train_ds.items)
    total=sum(x['total_tokens'] for x in train_ds.items)

    train_args=TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=float(cfg.get('learning_rate',5e-5)),
        num_train_epochs=float(cfg.get('num_train_epochs',30.0)),
        max_steps=int(cfg.get('max_steps',-1)),
        per_device_train_batch_size=int(cfg.get('per_device_train_batch_size',1)),
        per_device_eval_batch_size=int(cfg.get('per_device_eval_batch_size',1)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps',1)),
        eval_strategy=cfg.get('eval_strategy','steps'),
        eval_steps=int(cfg.get('eval_steps',25)),
        save_steps=int(cfg.get('save_steps',25)),
        logging_steps=int(cfg.get('logging_steps',1)),
        warmup_ratio=float(cfg.get('warmup_ratio',0.0)),
        lr_scheduler_type=cfg.get('lr_scheduler_type','constant'),
        bf16=bool(cfg.get('bf16',True)) and torch.cuda.is_available(),
        fp16=bool(cfg.get('fp16',False)) and torch.cuda.is_available(),
        gradient_checkpointing=bool(cfg.get('gradient_checkpointing',True)),
        report_to=cfg.get('report_to','none'),
        seed=int(cfg.get('seed',42)),
        remove_unused_columns=False,
        max_grad_norm=float(cfg.get('max_grad_norm',1.0)),
    )
    trainer=Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    summary={
        'base_model': base_model,
        'train_rows': len(train_rows),
        'eval_rows': len(dev_rows),
        'output_dir': str(args.output_dir),
        'loss_masking': 'assistant_final_response_only',
        'chat_template': 'enable_thinking_false',
        'train_supervised_tokens': int(supervised),
        'train_total_tokens': int(total),
        'train_supervised_token_share': float(supervised/max(1,total)),
        'config': cfg,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.summary_json).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
