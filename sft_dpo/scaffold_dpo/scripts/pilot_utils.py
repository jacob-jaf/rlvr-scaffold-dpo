from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("eval_case_id") or row.get("scaffold_case_id") or row.get("source_pair_id") or "")


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("prompt", "prompt_messages", "messages"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        if key == "messages" and value and value[-1].get("role") == "assistant":
            value = value[:-1]
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in value]
    raise KeyError("no prompt, prompt_messages, or messages list")


def yes(value: Any) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "approve", "approved"}


def tags(value: Any) -> list[str]:
    return [x.strip() for x in str(value or "").replace(",", ";").split(";") if x.strip()]


def add_stage_scripts_to_path(file: str) -> None:
    scripts = Path(file).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
