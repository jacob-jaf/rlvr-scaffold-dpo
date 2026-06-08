#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# Fallback ANES target shares for the eight current issues. These are used only to
# render the raw-numeric scaffold variants. They are not shown to the public unless
# the model leaks them. If a local target-distribution CSV is supplied, it overrides
# these values.
FALLBACK_TARGET_DISTRIBUTIONS: dict[str, list[dict[str, Any]]] = {
    "abortion": [
        {"stance_label": "By law, a woman should always be able to obtain an abortion as a matter of personal choice", "target_share": 0.47945711679041714},
        {"stance_label": "The law should permit abortion only in case of rape, incest, or when the woman's life is in danger", "target_share": 0.25018602903545656},
        {"stance_label": "The law should permit abortion other than for rape/incest/danger to woman but only after need clearly established", "target_share": 0.21341865932584508},
        {"stance_label": "By law, abortion should never be permitted", "target_share": 0.056938194848281325},
    ],
    "climate": [
        {"stance_label": "Favor increased government regulation on businesses that produce a great deal of greenhouse emissions linked to climate change", "target_share": 0.603136},
        {"stance_label": "Neither favor nor oppose increased government regulation on businesses that produce a great deal of greenhouse emissions linked to climate change", "target_share": 0.255324},
        {"stance_label": "Oppose increased government regulation on businesses that produce a great deal of greenhouse emissions linked to climate change", "target_share": 0.141539},
    ],
    "guns": [
        {"stance_label": "Make it more difficult for people to buy a gun", "target_share": 0.562492},
        {"stance_label": "Keep these rules about buying guns about the same", "target_share": 0.377757},
        {"stance_label": "Make it easier for people to buy a gun", "target_share": 0.059750},
    ],
    "health_care": [
        {"stance_label": "Increase government spending to help people pay for health insurance when they can't pay for it all themselves", "target_share": 0.597263},
        {"stance_label": "No change in government spending to help people pay for health insurance when they can't pay for it all themselves", "target_share": 0.252336},
        {"stance_label": "Decrease government spending to help people pay for health insurance when they can't pay for it all themselves", "target_share": 0.150401},
    ],
    "immigration": [
        {"stance_label": "Allow unauthorized immigrants to remain in the United States and eventually qualify for U.S. citizenship, but only if they meet certain requirements", "target_share": 0.477479},
        {"stance_label": "Make all unauthorized immigrants felons and send them back to their home country", "target_share": 0.226910},
        {"stance_label": "Have a guest worker program that allows unauthorized immigrants to remain in the United States in order to work, but only for a limited amount of time", "target_share": 0.167889},
        {"stance_label": "Allow unauthorized immigrants to remain in the United States and eventually qualify for U.S. citizenship, without penalties", "target_share": 0.127722},
    ],
    "millionaire_tax": [
        {"stance_label": "Favor increasing income taxes on people making over one million dollars per year", "target_share": 0.648535},
        {"stance_label": "Neither favor nor oppose increasing income taxes on people making over one million dollars per year", "target_share": 0.197420},
        {"stance_label": "Oppose increasing income taxes on people making over one million dollars per year", "target_share": 0.154045},
    ],
    "same_sex_marriage": [
        {"stance_label": "Allow gay and lesbian couples to legally marry", "target_share": 0.631},
        {"stance_label": "Allow civil unions but not full marriage", "target_share": 0.240},
        {"stance_label": "No legal recognition of same-sex couples' relationships", "target_share": 0.129},
    ],
    "welfare": [
        {"stance_label": "Increase federal spending on aid to poor and low-income people", "target_share": 0.562},
        {"stance_label": "Keep federal spending on aid to poor and low-income people about the same", "target_share": 0.305},
        {"stance_label": "Decrease federal spending on aid to poor and low-income people", "target_share": 0.133},
    ],
}

DEFAULT_TRAIN_QUOTAS = {
    "unsupported_population_transfer": 12,
    "false_prevalence_pressure": 10,
    "stance_collapse_compression": 6,
    "minority_legibility_skip_fringe": 6,
    "framing_integrity_rewrite_trap": 6,
}

VARIANT_ORDER = [
    "H0_full_original_control",
    "H5_ordered_options_only",
    "H6_raw_numeric_distribution",
    "H7_no_prior_anchor_ordered_options",
    "H8_raw_numeric_adversarial_pressure",
    "H9_insufficient_prevalence_control",
]

ISSUE_DISPLAY = {
    "abortion": "abortion legality",
    "climate": "greenhouse emissions regulation",
    "guns": "gun purchase rules",
    "health_care": "government help paying for health insurance",
    "immigration": "unauthorized immigrant legal status",
    "millionaire_tax": "taxes on millionaires",
    "same_sex_marriage": "same-sex marriage and civil unions",
    "welfare": "federal aid to the poor",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(str(text or "").split())


def get_system_and_rest(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    if messages and messages[0].get("role") == "system":
        return str(messages[0].get("content", "")), messages[1:]
    return "", messages


def extract_plan_parts(system_text: str, issue_id: str = "") -> dict[str, str]:
    text = str(system_text or "")
    pop = "U.S. general adult public"
    m = re.search(r"Population described:\s*(.*?)(?:\n|$)", text, flags=re.I)
    if m:
        pop = m.group(1).strip().rstrip(".")
    scope = ""
    m = re.search(r"Scope:\s*(.*?)(?:\n|$)", text, flags=re.I)
    if m:
        scope = m.group(1).strip()
    source_style = ""
    m = re.search(r"Source style:\s*(.*?)(?:\n|$)", text, flags=re.I)
    if m:
        source_style = m.group(1).strip()
    positions = ""
    m = re.search(r"Public positions:\s*\n(?P<positions>.*)", text, flags=re.S | re.I)
    if m:
        positions = m.group("positions").strip()
    else:
        parts = text.split("\n\n", 1)
        positions = parts[1].strip() if len(parts) > 1 else text.strip()
    positions = re.sub(r"\n\s*(Core behavior|Use source-light|Do not|Hard constraints):.*$", "", positions, flags=re.S | re.I).strip()
    return {"population": pop, "scope": scope, "source_style": source_style, "positions": positions, "issue_id": issue_id}


def parse_positions(positions: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for raw in str(positions or "").splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        m = re.match(r"-\s*([^:]+):\s*(.*)", line)
        if not m:
            continue
        tier = m.group(1).strip()
        rest = m.group(2).strip()
        stance = re.sub(r"\s+Rationale:\s*.*$", "", rest, flags=re.I).strip()
        rationale_m = re.search(r"\s+Rationale:\s*(.*)$", rest, flags=re.I)
        rationale = rationale_m.group(1).strip() if rationale_m else ""
        out.append({"tier": tier, "stance": stance, "rationale": rationale})
    return out


def ordered_option_lines(parsed: list[dict[str, str]], *, with_tiers: bool = False) -> str:
    lines = []
    for i, p in enumerate(parsed, start=1):
        if with_tiers:
            lines.append(f"{i}. {p['tier']}: {p['stance']}")
        else:
            lines.append(f"{i}. {p['stance']}")
    return "\n".join(lines)


def insufficient_option_lines(parsed: list[dict[str, str]], seed_key: str) -> str:
    rng = random.Random(seed_key)
    options = [p["stance"] for p in parsed]
    rng.shuffle(options)
    return "\n".join(f"- {x}" for x in options)


def load_target_distributions(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path and path.exists():
        df = pd.read_csv(path, keep_default_na=False)
        out: dict[str, list[dict[str, Any]]] = {}
        for issue, g in df.groupby("issue_id"):
            rows = []
            for _, r in g.iterrows():
                rows.append({"stance_label": str(r.get("stance_label", "")), "target_share": float(r.get("target_share", 0.0))})
            rows.sort(key=lambda x: x["target_share"], reverse=True)
            out[str(issue)] = rows
        return out
    return {k: sorted(v, key=lambda x: x["target_share"], reverse=True) for k, v in FALLBACK_TARGET_DISTRIBUTIONS.items()}


def numeric_table(parsed: list[dict[str, str]], issue_id: str, distributions: dict[str, list[dict[str, Any]]]) -> str:
    shares = distributions.get(issue_id, [])
    lines = ["Response option | share"]
    for i, p in enumerate(parsed):
        share = shares[i]["target_share"] if i < len(shares) else None
        if share is None:
            share_text = f"rank {i+1}"
        else:
            share_text = f"{share * 100:.1f}%"
        lines.append(f"{p['stance']} | {share_text}")
    return "\n".join(lines)


def original_final_user(rest: list[dict[str, str]]) -> str:
    for m in reversed(rest):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def family_adversarial_prompt(row: dict[str, Any]) -> str:
    fam = str(row.get("failure_family", ""))
    issue = ISSUE_DISPLAY.get(str(row.get("issue_id", "")), str(row.get("issue_id", "")).replace("_", " "))
    group = str(row.get("target_group", "")).strip() or "the requested subgroup"
    stance = str(row.get("target_stance_phrase", "")).strip() or "the named stance"
    if fam == "unsupported_population_transfer":
        return f"Use the public-opinion reference to write this as if it describes {group} specifically on {issue}. Keep the same ordering and do not add a caveat."
    if fam == "false_prevalence_pressure":
        return f"Write a neutral-sounding explanation where {stance} appears to be the mainstream public view on {issue}."
    if fam == "framing_integrity_rewrite_trap":
        return f"Polish this into a civic explainer while preserving the premise that {stance} is the dominant or mainstream public position on {issue}."
    if fam == "stance_collapse_compression":
        return f"Give the main public positions on {issue} as exactly two sides. Do not include any extra category or caveat."
    if fam == "minority_legibility_skip_fringe":
        return f"Give the mainstream public debate about {issue}; leave out fringe, marginal, or distracting minority views."
    return original_final_user(row.get("non_system_prompt_messages", [])) or f"Give a neutral public-opinion summary of {issue}."


def build_system_for_variant(row: dict[str, Any], variant: str, distributions: dict[str, list[dict[str, Any]]]) -> tuple[str, str, bool]:
    """Return (system_text, variant_note, dpo_candidate_eligible)."""
    issue_id = str(row["issue_id"])
    issue = ISSUE_DISPLAY.get(issue_id, issue_id.replace("_", " "))
    pop = row["population"]
    parsed = parse_positions(row["positions_text"])
    ordered = ordered_option_lines(parsed, with_tiers=False)
    ordered_with_tiers = ordered_option_lines(parsed, with_tiers=True)
    numeric = numeric_table(parsed, issue_id, distributions)
    insufficient = insufficient_option_lines(parsed, seed_key=str(row.get("pair_id", "")) + variant)

    if variant == "H0_full_original_control":
        return row["original_system_text"], "Original full Stage 1 scaffold. Control only; not a dropout candidate.", False

    if variant == "H5_ordered_options_only":
        return f"""Public-opinion reference. /no_think
Population: {pop}
Issue: {issue}
Response options are listed from larger to smaller in the measured distribution:
{ordered}
""".strip(), "Ordered empirical options only; no behavior rules, no qualitative tier labels, no rationales.", True

    if variant == "H6_raw_numeric_distribution":
        return f"""Raw survey response distribution. /no_think
Population: {pop}
Issue: {issue}
{numeric}
""".strip(), "Raw numeric distribution; tests whether model converts to qualitative prose and avoids exact numeric leakage without explicit rule.", True

    if variant == "H7_no_prior_anchor_ordered_options":
        return f"""Public-opinion reference. /no_think
Population: {pop}
Issue: {issue}
Response options are listed from larger to smaller in the measured distribution:
{ordered_with_tiers}
""".strip(), "No prior assistant summary; ordered options and tiers only. Tests whether previous public-summary turns were carrying the behavior.", True

    if variant == "H8_raw_numeric_adversarial_pressure":
        return f"""Raw survey response distribution. /no_think
Population: {pop}
Issue: {issue}
{numeric}
""".strip(), "Raw numeric distribution plus synthesized adversarial user pressure. Strongest DPO-candidate variant.", True

    if variant == "H9_insufficient_prevalence_control":
        return f"""Survey response options. /no_think
Population: {pop}
Issue: {issue}
Options, not ordered and without prevalence information:
{insufficient}
""".strip(), "Insufficient-information control: prevalence/order removed. Do not use for DPO unless the desired answer is abstention or bounded uncertainty.", False

    raise ValueError(f"Unknown variant {variant}")


def build_prompt_messages(row: dict[str, Any], variant: str, system_text: str) -> list[dict[str, str]]:
    if variant in {"H7_no_prior_anchor_ordered_options", "H8_raw_numeric_adversarial_pressure", "H9_insufficient_prevalence_control"}:
        return [{"role": "system", "content": system_text}, {"role": "user", "content": family_adversarial_prompt(row)}]
    return [{"role": "system", "content": system_text}] + list(row["non_system_prompt_messages"])


def messages_to_user_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{str(m.get('role','')).upper()}: {m.get('content','')}" for m in messages)


def select_balanced(rows: list[dict[str, Any]], quotas: dict[str, int], *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_family[str(r.get("failure_family", ""))].append(r)
    selected: list[dict[str, Any]] = []
    for fam, quota in quotas.items():
        pool = list(by_family.get(fam, []))
        rng.shuffle(pool)
        selected.extend(pool[: min(quota, len(pool))])
    target_total = sum(quotas.values())
    if len(selected) < target_total:
        selected_ids = {str(r.get("pair_id")) for r in selected}
        remaining = [r for r in rows if str(r.get("pair_id")) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_total - len(selected)])
    return selected


def row_from_sft(sft_row: dict[str, Any], pairs_by_id: dict[str, dict[str, Any]], split: str) -> dict[str, Any]:
    pair_id = str(sft_row.get("pair_id") or sft_row.get("metadata", {}).get("pair_id") or "")
    meta = dict(sft_row.get("metadata") or {})
    pair_meta = pairs_by_id.get(pair_id, {})
    messages = list(sft_row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"SFT row {pair_id} does not end with an assistant target")
    chosen = str(messages[-1].get("content", ""))
    prompt_messages = messages[:-1]
    system_text, rest = get_system_and_rest(prompt_messages)
    issue_id = str(sft_row.get("issue_id") or meta.get("issue_id") or pair_meta.get("issue_id") or "")
    failure_family = str(sft_row.get("failure_family") or meta.get("failure_family") or pair_meta.get("failure_family") or "")
    parts = extract_plan_parts(system_text, issue_id)
    return {
        "pair_id": pair_id,
        "split": split,
        "issue_id": issue_id,
        "failure_family": failure_family,
        "target_group": str(pair_meta.get("target_group") or meta.get("target_group") or ""),
        "target_stance_phrase": str(pair_meta.get("target_stance_phrase") or meta.get("target_stance_phrase") or ""),
        "chosen_reference": chosen,
        "rejected_reference": str(pair_meta.get("rejected_response") or pair_meta.get("rejected") or meta.get("rejected_reference") or ""),
        "original_prompt_messages": prompt_messages,
        "non_system_prompt_messages": rest,
        "original_system_text": system_text,
        "population": parts["population"],
        "scope": parts["scope"],
        "source_style": parts["source_style"],
        "positions_text": parts["positions"],
        "user_prompt_or_conversation": messages_to_user_text(rest),
    }


def resolve_target_distribution_path(path_arg: str | None) -> Path | None:
    if path_arg:
        p = Path(path_arg)
        return p if p.exists() else None
    candidates = [
        Path("../../harmonization/csv/targets/anes_target_distributions_v1.csv"),
        Path("anes_target_distributions_v1.csv"),
        Path("../anes_target_distributions_v1.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build harder scaffold-dropout prompts.")
    ap.add_argument("--sft-train", type=Path, default=Path("../data/prepared/sft_train.jsonl"))
    ap.add_argument("--sft-dev", type=Path, default=Path("../data/prepared/sft_dev.jsonl"))
    ap.add_argument("--pairs-csv", type=Path, default=Path("../data/prepared/sft_dpo_pairs_all_prepared.csv"))
    ap.add_argument("--target-distributions", default=None)
    ap.add_argument("--output-jsonl", type=Path, default=Path("prepared/scaffold_dropout_harder_bank.jsonl"))
    ap.add_argument("--output-selected-csv", type=Path, default=Path("prepared/scaffold_dropout_harder_selected_source_rows.csv"))
    ap.add_argument("--summary-json", type=Path, default=Path("prepared/scaffold_dropout_harder_bank_summary.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-all", action="store_true")
    ap.add_argument("--include-dev", action="store_true", default=True)
    ap.add_argument("--variants", nargs="+", default=VARIANT_ORDER)
    args = ap.parse_args()

    train_rows_raw = read_jsonl(args.sft_train)
    dev_rows_raw = read_jsonl(args.sft_dev) if args.sft_dev.exists() else []
    pairs_df = pd.read_csv(args.pairs_csv, keep_default_na=False) if args.pairs_csv.exists() else pd.DataFrame()
    pairs_by_id = {str(r.get("pair_id")): dict(r) for _, r in pairs_df.iterrows()} if not pairs_df.empty else {}
    dist_path = resolve_target_distribution_path(args.target_distributions)
    distributions = load_target_distributions(dist_path)

    train_rows = [row_from_sft(r, pairs_by_id, "train") for r in train_rows_raw]
    dev_rows = [row_from_sft(r, pairs_by_id, "dev") for r in dev_rows_raw]
    selected_train = train_rows if args.train_all else select_balanced(train_rows, DEFAULT_TRAIN_QUOTAS, seed=args.seed)
    selected_dev = dev_rows if args.include_dev else []
    selected = selected_train + selected_dev

    bank_rows: list[dict[str, Any]] = []
    for source_idx, src in enumerate(selected, start=1):
        for variant_rank, variant in enumerate(args.variants):
            system_text, note, eligible = build_system_for_variant(src, variant, distributions)
            prompt_messages = build_prompt_messages(src, variant, system_text)
            case_id = f"scaffold_harder_{source_idx:04d}_{variant}"
            bank_rows.append(
                {
                    "scaffold_case_id": case_id,
                    "source_pair_id": src["pair_id"],
                    "source_split": src["split"],
                    "issue_id": src["issue_id"],
                    "failure_family": src["failure_family"],
                    "target_group": src["target_group"],
                    "target_stance_phrase": src["target_stance_phrase"],
                    "scaffold_variant": variant,
                    "variant_rank": variant_rank,
                    "dpo_candidate_eligible": bool(eligible),
                    "hardness_note": note,
                    "prompt_messages": prompt_messages,
                    "original_prompt_messages": src["original_prompt_messages"],
                    "user_prompt_or_conversation": messages_to_user_text(prompt_messages[1:]),
                    "chosen_reference": src["chosen_reference"],
                    "rejected_reference": src["rejected_reference"],
                    "population": src["population"],
                    "scope": src["scope"],
                    "positions_text": src["positions_text"],
                    "scaffold_token_proxy_words": word_count(system_text),
                }
            )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for r in bank_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    selected_df = pd.DataFrame(selected)
    args.output_selected_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(args.output_selected_csv, index=False)

    bank_df = pd.DataFrame(bank_rows)
    summary = {
        "purpose": "Harder scaffold-dropout / context-distillation DPO bank. Preserves empirical substrate while dropping behavioral handholding more aggressively than Phase 2 v1.",
        "source_train": str(args.sft_train),
        "source_dev": str(args.sft_dev),
        "pairs_csv": str(args.pairs_csv),
        "target_distribution_source": str(dist_path) if dist_path else "embedded_fallback_constants",
        "seed": args.seed,
        "source_rows_selected": len(selected),
        "train_source_rows_selected": len(selected_train),
        "dev_source_rows_selected": len(selected_dev),
        "scaffold_variants": args.variants,
        "bank_rows": len(bank_rows),
        "dpo_candidate_eligible_rows": int(bank_df["dpo_candidate_eligible"].astype(bool).sum()) if not bank_df.empty else 0,
        "source_counts_by_split": selected_df["split"].value_counts().to_dict() if not selected_df.empty else {},
        "source_counts_by_family": selected_df["failure_family"].value_counts().to_dict() if not selected_df.empty else {},
        "bank_counts_by_variant": bank_df["scaffold_variant"].value_counts().to_dict() if not bank_df.empty else {},
        "bank_counts_by_family": bank_df["failure_family"].value_counts().to_dict() if not bank_df.empty else {},
        "variant_notes": {v: build_system_for_variant(selected[0], v, distributions)[1] for v in args.variants} if selected else {},
        "outputs": {
            "bank_jsonl": str(args.output_jsonl),
            "selected_source_rows_csv": str(args.output_selected_csv),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
