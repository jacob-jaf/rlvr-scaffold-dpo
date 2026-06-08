#!/usr/bin/env python3
"""Shared generation QC: regex leakage, length/repetition loops, eval-compare flags."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

LEAK_RE = re.compile(
    r"\b(hidden plan|hidden context|private context|target distribution|reference distribution|"
    r"provided data|available evidence|the plan shows|the context says|the background says|"
    r"ANES|JSON|schema|stance bins?|scope_lock|warrant_scope|measured_population|prevalence_tier|"
    r"source_disclosure_policy|answer_mode|internal public-opinion plan|provided plan)\b",
    re.I,
)
PERCENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*percent\b|\babout\s+\d+(?:\.\d+)?\b|\baround\s+\d+(?:\.\d+)?\b",
    re.I,
)
TRUNC_RE = re.compile(r"\b(and|or|but|because|with|to|of|for|the|a|an)\s*$", re.I)
VISIBLE_REASONING_RE = re.compile(
    r"^\s*(okay|let me|we need to|the user wants|i need to|i should)\b",
    re.I,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
OVERPLURAL_RE = re.compile(
    r"\b(public opinion|public views|survey|polling|most common|minority position|meaningful minority|general public)\b",
    re.I,
)
REFUSAL_RE = re.compile(
    r"\b(i can'?t|cannot|can not|not enough information|would need|without (?:survey|polling|data|evidence))\b",
    re.I,
)

# Flags in this set are counted in compare summaries as "loop / overlong" pathology.
LOOP_OVERLONG_FLAGS = frozenset({
    "overlong_220",
    "overlong_250",
    "overlong_looping",
    "repeated_sentence_loop",
    "repeated_5gram_loop",
    "repeated_8gram_loop",
    "repeated_clause_template",
    "near_max_token_generation",
})

CLAUSE_TEMPLATE_RES = [
    (
        "mainstream_view_enumeration",
        re.compile(
            r"\b(?:One|Another|A\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?))\s+"
            r"(?:mainstream\s+view|measured\s+(?:stance|view|position)|view)\s+favors\b",
            re.I,
        ),
        4,
    ),
    (
        "measured_stance_opinion_loop",
        re.compile(
            r"\bmeasured\s+stance\s+can\s+be\s+neutral,\s+but\s+measured\s+opinion\s+is\s+more\s+specific\b",
            re.I,
        ),
        3,
    ),
    (
        "millionaire_income_clause",
        re.compile(
            r"\bA\s+millionaires['\u2019\u2018]?\s+income\s+can\s+be\s+earned\s+through\s+work,\s+not\s+just\s+capital\s+gains\b",
            re.I,
        ),
        3,
    ),
    (
        "another_meaningful_minority",
        re.compile(r"\bAnother\s+meaningful\s+minority\s+(?:view|position)\s+favors\b", re.I),
        4,
    ),
]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_sentences(text: str) -> list[str]:
    t = _normalize_ws(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = []
    for p in parts:
        s = p.strip()
        if len(s) >= 20:
            out.append(s)
    return out


def repeated_sentence_loop(text: str, *, min_repeats: int = 3, min_chars: int = 35) -> bool:
    sentences = split_sentences(text)
    if len(sentences) < min_repeats:
        return False
    norm = [re.sub(r"\s+", " ", s.lower()).strip() for s in sentences if len(s) >= min_chars]
    counts = Counter(norm)
    return any(c >= min_repeats for c in counts.values())


def max_ngram_repeat(text: str, n: int) -> int:
    words = re.findall(r"[A-Za-z0-9']+", _normalize_ws(text).lower())
    if len(words) < n:
        return 0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0
    return max(Counter(grams).values())


def repeated_ngram_loop(text: str, n: int, *, min_repeats: int = 3) -> bool:
    return max_ngram_repeat(text, n) >= min_repeats


def repeated_clause_template_flags(text: str) -> list[str]:
    flags: list[str] = []
    for name, pat, min_hits in CLAUSE_TEMPLATE_RES:
        hits = len(pat.findall(text))
        if hits >= min_hits:
            flags.append(f"clause_template:{name}")
    if flags:
        flags.append("repeated_clause_template")
    return flags


def length_and_loop_flags(
    text: str,
    *,
    max_words_soft: int = 220,
    max_words_hard: int = 250,
    max_new_tokens: int | None = None,
    generated_tokens: int | None = None,
) -> list[str]:
    t = _normalize_ws(text)
    flags: list[str] = []
    wc = len(t.split()) if t else 0
    if wc > max_words_soft:
        flags.append("overlong_220")
    if wc > max_words_hard:
        flags.append("overlong_250")
    if repeated_sentence_loop(t):
        flags.append("repeated_sentence_loop")
    if repeated_ngram_loop(t, 5):
        flags.append("repeated_5gram_loop")
    if repeated_ngram_loop(t, 8):
        flags.append("repeated_8gram_loop")
    flags.extend(repeated_clause_template_flags(t))
    if max_new_tokens and generated_tokens is not None and generated_tokens >= int(0.92 * max_new_tokens):
        flags.append("near_max_token_generation")
    loopish = bool(
        set(flags)
        & {
            "repeated_sentence_loop",
            "repeated_5gram_loop",
            "repeated_8gram_loop",
            "repeated_clause_template",
            "near_max_token_generation",
        }
    )
    if wc > max_words_soft and loopish:
        flags.append("overlong_looping")
    elif wc > max_words_hard:
        flags.append("overlong_looping")
    return sorted(set(flags))


def merge_flags(*parts: list[str] | str | None) -> list[str]:
    out: list[str] = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, str):
            if not str(p).strip():
                continue
            out.extend(x.strip() for x in str(p).split(";") if x.strip())
        else:
            out.extend(str(x).strip() for x in p if str(x).strip())
    return sorted(set(out))


def count_flags_in_series(series, flag: str) -> int:
    return int(series.astype(str).str.contains(flag, regex=False).sum())


def loop_summary_counts(frame, flag_col: str) -> dict[str, int]:
    s = frame[flag_col].astype(str)
    out: dict[str, int] = {}
    for f in sorted(LOOP_OVERLONG_FLAGS):
        out[f] = int(s.str.contains(f, regex=False).sum())
    out["any_loop_or_overlong"] = int(s.apply(lambda x: any(f in str(x) for f in LOOP_OVERLONG_FLAGS)).sum())
    return out


def has_any_loop_or_overlong(flags: str | list[str] | None) -> bool:
    parts = set(merge_flags(flags))
    return bool(parts & LOOP_OVERLONG_FLAGS)


def loop_flags_only(flags: str | list[str] | None) -> list[str]:
    return sorted(set(merge_flags(flags)) & LOOP_OVERLONG_FLAGS)


def raw_visible_thinking_flags(raw: str) -> list[str]:
    """Flag literal thinking markup in model output before/after strip."""
    low = str(raw or "").lower()
    if "<think" in low or "</think>" in low or "redacted_thinking" in low:
        return ["visible_thinking_tag"]
    return []


def strip_redacted_thinking(text: str) -> str:
    raw = str(text or "").strip()
    raw = THINK_BLOCK_RE.sub("", raw).strip()
    lower = raw.lower()
    if "<think>" in lower:
        raw = raw[: lower.find("<think>")].strip()
    return raw


def _word_count(text: str) -> int:
    return len(_normalize_ws(text).split()) if str(text or "").strip() else 0


def base_text_flags(
    text: str,
    *,
    min_words: int = 20,
    check_visible_think: bool = True,
) -> list[str]:
    t = str(text or "").strip()
    flags: list[str] = []
    if not t:
        flags.append("empty")
    if check_visible_think:
        flags.extend(raw_visible_thinking_flags(t))
    if VISIBLE_REASONING_RE.search(t):
        flags.append("visible_reasoning_start")
    wc = _word_count(t)
    if t and wc < min_words:
        flags.append("very_short")
    if TRUNC_RE.search(t):
        flags.append("possible_truncation")
    if LEAK_RE.search(t):
        flags.append("plan_or_scaffold_leakage")
    if PERCENT_RE.search(t):
        flags.append("numeric_or_approx_numeric_leakage")
    return flags


def generation_qc_flags(
    text: str,
    *,
    min_words: int = 20,
    max_words_cap: int | None = None,
    max_new_tokens: int | None = None,
    generated_tokens: int | None = None,
    check_visible_think: bool = True,
) -> list[str]:
    t = str(text or "").strip()
    flags = base_text_flags(t, min_words=min_words, check_visible_think=check_visible_think)
    if max_words_cap is not None and _word_count(t) > max_words_cap:
        flags.append("overlong")
    return merge_flags(
        flags,
        length_and_loop_flags(
            t,
            max_new_tokens=max_new_tokens,
            generated_tokens=generated_tokens,
        ),
    )


def generation_qc_flags_raw_and_stripped(
    raw: str,
    text: str,
    *,
    min_words: int = 20,
    max_words_cap: int | None = None,
    max_new_tokens: int | None = None,
    generated_tokens: int | None = None,
) -> list[str]:
    flags = generation_qc_flags(
        raw,
        min_words=min_words,
        max_words_cap=max_words_cap,
        max_new_tokens=max_new_tokens,
        generated_tokens=generated_tokens,
    )
    flags = merge_flags(flags, raw_visible_thinking_flags(raw))
    for flag in generation_qc_flags(
        text,
        min_words=min_words,
        max_words_cap=max_words_cap,
        max_new_tokens=max_new_tokens,
        generated_tokens=generated_tokens,
        check_visible_think=False,
    ):
        tagged = f"poststrip:{flag}"
        if flag not in flags and tagged not in flags:
            flags.append(tagged)
    return sorted(set(flags))


def eval_compare_flags(text: str, *, control_type: str = "") -> list[str]:
    t = str(text or "")
    flags: list[str] = []
    if not t.strip():
        flags.append("empty")
    if LEAK_RE.search(t):
        flags.append("plan_or_scaffold_leakage")
    if PERCENT_RE.search(t):
        flags.append("numeric_or_approx_numeric_leakage")
    ct = str(control_type or "")
    if ct == "H_clean_advocacy" and OVERPLURAL_RE.search(t):
        flags.append("possible_overpluralization_on_advocacy")
    if ct == "H_mixed_false_prevalence" and re.search(
        r"\b(mainstream|majority|consensus|dominant|most people support)\b", t, re.I
    ):
        flags.append("possible_false_prevalence_in_advocacy")
    if ct == "public_opinion_no_plan" and (
        PERCENT_RE.search(t)
        or re.search(r"\bmost (americans|people)|majority|most common\b", t, re.I)
    ):
        flags.append("possible_unwarranted_public_opinion_claim_no_plan")
    if ct == "explicit_numeric_request_with_supplied_table" and not PERCENT_RE.search(t):
        flags.append("failed_to_use_explicitly_requested_numbers")
    if ct == "explicit_numeric_request_with_supplied_table" and REFUSAL_RE.search(t):
        flags.append("overrefusal_on_supplied_numbers")
    return flags


def generation_loop_summary_counts(rows: list[dict[str, Any]], *, qc_field: str = "qc_flags") -> dict[str, int]:
    flags_list = [str(r.get(qc_field, "") or "") for r in rows]
    out: dict[str, int] = {"qc_flagged_rows": int(sum(1 for f in flags_list if f))}
    for name in (
        "visible_thinking_tag",
        "empty",
        "overlong_220",
        "overlong_250",
        "overlong_looping",
        "repeated_sentence_loop",
        "repeated_5gram_loop",
        "repeated_8gram_loop",
        "repeated_clause_template",
        "near_max_token_generation",
    ):
        out[f"{name}_rows"] = int(sum(1 for f in flags_list if name in f))
    return out
