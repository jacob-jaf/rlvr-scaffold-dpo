#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)

# Keep stance-bearing words and negations. Do not remove:
# favor, oppose, neither, same, increase, decrease, not, no, legal, illegal, allow, prohibit, ban.
STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "as",
    "is", "are", "be", "it", "that", "this", "people", "government", "federal",
    "do", "does", "about", "they", "their", "them", "who", "what", "which", "when",
    "where", "why", "how", "from", "than", "now", "has", "have", "had", "can"
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[-–—]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    return [
        w.lower()
        for w in WORD_RE.findall(str(text or ""))
        if w.lower() not in STOP and len(w) > 1
    ]


def feature_counts(text: str) -> Counter:
    toks = word_tokens(text)
    c: Counter[str] = Counter()
    for t in toks:
        c[t] += 1.0
    for i in range(len(toks) - 1):
        c[f"{toks[i]} {toks[i+1]}"] += 1.75
    for i in range(len(toks) - 2):
        c[f"{toks[i]} {toks[i+1]} {toks[i+2]}"] += 2.5
    return c


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def js_divergence(p: list[float], q: list[float]) -> float:
    def kl(x: list[float], y: list[float]) -> float:
        out = 0.0
        for xi, yi in zip(x, y):
            if xi > 0 and yi > 0:
                out += xi * math.log(xi / yi)
        return out
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_phrase_cues(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"issue_cues": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def protect_abbreviations(text: str) -> str:
    replacements = {
        "e.g.": "eg",
        "i.e.": "ie",
        "u.s.": "us",
        "roe v. wade": "roe v wade",
        "v.": "v",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def text_snippets(norm_text: str) -> list[str]:
    """Split text into local contexts so neutral clauses do not suppress directional clauses elsewhere."""
    norm_text = protect_abbreviations(norm_text)
    # Keep this conservative: enough to isolate most list/sentence clauses without destroying phrases.
    parts = re.split(
        r"(?:\n+|[.;:!?]+|\s+[-•]\s+|\bwhile\b|\bwhereas\b|\bothers?\b|\banother view\b|\ba third view\b)",
        norm_text,
        flags=re.IGNORECASE,
    )
    out = [p.strip(" ,") for p in parts if p and p.strip(" ,")]
    return out or [norm_text]


SSM_CIVIL_UNION_ONLY_CONTEXT_RE = re.compile(
    r"""
    (
      \bcivil\s+unions?\b
      (?:(?![.!?\n]).){0,120}
      \b(?:but\s+not|without|instead\s+of|rather\s+than|not)\b
      (?:(?![.!?\n]).){0,80}
      \b(?:legal\s+marriage|marriage|marry|legally\s+marry|permit(?:ting)?\s+marriage)\b
    )
    |
    (
      \b(?:legal\s+marriage|marriage|marry|legally\s+marry|permit(?:ting)?\s+marriage)\b
      (?:(?![.!?\n]).){0,80}
      \b(?:but\s+not|without|instead\s+of|rather\s+than|not)\b
      (?:(?![.!?\n]).){0,120}
      \bcivil\s+unions?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def local_context(text: str, start: int, end: int, window: int = 120) -> str:
    return text[max(0, start - window): min(len(text), end + window)]


def iter_guarded_matches(
    *,
    text: str,
    pattern: str,
    exclude_regexes: list[str],
    issue_id: str,
    stance_bin_id: str,
    context_window: int = 120,
):
    try:
        rx = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    except re.error:
        return

    for m in rx.finditer(text):
        ctx = local_context(text, m.start(), m.end(), window=context_window)
        blocked = False
        for ex in exclude_regexes:
            try:
                if re.search(ex, ctx, flags=re.IGNORECASE | re.DOTALL):
                    blocked = True
                    break
            except re.error:
                continue
        if blocked:
            continue

        if issue_id == "same_sex_marriage" and stance_bin_id == "ssm_legal_marriage":
            if SSM_CIVIL_UNION_ONLY_CONTEXT_RE.search(ctx):
                continue

        yield m, ctx


def is_neutral_stance_bin(stance_bin_id: str) -> bool:
    sid = stance_bin_id.lower()
    return (
        "neither" in sid
        or "neutral" in sid
        or sid.endswith("_same")
        or "_same_" in sid
        or sid.endswith("_no_change")
        or "_no_change_" in sid
    )


def infer_cue_type(cue: dict[str, Any], stance_bin_id: str) -> str:
    cue_type = str(cue.get("cue_type", "")).strip().lower()
    if cue_type in {"neutral", "directional"}:
        return cue_type
    return "neutral" if is_neutral_stance_bin(stance_bin_id) else "directional"


def phrase_scores_for_issue(text: str, issue_id: str, cues: dict[str, Any], phrase_cap: float) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    norm = normalize_text(text)
    snippets = text_snippets(norm)
    issue_cues = cues.get("issue_cues", {}).get(issue_id, {})
    scores: dict[str, float] = {}
    hits: dict[str, list[dict[str, Any]]] = {}

    for stance_bin_id, cue_list in issue_cues.items():
        scores[stance_bin_id] = 0.0
        hits[stance_bin_id] = []

    for snippet in snippets:
        snippet_matches: dict[str, list[dict[str, Any]]] = {}
        for stance_bin_id, cue_list in issue_cues.items():
            local_hits: list[dict[str, Any]] = []
            for cue in cue_list:
                exclude_patterns = cue.get("exclude_regex", cue.get("block_if_regex", [])) or []
                if isinstance(exclude_patterns, str):
                    exclude_patterns = [exclude_patterns]

                pattern = cue.get("include_regex", cue.get("regex"))
                if not pattern and "phrase" in cue:
                    phrase = normalize_text(str(cue["phrase"]))
                    if phrase:
                        pattern = re.escape(phrase)

                if not pattern:
                    continue

                first_match = None
                for m, ctx in iter_guarded_matches(
                    text=snippet,
                    pattern=str(pattern),
                    exclude_regexes=[str(x) for x in exclude_patterns],
                    issue_id=issue_id,
                    stance_bin_id=str(stance_bin_id),
                    context_window=120,
                ):
                    first_match = (m, ctx)
                    break

                if first_match is None:
                    continue
                m, ctx = first_match
                local_hits.append(
                    {
                        "label": cue.get("label", ""),
                        "weight": float(cue.get("weight", 1.0)),
                        "matched": m.group(0)[:200],
                        "context": ctx[:240],
                        "cue_type": infer_cue_type(cue, str(stance_bin_id)),
                        "suppressed": False,
                    }
                )
            snippet_matches[stance_bin_id] = local_hits

        # Local-only precedence: strong neutral hits suppress directional cues in this snippet only.
        neutral_strength = 0.0
        for _, local_hits in snippet_matches.items():
            neutral_strength = max(
                neutral_strength,
                sum(float(h["weight"]) for h in local_hits if str(h.get("cue_type", "")).lower() == "neutral"),
            )

        if neutral_strength >= 1.3:
            for stance_bin_id in list(snippet_matches.keys()):
                for h in snippet_matches[stance_bin_id]:
                    if str(h.get("cue_type", "")).lower() == "directional":
                        h["suppressed"] = True

        for stance_bin_id, local_hits in snippet_matches.items():
            scores[stance_bin_id] += sum(float(h["weight"]) for h in local_hits if not bool(h.get("suppressed", False)))
            hits[stance_bin_id].extend(local_hits)

    for stance_bin_id in list(scores.keys()):
        total = scores[stance_bin_id]
        scores[stance_bin_id] = min(total, phrase_cap) / phrase_cap if phrase_cap > 0 else total

    return scores, hits


def read_responses(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def included_descriptors(desc: pd.DataFrame) -> pd.DataFrame:
    if "include_in_scoring" not in desc.columns:
        raise ValueError("Descriptor file must contain include_in_scoring")
    return desc[desc["include_in_scoring"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()


def score_response_text(
    response_text: str,
    issue_id: str,
    desc: pd.DataFrame,
    targets: pd.DataFrame,
    cues: dict[str, Any],
    *,
    coverage_threshold: float = 0.25,
    descriptor_weight: float = 1.0,
    phrase_weight: float = 2.0,
    phrase_cap: float = 2.0,
) -> dict[str, Any]:
    issue_desc = included_descriptors(desc)
    issue_desc = issue_desc[issue_desc["issue_id"] == issue_id]
    issue_targets = targets[targets["issue_id"] == issue_id].sort_values("stance_order")

    if issue_desc.empty or issue_targets.empty:
        return {
            "score_status": "missing_issue_resources",
            "predicted_distribution_json": "{}",
            "target_distribution_json": "{}",
            "distribution_fidelity_l1": 0.0,
            "coverage_share": 0.0,
            "max_descriptor_similarity": 0.0,
            "scorer_notes": "Missing descriptors or targets for issue."
        }

    response_vec = feature_counts(response_text)

    descriptor_scores: dict[str, float] = {}
    for stance_bin_id in issue_targets["stance_bin_id"]:
        sid = str(stance_bin_id)
        stance_desc = issue_desc[issue_desc["stance_bin_id"].astype(str) == sid]
        best = 0.0
        for _, d in stance_desc.iterrows():
            best = max(best, cosine(response_vec, feature_counts(str(d.get("descriptor_text", "")))))
        descriptor_scores[sid] = best

    phrase_scores, phrase_hits = phrase_scores_for_issue(response_text, issue_id, cues, phrase_cap)

    final_scores: dict[str, float] = {}
    for stance_bin_id in issue_targets["stance_bin_id"]:
        sid = str(stance_bin_id)
        final_scores[sid] = (
            descriptor_weight * descriptor_scores.get(sid, 0.0)
            + phrase_weight * phrase_scores.get(sid, 0.0)
        )

    eps = 1e-6
    total = sum(final_scores.values()) + eps * len(final_scores)
    pred = {k: (v + eps) / total for k, v in final_scores.items()}
    target = {str(r["stance_bin_id"]): float(r["target_share"]) for _, r in issue_targets.iterrows()}

    ordered_bins = [str(r["stance_bin_id"]) for _, r in issue_targets.iterrows()]
    p = [pred[b] for b in ordered_bins]
    q = [target[b] for b in ordered_bins]
    l1 = sum(abs(pi - qi) for pi, qi in zip(p, q))
    fidelity = max(0.0, 1.0 - l1 / 2.0)
    coverage = sum(1 for b in ordered_bins if final_scores.get(b, 0.0) >= coverage_threshold) / len(ordered_bins)
    max_phrase_score = max(phrase_scores.values()) if phrase_scores else 0.0
    if coverage == 0.0 and max_phrase_score == 0.0:
        score_confidence = "low"
    elif coverage >= 0.5 or max_phrase_score >= 0.7:
        score_confidence = "high"
    else:
        score_confidence = "medium"
    js = js_divergence(p, q)
    top_bin = max(final_scores, key=lambda k: final_scores[k]) if final_scores else ""

    return {
        "score_status": "ok",
        "scorer_version": "descriptor_phrase_v2_guarded",
        "top_predicted_bin": top_bin,
        "predicted_distribution_json": json.dumps(pred, sort_keys=True),
        "target_distribution_json": json.dumps(target, sort_keys=True),
        "distribution_fidelity_l1": fidelity,
        "l1_distance": l1,
        "js_divergence": js,
        "coverage_share": coverage,
        "score_confidence": score_confidence,
        "max_descriptor_similarity": max(descriptor_scores.values()) if descriptor_scores else 0.0,
        "max_phrase_score": max_phrase_score,
        "stance_similarity_json": json.dumps(final_scores, sort_keys=True),
        "descriptor_similarity_json": json.dumps(descriptor_scores, sort_keys=True),
        "phrase_score_json": json.dumps({k: phrase_scores.get(k, 0.0) for k in ordered_bins}, sort_keys=True),
        "phrase_hits_json": json.dumps({k: phrase_hits.get(k, []) for k in ordered_bins}, sort_keys=True),
        "scorer_notes": "V2 guarded descriptor + phrase cue scorer with local-context exclude guards. Use for validation/smoke testing before final semantic scorer."
    }


def main() -> None:
    p = argparse.ArgumentParser(description="V2 guarded descriptor + phrase-cue scorer for ANES-aligned outputs.")
    p.add_argument(
        "--responses",
        type=Path,
        default=Path("csv/model_outputs/smoke/model_outputs_main_distributional_smoke_qwen3_8b_v2.csv"),
        help="CSV or JSONL with issue_id and response_text/model_response.",
    )
    p.add_argument("--descriptors", type=Path, default=Path("csv/descriptors/stance_descriptor_bank_v1_expanded.csv"))
    p.add_argument("--targets", type=Path, default=Path("csv/targets/anes_target_distributions_v1.csv"))
    p.add_argument("--phrase-cues", type=Path, default=Path("json/phrase_cues/stance_phrase_cues_v2.json"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("csv/model_outputs/scored/model_outputs_main_distributional_smoke_qwen3_8b_v2_scored_v2_guarded.csv"),
    )
    p.add_argument("--response-column", default="response_text")
    p.add_argument("--issue-column", default="issue_id")
    p.add_argument("--coverage-threshold", type=float, default=0.25)
    p.add_argument("--descriptor-weight", type=float, default=1.0)
    p.add_argument("--phrase-weight", type=float, default=2.0)
    p.add_argument("--phrase-cap", type=float, default=2.0)
    args = p.parse_args()

    responses = read_responses(args.responses)
    if args.response_column not in responses.columns:
        if "model_response" in responses.columns:
            args.response_column = "model_response"
        else:
            raise ValueError(
                f"No response column found. Expected {args.response_column!r} or 'model_response'. "
                "Do not pass a prompt-only file to this scorer."
            )
    if args.issue_column not in responses.columns:
        raise ValueError(f"No issue column found: {args.issue_column!r}")

    desc = pd.read_csv(args.descriptors)
    targets = pd.read_csv(args.targets)
    cues = load_phrase_cues(args.phrase_cues)

    rows = []
    for _, r in responses.iterrows():
        issue_id = str(r.get(args.issue_column, ""))
        response_text = str(r.get(args.response_column, ""))
        score = score_response_text(
            response_text,
            issue_id,
            desc,
            targets,
            cues,
            coverage_threshold=args.coverage_threshold,
            descriptor_weight=args.descriptor_weight,
            phrase_weight=args.phrase_weight,
            phrase_cap=args.phrase_cap,
        )
        rows.append({**r.to_dict(), **score})

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
