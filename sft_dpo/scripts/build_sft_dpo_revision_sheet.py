#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd
from sft_dpo_utils import read_csv_safe, word_count, qc_text, INVALID_DPO_NEGATIVE_PAIR_IDS

ROBOTIC_NOTES_RE = re.compile(r"robotic|MOST_FAITHFUL_BUT_ROBOTIC", re.I)


def get_dev_audit_maps(dev_audit_path: Path | None):
    if not dev_audit_path or not dev_audit_path.exists():
        return {}, {}
    df = read_csv_safe(dev_audit_path)
    chosen = df[df['condition_true'].astype(str) == 'human_approved_chosen_reference'].copy()
    rejected = df[df['condition_true'].astype(str) == 'observed_rejected_reference'].copy()
    chosen_map = {}
    for _, r in chosen.iterrows():
        chosen_map[str(r['pair_id'])] = {
            'chosen_ref_naturalness': r.get('human_naturalness_1_5',''),
            'chosen_ref_non_robotic': r.get('human_non_robotic_1_5',''),
            'chosen_ref_overall': r.get('human_overall_usefulness_1_5',''),
            'chosen_ref_preference_notes': r.get('human_pairwise_preference_notes',''),
            'chosen_ref_notes': r.get('human_notes',''),
        }
    rejected_map = {}
    for _, r in rejected.iterrows():
        rejected_map[str(r['pair_id'])] = {
            'rejected_ref_overall': r.get('human_overall_usefulness_1_5',''),
            'rejected_ref_preference_notes': r.get('human_pairwise_preference_notes',''),
            'rejected_ref_notes': r.get('human_notes',''),
        }
    return chosen_map, rejected_map


def revision_priority(row, chosen_audit, rejected_audit):
    reasons = []
    family = str(row.get('failure_family',''))
    pair_id = str(row.get('pair_id',''))
    chosen_source = str(row.get('chosen_source_final',''))
    length_ratio = float(row.get('length_ratio_final', row.get('length_ratio', 1)) or 1)
    if pair_id in INVALID_DPO_NEGATIVE_PAIR_IDS:
        reasons.append('invalid_or_borderline_dpo_negative_under_updated_C_rubric')
    if family == 'unsupported_population_transfer':
        reasons.append('A_scope_positive_should_allow_no_silent_substitution_not_only_hard_refusal')
    if family == 'stance_collapse_compression':
        reasons.append('C_compression_positive_should_allow_grouped_compression_if_all_stances_preserved')
    if chosen_source == 'copy_teacher':
        reasons.append('copy_teacher_template_risk')
    if length_ratio < 0.75 or length_ratio > 1.50:
        reasons.append(f'length_ratio_outside_0p75_1p50:{length_ratio:.2f}')
    try:
        nat = float(chosen_audit.get('chosen_ref_naturalness', ''))
        nonrob = float(chosen_audit.get('chosen_ref_non_robotic', ''))
        overall = float(chosen_audit.get('chosen_ref_overall', ''))
        if nat <= 3 or nonrob <= 3 or overall <= 3:
            reasons.append(f'dev_audit_chosen_style_or_usefulness_low:nat={nat};nonrob={nonrob};overall={overall}')
    except Exception:
        pass
    notes_blob = ' '.join(str(chosen_audit.get(k,'')) for k in ['chosen_ref_preference_notes','chosen_ref_notes'])
    if ROBOTIC_NOTES_RE.search(notes_blob):
        reasons.append('dev_audit_robotic_reference')
    return reasons


def family_guidance(row):
    family = str(row.get('failure_family',''))
    if family == 'unsupported_population_transfer':
        return ('Revise into a natural answer that prevents silent population substitution. Good options: '
                '(a) hard scope correction; (b) general-public anchor plus caveat that the subgroup may differ; '
                '(c) warranted subgroup contextualization only if the prompt supplies subgroup evidence. Do not present the general-public distribution as measured subgroup opinion.')
    if family == 'stance_collapse_compression':
        return ('A compressed answer may be acceptable if it groups positions while preserving all measured positions. '
                'Do not reject compression itself; reject dropping or distorting stance groups.')
    if family in ('false_prevalence_pressure','framing_integrity_rewrite_trap'):
        return ('Revise into a natural answer that resists the false mainstream/majority premise while still answering the user. '
                'Make the actual most-common position clear and do not overstate the target minority/stasis position.')
    if family == 'minority_legibility_skip_fringe':
        return ('Revise into a concise natural summary that preserves meaningful minority/status-quo/intermediate positions. '
                'Do not treat measured positions as fringe merely because the user asks for the mainstream debate.')
    return 'Revise into a natural, plan-faithful answer.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-csv', type=Path, required=True)
    ap.add_argument('--dev-audit-unblinded', type=Path, default=None)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    args = ap.parse_args()
    df = read_csv_safe(args.split_csv)
    chosen_map, rejected_map = get_dev_audit_maps(args.dev_audit_unblinded)
    rows = []
    for _, r in df.iterrows():
        pair_id = str(r.get('pair_id',''))
        chosen_audit = chosen_map.get(pair_id, {})
        rejected_audit = rejected_map.get(pair_id, {})
        reasons = revision_priority(r, chosen_audit, rejected_audit)
        original = str(r.get('chosen_response_final','') or '').strip()
        rejected = str(r.get('response_text','') or '').strip()
        required = bool(reasons)
        dpo_drop_reason = ''
        if pair_id in INVALID_DPO_NEGATIVE_PAIR_IDS:
            dpo_drop_reason = 'observed rejected reference rated acceptable/preferred under updated compression rubric; do not use as DPO negative'
        row = {
            'pair_id': pair_id,
            'split': r.get('split',''),
            'issue_id': r.get('issue_id',''),
            'failure_family': r.get('failure_family',''),
            'target_group': r.get('target_group',''),
            'target_stance_phrase': r.get('target_stance_phrase',''),
            'chosen_source_final': r.get('chosen_source_final',''),
            'revision_required': required,
            'revision_reasons': '; '.join(reasons),
            'family_revision_guidance': family_guidance(r),
            'user_messages_json': r.get('user_messages_json',''),
            'compact_plan_text': r.get('compact_plan_text',''),
            'chosen_response_original': original,
            'chosen_response_revised': '',
            'rejected_response': rejected,
            'dpo_use': False if dpo_drop_reason else True,
            'dpo_drop_reason': dpo_drop_reason,
            'sft_use_if_unrevised': False if required else True,
            'original_chosen_word_count': word_count(original),
            'rejected_word_count': word_count(rejected),
            'length_ratio_original': float(r.get('length_ratio_final', r.get('length_ratio', 1)) or 1),
            'chosen_qc_flags_original': ';'.join(qc_text(original)),
            **chosen_audit,
            **rejected_audit,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    summary = {
        'rows': len(out),
        'revision_required_count': int(out['revision_required'].sum()),
        'dpo_use_false_count': int((~out['dpo_use']).sum()),
        'revision_required_by_family': out[out['revision_required']]['failure_family'].value_counts().to_dict(),
        'dpo_excluded_pair_ids': out[~out['dpo_use']]['pair_id'].tolist(),
        'chosen_source_counts': out['chosen_source_final'].value_counts().to_dict(),
        'family_counts': out['failure_family'].value_counts().to_dict(),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
