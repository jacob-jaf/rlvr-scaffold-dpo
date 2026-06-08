#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--revision-sheet', type=Path, required=True)
    ap.add_argument('--summary-json', type=Path, required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.revision_sheet, keep_default_na=False)
    revision_required=df['revision_required'].astype(str).str.lower().isin(['true','1','yes'])
    revised_nonempty=df['chosen_response_revised'].astype(str).str.strip().ne('')
    dpo_use=df['dpo_use'].astype(str).str.lower().isin(['true','1','yes'])
    sft_unrevised=df['sft_use_if_unrevised'].astype(str).str.lower().isin(['true','1','yes'])
    blocking=(revision_required & ~revised_nonempty & ~sft_unrevised)
    summary={
        'rows': int(len(df)),
        'revision_required': int(revision_required.sum()),
        'chosen_response_revised_nonempty': int(revised_nonempty.sum()),
        'blocking_missing_required_revisions': int(blocking.sum()),
        'dpo_use_false': int((~dpo_use).sum()),
        'can_build_stage1_without_allow_missing': int(blocking.sum())==0,
        'blocking_pair_ids': df.loc[blocking,'pair_id'].astype(str).tolist()[:200],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
