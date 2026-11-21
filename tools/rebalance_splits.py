"""Rebalance the prepared pool into brand-balanced component-disjoint sets."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    data = out / 'data'
    rows = []
    for split in ['index','development','test']:
        rows.extend(read_rows(data / f'{split}_pairs.jsonl'))
    assert len({r['case_id'] for r in rows}) == len(rows)
    assert len({r['normalized_query_hash'] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['conversation_id']].append(row)
    brands = sorted({r['brand'] for r in rows})
    selected = {'test': [], 'development': []}
    heldout_groups = set()
    for brand in brands:
        # Singleton refers to selected pairs, not the original conversation's
        # number of tweets. Its entire source component remains held out.
        candidates = [r for r in rows if r['brand'] == brand and len(groups[r['conversation_id']]) == 1]
        candidates.sort(key=lambda r: hashlib.sha256(('balanced-v1-' + r['conversation_id']).encode()).digest())
        if len(candidates) < 60:
            raise ValueError(f'Insufficient independent groups for {brand}')
        selected['test'].extend(candidates[:40])
        selected['development'].extend(candidates[40:60])
        heldout_groups.update(r['conversation_id'] for r in candidates[:60])
    selected['index'] = [r for r in rows if r['conversation_id'] not in heldout_groups]
    assert len(selected['index']) == 50000
    for split, records in selected.items():
        records.sort(key=lambda r: (r['brand'], r['created_at'], r['case_id']))
        write_rows(data / f'{split}_pairs.jsonl', records)
    write_rows(data / 'test_inputs.jsonl', [
        {k:r[k] for k in ['case_id','brand','created_at','customer_text']} for r in selected['test']])
    annotations = []
    for brand in brands:
        examples = [r for r in selected['test'] if r['brand'] == brand][:20]
        for row in examples:
            annotations.append({**{k:row[k] for k in ['case_id','brand','customer_text']},
                'intent':'','urgency':'','needs_human':'','escalation_reason':'','annotator':''})
    write_rows(data / 'triage_annotation_template.jsonl', annotations)
    report_path = out / 'data_audit.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['initial_split_method'] = report['split_method']
    report['split_method'] = 'Rebalanced deterministic hash sample of connected groups from the prepared 50,300-pair pool, with exact per-brand quotas. Test/development groups contain one selected pair; their source conversations can contain multiple tweets.'
    report['split_counts'] = {k:len(v) for k,v in selected.items()}
    report['split_brand_counts'] = {k:dict(Counter(r['brand'] for r in v)) for k,v in selected.items()}
    report['split_time_ranges'] = {k:{'start':min(r['created_at'] for r in v),'end':max(r['created_at'] for r in v)} for k,v in selected.items()}
    report['sample_compatibility'] = json.loads((out / 'sample_compatibility_audit.json').read_text(encoding='utf-8'))
    report['sample_handling'] = 'Full twcs.csv is authoritative. Sample IDs are incompatible. Do not join the two attachments by ID or use sample.csv as an independent holdout.'
    report['date_handling'] = 'Data were collected around 2017, but retained conversation roots can be older. Use actual timestamps rather than assuming every tweet was written in 2017.'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    for left,right in [('index','test'),('index','development'),('development','test')]:
        assert not ({r['conversation_id'] for r in selected[left]} & {r['conversation_id'] for r in selected[right]})
        assert not ({r['normalized_query_hash'] for r in selected[left]} & {r['normalized_query_hash'] for r in selected[right]})
    print(json.dumps({'counts':report['split_counts'],'dates':report['split_time_ranges'],'sample':report['sample_compatibility']},indent=2))


if __name__ == '__main__':
    main()
