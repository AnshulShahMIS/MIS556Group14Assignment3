"""Prepare blinded reply-quality forms from a completed benchmark revision."""
import argparse
import hashlib
import json
import random
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--per-brand',type=int,default=6)
args=parser.parse_args()
manifest=json.loads((args.report/'manifest.json').read_text(encoding='utf-8'))
cases=[json.loads(p.read_text(encoding='utf-8')) for p in (args.report/'cases').glob('*.json')]
cases=[c for c in cases if c.get('status')=='complete']
selected=[]
for brand in sorted({c['brand'] for c in cases}):
    group=sorted([c for c in cases if c['brand']==brand],key=lambda c:hashlib.sha256(c['case_id'].encode()).hexdigest())
    selected.extend(group[:args.per_brand])
forms=[];key=[]
for case in selected:
    systems=list(case['outputs'])
    random.Random(case['case_id']).shuffle(systems)
    issue=case['outputs']['full_agent']['customer_text']
    for label,system in zip(['A','B','C'],systems):
        output=case['outputs'][system]
        forms.append({'case_id':case['case_id'],'brand':case['brand'],'customer_text':issue,
            'variant':label,'candidate_reply':output['final_reply'],
            'decision':'human_review' if output['route']=='human_review' else
                       'ignored' if output['route']=='ignored' else 'reply_prepared',
            'policy_notes':case['outputs']['full_agent'].get('policies',[]),
            'helpfulness':'','factual_support':'','tone':'','notes':'','rater':''})
        key.append({'case_id':case['case_id'],'variant':label,'system':system})
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'blinded_reply_forms.json').write_text(json.dumps(forms,indent=2),encoding='utf-8')
(args.output/'PRIVATE_system_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
(args.output/'review_manifest.json').write_text(json.dumps({'benchmark_fingerprint':manifest['fingerprint'],
    'unique_cases':len(selected),'reply_forms':len(forms),'labels_status':'blank',
    'blinding_limit':'Routing decisions remain visible. Do not give reviewers the private key or judge scores.'},indent=2),encoding='utf-8')
print(json.dumps({'cases':len(selected),'forms':len(forms),'fingerprint':manifest['fingerprint']}))
