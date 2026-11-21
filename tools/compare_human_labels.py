"""Compare independent labels and create an unadjudicated consensus draft."""
import argparse
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from support_copilot.annotations import validate_labels,compare_raters,FIELDS
from support_copilot.retrieval import read_jsonl

parser=argparse.ArgumentParser()
parser.add_argument('--rater-a',type=Path,required=True)
parser.add_argument('--rater-b',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
reference=read_jsonl(root/'data/triage_annotation_template.jsonl')
a=validate_labels(json.loads(args.rater_a.read_text(encoding='utf-8')),reference)
b=validate_labels(json.loads(args.rater_b.read_text(encoding='utf-8')),reference)
report=compare_raters(a,b)
left={r['case_id']:r for r in a};right={r['case_id']:r for r in b}
consensus=[];pending=[]
for source in reference:
    case_id=source['case_id']
    entries=[m[case_id] for m in [left,right] if case_id in m]
    complete=bool(entries) and all(all(r[f] for f in FIELDS) for r in entries)
    agree=complete and all(len({r[f] for r in entries})==1 for f in FIELDS)
    row={**source}
    if agree:
        row.update({f:entries[0][f] for f in FIELDS})
        row['annotator']=' + '.join(r['annotator'] or 'Unidentified rater' for r in entries)
        row['escalation_reason']=' / '.join(r['escalation_reason'] for r in entries if r['escalation_reason'])
    else:
        pending.append(case_id)
        row['escalation_reason']='Adjudication or missing independent labels: '+str([{f:r[f] for f in FIELDS} for r in entries])
    consensus.append(row)
report['cases_requiring_adjudication_or_completion']=pending
report['ready_cases']=100-len(pending)
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'agreement_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(args.output/'consensus_draft.json').write_text(json.dumps(consensus,indent=2),encoding='utf-8')
print(json.dumps({'paired_cases':report['overlap_cases'],'ready_cases':report['ready_cases'],'pending':len(pending)}))
