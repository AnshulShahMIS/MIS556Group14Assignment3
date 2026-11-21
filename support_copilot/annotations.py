"""Validate human label imports without accepting modified source tweets."""
from .privacy import redact

FIELDS={'intent':{'','refund','delay','account','bug','praise','other'},
        'urgency':{'','low','med','high'},'needs_human':{'','true','false'}}


def validate_labels(records,reference):
    if not isinstance(records,list) or not records or len(records)>100:
        raise ValueError('Import a JSON list containing 1–100 assigned cases.')
    originals={r['case_id']:r for r in reference}
    seen=set()
    clean=[]
    for row in records:
        if not isinstance(row,dict) or row.get('case_id') not in originals:
            raise ValueError('A label record has an unknown case ID.')
        case_id=row['case_id']
        if case_id in seen:
            raise ValueError('Each case ID must appear only once.')
        seen.add(case_id)
        original=originals[case_id]
        item={k:original[k] for k in ['case_id','brand','customer_text']}
        for field,allowed in FIELDS.items():
            value=str(row.get(field,'')).strip().lower()
            if value not in allowed:
                raise ValueError('Invalid '+field+' label for '+case_id+'.')
            item[field]=value
        item['escalation_reason']=redact(str(row.get('escalation_reason',''))[:1000])
        item['annotator']=redact(str(row.get('annotator',''))[:100])
        clean.append(item)
    return clean


def compare_raters(first,second):
    left={r['case_id']:r for r in first}
    right={r['case_id']:r for r in second}
    shared=sorted(left.keys() & right.keys())
    report={'overlap_cases':len(shared),'metrics':{},'disagreements':[]}
    for field in FIELDS:
        pairs=[(left[c][field],right[c][field]) for c in shared if left[c][field] and right[c][field]]
        n=len(pairs)
        agreement=sum(a==b for a,b in pairs)/n if n else None
        categories=FIELDS[field]-{''}
        expected=sum(sum(a==k for a,b in pairs)*sum(b==k for a,b in pairs) for k in categories)/(n*n) if n else None
        kappa=(agreement-expected)/(1-expected) if n and expected<1 else None
        report['metrics'][field]={'paired_labels':n,'agreement':agreement,'cohen_kappa':kappa}
    for case_id in shared:
        fields=[f for f in FIELDS if left[case_id][f] and right[case_id][f] and left[case_id][f]!=right[case_id][f]]
        if fields:
            report['disagreements'].append({'case_id':case_id,'fields':fields})
    return report
