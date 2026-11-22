"""Bounded CSV ingestion and session-local batch processing. No credential storage."""
import csv
import hashlib
import io
import json
import uuid

from .config import BRANDS
from .privacy import redact, risk_signals

MAX_ROWS=50
MAX_BYTES=1_000_000
ALIASES={b.lower():b for b in BRANDS}
ALIASES.update(amazon='AmazonHelp',apple='AppleSupport',uber='Uber_Support',
               spotify='SpotifyCares',delta='Delta')
SAMPLE='''message_id,brand,message,customer_vip
case-001,Amazon,My package is two days late and tracking has not changed.,false
case-002,Apple,My account was hacked and I cannot sign in.,false
case-003,Delta,My flight was delayed and I need help with my next step.,false
case-004,Spotify,Thank you for helping me get my music working again.,false
case-005,Uber,I need help with an incorrect trip charge.,true
'''


def parse_inbox(content):
    if len(content)>MAX_BYTES:
        raise ValueError('Use a CSV smaller than 1 MB.')
    try:
        reader=csv.DictReader(io.StringIO(content.decode('utf-8-sig'),newline=''),strict=True)
        fields=reader.fieldnames or []
        if len(fields)!=len(set(fields)) or not {'brand','message'}<=set(fields):
            raise ValueError('CSV needs unique column names including brand and message.')
        if set(fields)-{'brand','message','message_id','customer_vip'}:
            raise ValueError('Allowed columns: brand, message, message_id, customer_vip. Remove other columns before uploading.')
        rows=[]
        errors=[]
        seen=set()
        for number,row in enumerate(reader,start=1):
            if number>MAX_ROWS:
                raise ValueError('Each inbox can contain at most 50 messages.')
            issue=None
            brand=ALIASES.get((row.get('brand') or '').strip().lstrip('@').lower())
            message=(row.get('message') or '').strip()
            identifier=(row.get('message_id') or f'row-{number:03}').strip()
            vip=(row.get('customer_vip') or 'false').strip().lower()
            if None in row or any(value is None for value in row.values()):
                issue='Row has missing or extra CSV cells.'
            elif not brand:
                issue='Unsupported brand; use Amazon, Apple, Uber, Spotify, or Delta.'
            elif not message or len(message)>6000:
                issue='Message must contain 1–6,000 characters.'
            elif not identifier or len(identifier)>80:
                issue='Message ID must contain 1–80 characters.'
            elif identifier in seen:
                issue='Duplicate message ID; use a unique ID for each row.'
            elif vip not in {'true','false'}:
                issue='customer_vip must be true or false; only mark operator-confirmed VIPs.'
            if issue:
                errors.append({'row':number,'error':issue})
                continue
            seen.add(identifier)
            flags=risk_signals(message)
            rows.append({'row':number,'message_id':redact(identifier),'brand':brand,
                         'message':redact(message),'customer_vip':vip=='true',
                         'priority':'High' if flags or vip=='true' else 'Normal',
                         'priority_reason':', '.join(flags) or ('Operator-confirmed VIP' if vip=='true' else 'Standard intake')})
    except (UnicodeDecodeError,csv.Error) as error:
        raise ValueError('Use a valid UTF-8 CSV. Quote messages that contain commas or line breaks.') from error
    if not rows and not errors:
        raise ValueError('The CSV has no message rows.')
    # Stable order within each priority; this is a rule-based intake flag, not LLM triage.
    rows.sort(key=lambda r:(r['priority']!='High',r['row']))
    return rows,errors


def new_inbox(rows,errors,context):
    return {'batch_id':uuid.uuid4().hex,'rows':rows,'errors':errors,'context':context,
            'fingerprint':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
            'results':{},'claimed':[]}


def process_next(inbox,engine_factory,count,on_progress=None):
    """Claim before calling; interrupted requests are never automatically repeated."""
    if not 1<=count<=10:
        raise ValueError('Process 1–10 messages at a time.')
    completed=0
    for row in inbox['rows']:
        token=str(row['row'])
        if token in inbox['claimed']:
            continue
        if completed>=count:
            break
        inbox['claimed'].append(token)
        engine=None
        try:
            engine=engine_factory()
            result=engine.run(row['brand'],row['message'],
                case_id=f"batch:{inbox['batch_id']}:{row['row']}",
                customer_vip=row['customer_vip'],policy_context=inbox['context'],auto_send=False)
            # Store only a run reference; the normal session review store holds the trace.
            failed=any(s['stage']=='execution_error' for s in result.get('trace',[]))
            inbox['results'][token]={'run_id':result['run_id'],'error':failed}
        except Exception:
            inbox['results'][token]={'run_id':'','error':True}
            failed=True
        finally:
            http=getattr(getattr(engine,'client',None),'http',None)
            if http:
                http.close()
        completed+=1
        if on_progress:
            on_progress(completed)
        if failed:
            break
    return completed


def inbox_results(inbox,store):
    stored={r['run_id']:r for r in store.list_runs(limit=10000)}
    output=[]
    for row in inbox['rows']:
        token=str(row['row'])
        result=inbox['results'].get(token,{})
        record=stored.get(result.get('run_id'),{})
        payload=json.loads(record.get('payload','{}'))
        status=record.get('status') or ('error' if result.get('error') else
            'interrupted' if token in inbox['claimed'] else 'waiting')
        output.append({**row,'status':status,'run_id':result.get('run_id',''),
                       'intent':payload.get('triage',{}).get('intent',''),
                       'urgency':payload.get('triage',{}).get('urgency',''),
                       'readiness':payload.get('confidence_score'),
                       'reply':record.get('reply',''),'review_reason':record.get('review_reason',''),
                       'estimated_cost_usd':payload.get('usage',{}).get('estimated_cost_usd'),
                       'latency_seconds':round(payload.get('latency_ms',0)/1000,2) if payload else None,
                       'execution_error':bool(result.get('error'))})
    return output


def spreadsheet_safe(value):
    return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')) else value


def export_csv(rows):
    """Prevent spreadsheet formula execution, including after leading whitespace."""
    if not rows:
        return b''
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        safe={k:spreadsheet_safe(v) for k,v in row.items()}
        writer.writerow(safe)
    return stream.getvalue().encode('utf-8-sig')
