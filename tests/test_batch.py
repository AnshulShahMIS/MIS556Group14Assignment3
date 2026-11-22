import json
from types import SimpleNamespace

import pytest

from support_copilot.batch import SAMPLE,parse_inbox,new_inbox,process_next,inbox_results,export_csv
from support_copilot.storage import RunStore


def test_import_prioritizes_risk_and_vip_and_redacts():
    rows,errors=parse_inbox(SAMPLE.encode())
    assert not errors and len(rows)==5
    assert [r['message_id'] for r in rows[:2]]==['case-002','case-005']
    private,_=parse_inbox(b'brand,message\n@AmazonHelp,Email me at student@example.com\n')
    assert private[0]['brand']=='AmazonHelp' and '[EMAIL]' in private[0]['message']


def test_invalid_rows_excluded_and_duplicate_ids_rejected():
    rows,errors=parse_inbox(b'brand,message,message_id,customer_vip\nAmazon,Help,a,false\nApple,Help,a,false\nUnknown,Help,b,false\nUber,Help,c,maybe\n')
    assert len(rows)==1 and len(errors)==3
    assert all(set(e)=={'row','error'} for e in errors)


@pytest.mark.parametrize('content',[
    b'brand,brand,message\nAmazon,Apple,help\n',b'brand,message,api_key\nAmazon,help,secret\n',
    b'brand,text\nAmazon,help\n',b'brand,message\n'+b'Amazon,help\n'*51,
    b'x'*1_000_001,b'brand,message\nAmazon,"unterminated',b'\xff\xfe',
])
def test_import_rejects_malformed_or_unbounded_files(content):
    with pytest.raises(ValueError):
        parse_inbox(content)


def test_csv_export_neutralizes_formulas():
    exported=export_csv([{'message':'  =WEBSERVICE("example")','reply':'+1+1','status':'ready'}]).decode('utf-8-sig')
    assert "'  =WEBSERVICE" in exported and "'+1+1" in exported


def test_processing_resumes_without_duplicate_calls_and_exports_review(tmp_path):
    rows,errors=parse_inbox(SAMPLE.encode())
    inbox=new_inbox(rows,errors,'historical_replay')
    store=RunStore(tmp_path/'runs.sqlite')
    calls=[]
    closed=[]
    class Engine:
        client=SimpleNamespace(http=SimpleNamespace(close=lambda:closed.append(True)))
        def run(self,brand,message,**kwargs):
            calls.append(kwargs)
            result={'run_id':str(len(calls)),'brand':brand,'route':'human_review' if kwargs['customer_vip'] else 'ready_to_send',
                    'customer_text':message,'review_reason':'VIP' if kwargs['customer_vip'] else '',
                    'final_reply':'Thanks for reaching out.','trace':[],
                    'triage':{'intent':'other','urgency':'med'},'confidence_score':75,
                    'usage':{'estimated_cost_usd':0.001},'latency_ms':1200}
            store.save(result)
            return result
    assert process_next(inbox,Engine,3)==3
    assert process_next(inbox,Engine,3)==2
    assert process_next(inbox,Engine,3)==0
    assert len(calls)==len(closed)==5
    assert all(c['auto_send'] is False and c['policy_context']=='historical_replay' for c in calls)
    assert len({c['case_id'] for c in calls})==5
    results=inbox_results(inbox,store)
    pending=next(r for r in results if r['status']=='pending')
    store.review(pending['run_id'],'approved','Test reviewer',reply='Please contact official support for help.')
    reviewed=next(r for r in inbox_results(inbox,store) if r['run_id']==pending['run_id'])
    assert reviewed['status']=='approved' and reviewed['reply']=='Please contact official support for help.'
    assert not store.list_runs('simulated_sent')
    assert 'api_key' not in json.dumps(inbox)


def test_failure_stops_action_and_is_not_automatically_retried(tmp_path):
    rows,errors=parse_inbox(SAMPLE.encode())
    inbox=new_inbox(rows,errors,'current_simulation')
    class Broken:
        def run(self,*args,**kwargs):
            raise RuntimeError('secret exception details must not be retained')
    assert process_next(inbox,Broken,5)==1
    assert len(inbox['claimed'])==1 and len(inbox['results'])==1
    assert 'secret exception' not in json.dumps(inbox)
    assert process_next(inbox,Broken,5)==1
    assert len(inbox['claimed'])==2


def test_interrupted_request_stays_claimed():
    rows,errors=parse_inbox(b'brand,message\nAmazon,Help\n')
    inbox=new_inbox(rows,errors,'current_simulation')
    class Interrupted:
        def run(self,*args,**kwargs):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        process_next(inbox,Interrupted,1)
    assert process_next(inbox,Interrupted,1)==0
