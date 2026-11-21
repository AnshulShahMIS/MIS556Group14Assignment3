"""Exercise the actual UI with real retrieval and a test-only deterministic provider.

No API calls or production provider changes. This does not validate live LLM behavior.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import support_copilot.provider as provider
import support_copilot.labs as labs
from support_copilot.storage import RunStore
from streamlit.testing.v1 import AppTest


class QAProvider(provider.DemoClient):
    def __init__(self,settings):
        self.http=SimpleNamespace(close=lambda:None)


provider.ProviderClient=QAProvider
labs.ProviderClient=QAProvider
app=AppTest.from_file(str(root/'app.py'),default_timeout=120).run()
assert not app.exception,[e.message for e in app.exception]
app.session_state['api_credentials']={'openai':'sk-proj-'+('demo_qa_not_a_real_key_'*3)}
app.run()
next(b for b in app.button if b.label=='Prepare reply').click().run()
assert not app.exception,[e.message for e in app.exception]
routine=app.session_state['last_result']
assert routine['route']=='auto_sent' and routine['confidence_score']==100
assert len(routine['cases'])==5 and all(r['brand']=='AmazonHelp' for r in routine['cases'])
assert len(routine['final_reply'])<=280
assert {'triage','router','retrieval','policy_check','draft','critic','finalize'} <= {s['stage'] for s in routine['trace']}
next(c for c in app.checkbox if c.label=='Known VIP customer (operator confirmed)').check().run()
next(b for b in app.button if b.label=='Prepare reply').click().run()
assert not app.exception,[e.message for e in app.exception]
vip=app.session_state['last_result']
assert vip['route']=='human_review' and vip['cases']==[]
next(t for t in app.text_input if t.label=='Reviewer name').set_value('Demo QA')
next(b for b in app.button if b.label=='Approve edited reply').click().run()
assert not app.exception,[e.message for e in app.exception]
store=RunStore(root/'.runtime/sessions'/app.session_state['workspace_id']/'runs.sqlite')
assert len(store.list_runs('approved'))==1
assert len(store.list_runs('simulated_sent'))==1
next(b for b in app.button if b.label=='Run selected safety tests').click().run()
assert not app.exception,[e.message for e in app.exception]
safety=app.session_state['safety_results']
assert len(safety)==5 and all(r['passed'] for r in safety)
next(b for b in app.button if b.label=='Clear API key').click().run()
assert 'api_credentials' not in app.session_state
result={'mode':'test-only rule-based provider with real UI and real Chroma retrieval; no live LLM calls',
        'routine_reply':'passed','five_same_brand_cases':'passed','confidence_display':'passed',
        'vip_escalation':'passed','human_approval_persisted':'passed','simulated_outbox':'passed',
        'safety_tests_passed':len(safety),'clear_key':'passed','ui_exceptions':0,
        'live_model_validation':'pending real session key'}
(root/'reports/demo_journey_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
