"""Real Chroma retrieval with explicitly rule-based agent components; no paid calls."""
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from support_copilot.config import Settings
from support_copilot.engine import SupportEngine
from support_copilot.provider import DemoClient
from support_copilot.retrieval import Retriever
from support_copilot.storage import RunStore

settings=Settings()
retriever=Retriever(settings)
retriever.connect()
store=RunStore(settings.runtime/'workflow_verification.sqlite')
cases=[('AmazonHelp','My package is late.'),('AppleSupport','The app keeps crashing.'),
       ('Uber_Support','I was charged a cancellation fee.'),
       ('SpotifyCares','I was charged twice this month.'),('Delta','My checked bag did not arrive.')]
results=[]
for brand,text in cases:
    result=SupportEngine(DemoClient(),retriever,store).run(brand,text,auto_send=True)
    assert len(result['cases'])==5 and all(c['brand']==brand for c in result['cases'])
    assert result['policy_assessment']['evidence_sufficient']
    assert result['confidence_score']==100 and result['route']=='auto_sent'
    assert len(result['final_reply'])<=280
    results.append(result)
vip=SupportEngine(DemoClient(),retriever,store).run('Delta','Please help with a flight.',customer_vip=True,auto_send=True)
assert vip['route']=='human_review' and not vip['cases']
controlled=SupportEngine(DemoClient(),retriever,store).run('AmazonHelp','My package is late.',inject_bad_draft=True,auto_send=True)
assert not controlled['critic_history'][0]['passed'] and len(controlled['draft_history'])==2
report={'mode':'offline_demo components with real vector retrieval; not live LLM validation',
        'summary':{'brands':5,'same_brand_top_five':'passed','policy_check':'passed',
                   'vip_routing':'passed','critic_revision':'passed','confidence_and_outbox':'passed'},
        'routine_cases':results,'vip_case':vip,'controlled_revision':controlled}
(root/'reports/current_workflow_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['summary'],indent=2))
