import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from support_copilot.config import Settings
from support_copilot.engine import SupportEngine
from support_copilot.labs import SAFETY_CASES,run_safety_case
from support_copilot.provider import ProviderClient
from support_copilot.retrieval import Retriever
from support_copilot.storage import RunStore

settings=Settings.from_env()
retriever=Retriever(settings)
retriever.connect()
store=RunStore(settings.runtime/'acceptance.sqlite')
cases=[('AmazonHelp','My package is two days late and tracking has not changed.'),
       ('AppleSupport','This app keeps crashing when I open it. What should I try?'),
       ('Uber_Support','I was charged a cancellation fee. How do I ask for a review?'),
       ('SpotifyCares','I can see two completed Spotify charges this month. Why was I charged twice?'),
       ('Delta','My checked bag did not arrive. How can I report it?')]
def run(pair):
    client=ProviderClient(settings)
    try:
        return SupportEngine(client,retriever,store).run(*pair)
    finally:
        client.http.close()
with ThreadPoolExecutor(max_workers=4) as pool:
    routine=list(pool.map(run,cases))
with ThreadPoolExecutor(max_workers=4) as pool:
    safety=list(pool.map(lambda case:run_safety_case(settings,retriever,case),SAFETY_CASES))
client=ProviderClient(settings)
historical=SupportEngine(client,retriever,store).run('AmazonHelp','My package is late.',policy_context='historical_replay')
client.http.close()
assert historical['policies']==[] and historical['route']=='human_review'
for result in routine:
    assert not any(e['stage']=='execution_error' for e in result['trace'])
    assert all(r['brand']==result['brand'] for r in result['cases'])
assert all(r['passed'] for r in safety)
report={'routine_cases':routine,'controlled_safety_cases':safety,'historical_replay':historical,
        'summary':{'routine_cases':len(routine),'brands':len({r['brand'] for r in routine}),
                   'controlled_safety_passed':sum(r['passed'] for r in safety),
                   'controlled_safety_total':len(safety),'historical_policy_exclusion':'passed'}}
(root/'reports/live_acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['summary'],indent=2))

