"""Actual Streamlit batch UI with real retrieval and a test-only offline provider."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import support_copilot.provider as provider
from streamlit.testing.v1 import AppTest
from support_copilot.storage import RunStore

class QAProvider(provider.DemoClient):
    def __init__(self,settings):
        self.http=SimpleNamespace(close=lambda:None)

provider.ProviderClient=QAProvider
app=AppTest.from_file(str(root/'app.py'),default_timeout=120).run()
assert not app.exception,[e.message for e in app.exception]
next(b for b in app.button if b.label=='Try sample inbox').click().run()
assert not app.exception,[e.message for e in app.exception]
inbox=app.session_state['batch_inbox']
batch_id=inbox['batch_id']
assert len(inbox['rows'])==5
assert not next(b for b in app.button if b.label=='Process next 5 messages').value
assert next(b for b in app.button if b.label=='Process next 5 messages').disabled
app.session_state['api_credentials']={'openai':'sk-demo-not-a-real-key-for-ui-checks-only'}
app.run()
next(s for s in app.select_slider if s.label=='Messages per processing action').set_value(3).run()
next(b for b in app.button if b.label=='Process next 3 messages').click().run()
assert not app.exception,[e.message for e in app.exception]
assert len(app.session_state['batch_inbox']['results'])==3
next(b for b in app.button if b.label=='Process next 2 messages').click().run()
assert not app.exception,[e.message for e in app.exception]
assert len(app.session_state['batch_inbox']['results'])==5
store=RunStore(Path(os.getenv('SUPPORT_RUNTIME_DIR',root/'.runtime'))/'sessions'/app.session_state['workspace_id']/'runs.sqlite')
assert len(store.list_runs())==5
assert len(store.list_runs('pending'))>=2
assert not store.list_runs('simulated_sent')
next(b for b in app.button if b.label=='Try sample inbox').click().run()
assert app.session_state['batch_inbox']['batch_id']==batch_id
assert len(store.list_runs())==5
assert not app.exception,[e.message for e in app.exception]
report={'ui_exceptions':0,'sample_rows':5,'processing_actions':[3,2],
        'human_review_integrated':True,'repeat_import_preserves_progress':True,
        'auto_send_disabled':True,'provider':'test-only offline provider; no paid requests'}
(root/'reports/batch_journey_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
