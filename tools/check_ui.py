import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert not app.exception,[e.message for e in app.exception]
assert not any(w.label=='Model connection' for w in app.selectbox)
assert not any(w.label=='Model' for w in app.text_input)
assert not any(b.label=='Use your own API key' for b in app.sidebar.button)
assert not app.sidebar.selectbox and not app.sidebar.number_input
assert [t.label for t in app.tabs] == ['Draft reply','Review queue','Evaluation lab','Operations & ROI','Safety lab','Knowledge base']
next(b for b in app.button if b.label=='Use your own API key').click().run()
assert not app.exception,[e.message for e in app.exception]
# Dialog submission is verified in the real browser: AppTest does not model
# this dialog's clear-on-submit form behavior consistently. Check rendering,
# isolated accepted state, and the clear action here without calling a provider.
assert any(t.label=='OpenAI API key' for t in app.text_input)
app.session_state['api_credentials']={'openai':'sk-proj-'+('ui_test_secret_'*4)}
next(b for b in app.button if b.label=='Cancel').click().run()
assert not app.exception,[e.message for e in app.exception]
assert 'byok_input' not in app.session_state or app.session_state['byok_input']==''
assert 'openai' in app.session_state['api_credentials']
other=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert 'api_credentials' not in other.session_state
assert app.session_state['workspace_id'] != other.session_state['workspace_id']
next(b for b in app.button if b.label=='Clear API key').click().run()
assert not app.exception,[e.message for e in app.exception]
assert 'api_credentials' not in app.session_state
next(s for s in app.selectbox if s.label=='Brand').set_value('SpotifyCares').run()
assert not app.exception,[e.message for e in app.exception]
assert 'Spotify' in next(t for t in app.text_area if t.label=='Customer message').value
print(json.dumps({'ui_errors':0,'tabs':6,'sidebar_removed':True,'key_dialog_and_clear':'passed',
                  'sessions_isolated':True,'brand_examples':'passed'}))

