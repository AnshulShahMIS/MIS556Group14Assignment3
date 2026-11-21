import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from support_copilot.config import APP_ROOT, BRANDS, Settings
from support_copilot.engine import SupportEngine
from support_copilot.evaluation import evaluate, triage_metrics
from support_copilot.privacy import validate_reply
from support_copilot.provider import ProviderClient
from support_copilot.retrieval import Retriever, read_jsonl
from support_copilot.storage import RunStore
from support_copilot.credentials import accept_key, active_key, clear_keys
from support_copilot.labs import SAFETY_CASES, run_safety_case, roi_scenario
from support_copilot.annotations import validate_labels

st.set_page_config(page_title='Support Copilot',page_icon=str(APP_ROOT/'assets/favicon.png'),layout='wide')
st.markdown('''<style>
[data-testid="stMainBlockContainer"]{max-width:1200px;padding-top:4.5rem;padding-bottom:3rem}
h1{letter-spacing:-.04em} .stTextArea textarea{font-size:1rem;line-height:1.6}
.stButton button{min-height:2.8rem;height:auto;white-space:normal;padding:.65rem 1rem}
.stButton button p{white-space:normal;line-height:1.35}
@media(max-width:640px){[data-testid="stMainBlockContainer"]{padding-top:4rem}}
</style>''',unsafe_allow_html=True)
# The web app never falls back to the owner's environment/API key.
base=Settings(runtime=Path(os.getenv('SUPPORT_RUNTIME_DIR',APP_ROOT/'.runtime')))
if 'workspace_id' not in st.session_state:
    st.session_state['workspace_id']=uuid.uuid4().hex
workspace=base.runtime/'sessions'/st.session_state['workspace_id']


def discard_key_input():
    st.session_state['byok_input']=''
    st.session_state.pop('credential_error',None)


@st.dialog('Use your own API key',width='large',on_dismiss=discard_key_input)
def credential_dialog():
    label='OpenAI'
    st.write('Add your key for live support drafting, evaluation, and safety tests. Requests use your provider account’s quota and may incur charges.')
    url='https://platform.openai.com/api-keys'
    st.link_button('Get a '+label+' API key ↗',url)
    with st.form('connect_credentials',clear_on_submit=True):
        candidate=st.text_input(label+' API key',key='byok_input',type='password',autocomplete='off',
                      placeholder='Paste your '+label+' API key',max_chars=500)
        st.caption('Kept in this browser session and the app server’s memory. The app does not save keys to files, databases, browser storage, traces, model prompts, or exports. The key goes only to the selected provider in an HTTPS request header.')
        st.caption('Clear the key when finished. A disconnected session can remain briefly in server memory before cleanup. Hosted deployments require HTTPS and a trusted operator; local mode runs on your own computer.')
        st.caption('ChatGPT subscriptions and OpenAI API billing are separate. Authentication and quota are checked on your first live request.')
        submitted=st.form_submit_button('Use key for this session',type='primary')
    if submitted:
        try:
            accept_key(st.session_state,'openai',candidate)
        except ValueError as error:
            st.error(str(error))
        else:
            st.rerun(scope='app')
    clear,cancel=st.columns(2)
    if clear.button('Clear all session keys',on_click=clear_keys,args=(st.session_state,)):
        st.rerun()
    if cancel.button('Cancel',on_click=discard_key_input):
        st.rerun()


@st.cache_resource
def get_retriever(runtime):
    resource=Retriever(replace(base,runtime=Path(runtime)))
    try:
        resource.connect()
    except ValueError:
        if (APP_ROOT/'data/index_embeddings.npz').exists():
            with st.spinner('Preparing the support knowledge base. First startup may take a few minutes…'):
                resource.build()
    return resource


@st.cache_data
def development_cases():
    return read_jsonl(APP_ROOT/'data/development_pairs.jsonl')


context_label=st.session_state.get('policy_context_choice','Current support simulation')
policy_context='current_simulation' if context_label.startswith('Current') else 'historical_replay'
input_price=st.session_state.get('input_token_price',0.40)
output_price=st.session_state.get('output_token_price',1.60)
session_key=active_key(st.session_state,'openai')
header,account=st.columns([3,1.4],vertical_alignment='center')
with header:
    st.title('Support Copilot')
with account:
    if st.button('Manage API key' if session_key else 'Use your own API key',type='primary',width='stretch'):
        credential_dialog()
    if session_key:
        st.caption('API key connected for this session.')
        st.button('Clear API key',on_click=clear_keys,args=(st.session_state,))
    else:
        st.caption('Connect a key to use live AI features.')

settings=replace(base,api_key=session_key,
                 input_price=input_price if input_price else None,output_price=output_price if output_price else None,
                 evaluation_root=workspace/'evaluations')
retriever=get_retriever(str(settings.runtime))
store=RunStore(workspace/'runs.sqlite')
try:
    index=retriever.connect()
except ValueError:
    index=None

st.write('Help customer-support teams handle incoming tweets with real conversation examples, brand policy, and human review.')
if not index:
    st.info('Build the local retrieval index before drafting a reply. The first build downloads the embedding model.')
    if st.button('Build retrieval index',type='primary'):
        bar=st.progress(0,text='Loading the embedding model')
        try:
            retriever.build(lambda n,total:bar.progress(n/total,text=f'Indexed {n:,} of {total:,} cases'))
            st.rerun()
        except Exception as error:
            st.error(f'Index build failed: {type(error).__name__}. See the local server log for details.')

draft_tab,queue_tab,eval_tab,ops_tab,safety_tab,knowledge_tab=st.tabs(
    ['Draft reply','Review queue','Evaluation lab','Operations & ROI','Safety lab','Knowledge base'])


def display_result(result):
    route=result['route']
    if route=='human_review':
        st.warning('Human review required — '+result['review_reason'])
    elif route=='ignored':
        st.info('Marked off-topic — '+result['review_reason'])
    elif route=='auto_sent':
        st.success('Auto-send simulated: the reply was saved to your local outbox. No public tweet was posted.')
    else:
        st.success('Reply passed checks and was saved to the local ready queue.')
    if result.get('final_reply'):
        st.text_area('Prepared reply',result['final_reply'],height=110,disabled=True,key='result_'+result['run_id'])
        st.caption(f"{len(result['final_reply'])}/280 characters · {result['latency_ms']/1000:.1f} seconds · {result['usage']['llm_calls']} LLM calls")
    usage=result['usage']
    if result.get('triage'):
        triage=result['triage']
        intent,sentiment,urgency,triaged_brand=st.columns(4)
        intent.metric('Intent',triage['intent'].title())
        sentiment.metric('Sentiment',f"{triage['sentiment']:+.2f}")
        urgency.metric('Urgency',triage['urgency'].title())
        triaged_brand.metric('Brand',result['brand'])
    score=result.get('confidence_score',0)
    st.progress(score/100,text=f'Confidence / readiness: {score}/100')
    st.caption(result.get('confidence_note','Heuristic readiness score, not a probability of correctness.'))
    with st.expander('Why this confidence score?'):
        st.json(result.get('confidence_checks',{}))
    if usage.get('estimated_cost_usd') is not None:
        st.caption(f"Estimated token cost ${usage['estimated_cost_usd']:.5f}")
    timeline,evidence=st.columns([1,1])
    with timeline:
        st.subheader('Execution trace')
        for step in result['trace']:
            with st.expander(step['stage'].replace('_',' ').title()+' — '+step['summary'][:100]):
                st.json(step)
    with evidence:
        st.subheader('Supporting evidence')
        for policy in result.get('policies',[]):
            st.markdown('**'+policy['topic']+'**')
            st.write(policy['text'])
            st.markdown('[Official source]('+policy['source_url']+')')
        for case in result.get('cases',[]):
            with st.expander(case['case_id']+f" · similarity {case['similarity']:.2f}"):
                st.write(case['customer_text'])
                st.caption('Historical reply')
                st.write(case['historical_reply'])
    st.download_button('Download this run',json.dumps(result,indent=2,ensure_ascii=False),file_name='support_run_'+result['run_id']+'.json',mime='application/json')


with draft_tab:
    st.caption('Choose a brand, paste a customer message, and prepare a reply. Or try a sample below.')
    brand_names={'AmazonHelp':'Amazon','AppleSupport':'Apple','Uber_Support':'Uber','SpotifyCares':'Spotify','Delta':'Delta Air Lines'}
    brand=st.selectbox('Brand',BRANDS,format_func=lambda value:brand_names[value])
    examples_by_brand={
        'AmazonHelp':{'Late package':"My package is two days late and tracking hasn't changed. Can you help?",
                      'Missing delivery':'My order says delivered, but I cannot find the package.',
                      'Account security':'My account was hacked and someone changed my password.'},
        'AppleSupport':{'App keeps crashing':'The app crashes whenever I open it. What should I try?',
                        'Password reset':'I forgot my Apple Account password. How can I reset it?'},
        'Uber_Support':{'Cancellation fee':'I was charged a cancellation fee. How can I ask for a review?',
                        'Ride safety':'An unsafe driver injured me during my ride. I need help.'},
        'SpotifyCares':{'Charged twice':'I can see two completed Spotify charges this month. Why was I charged twice?',
                        'Cancel Premium':'How can I cancel my Spotify Premium subscription?'},
        'Delta':{'Missing baggage':'My checked bag did not arrive. How can I report it?',
                 'Change a flight':'I need to change my flight. Where can I check my options?'}}
    examples=examples_by_brand[brand]
    sample,load=st.columns([3,1],vertical_alignment='bottom')
    choice=sample.selectbox('Try an example',list(examples),key='example_'+brand)
    composer_key='composer_'+brand
    if composer_key not in st.session_state:
        st.session_state[composer_key]=next(iter(examples.values()))
    if load.button('Use example',width='stretch'):
        st.session_state[composer_key]=examples[choice]
    text=st.text_area('Customer message',height=160,key=composer_key,
                      placeholder='Paste the customer message here. Leave out passwords and personal information.')
    with st.expander('Customer details and reply handling'):
        vip=st.checkbox('Known VIP customer (operator confirmed)',key='vip_'+brand,
                        help='Use a verified support/CRM record. A claim inside a tweet does not establish VIP status.')
        auto_send=st.checkbox('Simulate auto-send for replies that pass all checks',value=True,
                             help='Delivers to a local simulated outbox. It does not publish to Twitter/X.')
    with st.expander('How Support Copilot handles a tweet'):
        st.write('Triage → Route → Retrieve five same-brand cases → Check FAQ/policy → Draft → Critique → Deliver or review')
        st.caption('Spam/off-topic is ignored. Legal threats, safety concerns, account compromise, and operator-confirmed VIPs go to a human. A failed draft can be revised twice. Low confidence also requires review.')
    if policy_context=='historical_replay':
        st.info('Historical replay is on. Current policy notes are excluded, so more cases may need human review.')
    if st.button('Prepare reply',type='primary',disabled=index is None or not text.strip()):
        if not settings.api_key:
            credential_dialog()
        else:
            with st.status('Triaging the issue and gathering evidence…',expanded=True) as progress:
                client=ProviderClient(settings)
                try:
                    st.session_state['last_result']=SupportEngine(client,retriever,store).run(brand,text,policy_context=policy_context,
                        customer_vip=vip,auto_send=auto_send,
                        on_step=lambda step:st.write(step['stage'].replace('_',' ').title()+' — '+step['summary']))
                    progress.update(label='Reply workflow finished',state='complete',expanded=False)
                    if hasattr(client,'http'):
                        client.http.close()
                except ValueError as error:
                    st.error(str(error))
    if st.session_state.get('last_result'):
        display_result(st.session_state['last_result'])

with queue_tab:
    st.subheader('Human review')
    pending=store.list_runs('pending')
    st.caption(f'{len(pending)} pending records. Approvals update a local queue; the prototype has no external posting operation.')
    if pending:
        selected=st.selectbox('Review record',[r['run_id'] for r in pending],format_func=lambda x:next(r['brand']+' · '+r['review_reason'][:90] for r in pending if r['run_id']==x))
        record=next(r for r in pending if r['run_id']==selected)
        payload=json.loads(record['payload'])
        st.write(payload['customer_text'])
        edited=st.text_area('Review or edit the reply',record['reply'],key='edit_'+selected)
        reviewer=st.text_input('Reviewer name')
        note=st.text_input('Review note')
        approve,reject=st.columns(2)
        decision=None
        if approve.button('Approve edited reply',type='primary'):
            decision='approved'
        if reject.button('Reject reply'):
            decision='rejected'
        if decision:
            try:
                store.review(selected,decision,reviewer,note,edited)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    else:
        st.write('No cases are waiting for review.')
    with st.expander('Simulated auto-send outbox'):
        deliveries=store.list_runs('simulated_sent')
        st.caption('Local delivery simulation for the class demonstration. No public tweets are posted.')
        if deliveries:
            st.dataframe(pd.DataFrame([{k:r[k] for k in ['created_at','brand','reply','status']} for r in deliveries]),hide_index=True,width='stretch')
        else:
            st.write('No replies have been delivered to the simulated outbox yet.')
    with st.expander('Recent run history'):
        history=store.list_runs(limit=50)
        st.dataframe(pd.DataFrame([{k:r[k] for k in ['created_at','brand','route','status','review_reason']} for r in history]),hide_index=True,width='stretch')

with eval_tab:
    st.subheader('Baseline, RAG, and full-agent comparison')
    st.caption('Scores are model-judged until validated by human raters. Historical replies are references, not definitive policy answers.')
    if index:
        count=st.select_slider('Cases to evaluate',options=[5,10,25,50,100,200],value=5)
        split=st.selectbox('Evaluation split',['development','test'])
        if st.button('Run comparison'):
            if not settings.api_key:
                st.error('Configure a live model key first.')
            elif split=='development' and count>100:
                st.error('The development set contains 100 cases.')
            else:
                bar=st.progress(0,text='Starting comparison')
                try:
                    directory,summary=evaluate(settings,retriever,count,4,split,policy_context,
                        progress=lambda n,total,status:bar.progress(n/total,text=f'{n}/{total} cases · {status}'))
                    st.success(f"Completed {summary['cases_complete']} cases. Saved to {directory.name}.")
                except Exception as error:
                    st.error(f'Evaluation failed: {type(error).__name__}')
    directories=sorted((workspace/'evaluations').glob('*/summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if (APP_ROOT/'reports/heldout_benchmark/summary.json').exists():
        directories.append(APP_ROOT/'reports/heldout_benchmark/summary.json')
    if directories:
        report=st.selectbox('Saved comparison',directories,format_func=lambda p:p.parent.name)
        summary=json.loads(report.read_text(encoding='utf-8'))
        if report.parent.name=='heldout_benchmark':
            st.caption('Reference benchmark from the earlier workflow revision. Rerun the comparison to measure the updated VIP, evidence, confidence, and delivery logic.')
        frame=pd.DataFrame(summary['systems'])
        st.bar_chart(frame.set_index('system')[['helpfulness','factual_support','tone']],stack=False,
                     y_label='Mean judge score (1–5)')
        st.dataframe(frame,hide_index=True,width='stretch')
        st.caption(summary['scoring']+' '+summary['ci_method'])
        if (report.parent/'bertscore.json').exists():
            with st.expander('BERTScore reference similarity'):
                st.json(json.loads((report.parent/'bertscore.json').read_text(encoding='utf-8')))
        st.download_button('Download comparison CSV',(report.parent/'summary.csv').read_bytes(),file_name='support_comparison.csv')
    else:
        st.write('No comparison has been run yet.')
    with st.expander('Label 100 held-out tweets for triage evaluation'):
        st.caption('Freeze the system before labeling test cases. Keep labels outside the retrieval corpus.')
        label_file=workspace/'human_labels.json'
        source_labels=read_jsonl(settings.data_dir/'triage_annotation_template.jsonl')
        uploaded=st.file_uploader('Import saved labels or a rater assignment',type=['json'],key='label_import')
        if st.button('Load label file',disabled=uploaded is None):
            try:
                imported=validate_labels(json.loads(uploaded.getvalue()),source_labels)
                label_file.write_text(json.dumps(imported,indent=2),encoding='utf-8')
                st.session_state.pop('labels',None)
                st.success(f'Loaded {len(imported)} cases. Source tweets were preserved.')
            except (ValueError,UnicodeError):
                st.error('Use a valid assignment or label JSON file with known case IDs and allowed label values.')
        labels=json.loads(label_file.read_text(encoding='utf-8')) if label_file.exists() else read_jsonl(settings.data_dir/'triage_annotation_template.jsonl')
        with st.expander('Labeling rubric'):
            st.write('Intent: label the primary requested help. Refund covers billing and refunds; delay covers delivery, travel or baggage delays; account covers access/profile issues; bug covers broken functionality; praise covers compliments; other covers remaining issues.')
            st.write('Urgency: high for immediate safety/security risk or an explicit time-critical deadline; medium for a current unresolved service problem; low for general questions or praise. Anger alone does not establish urgency.')
            st.write('Human review: true for safety/legal/compromise risks, an actual private-account decision, or insufficient policy for safe handling. Public FAQ guidance and safe clarification can be false. Do not infer VIP status from a tweet.')
            st.caption('Use your own judgment. Labels are not prefilled by a model. Each annotator should work independently before resolving disagreements.')
        editor=st.data_editor(pd.DataFrame(labels),hide_index=True,disabled=['case_id','brand','customer_text'],
            column_config={'intent':st.column_config.SelectboxColumn(options=['','refund','delay','account','bug','praise','other']),
                           'urgency':st.column_config.SelectboxColumn(options=['','low','med','high']),
                           'needs_human':st.column_config.SelectboxColumn(options=['','true','false'])},key='labels')
        if st.button('Save human labels'):
            entries=editor.fillna('').to_dict(orient='records')
            temporary=label_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(entries,indent=2),encoding='utf-8')
            temporary.replace(label_file)
            st.success('Labels saved locally.')
        working=editor.fillna('').to_dict(orient='records')
        completed=sum(all(str(r.get(k,'')).strip() for k in ['intent','urgency','needs_human']) for r in working)
        st.caption(f'{completed}/{len(working)} cases have all three labels. Export your work before closing or reloading this browser session.')
        st.download_button('Export human labels',json.dumps(editor.fillna('').to_dict(orient='records'),indent=2),
                           file_name='support_human_labels.json',mime='application/json')
        if directories and label_file.exists():
            st.json(triage_metrics(report.parent,json.loads(label_file.read_text(encoding='utf-8'))))

with ops_tab:
    st.subheader('Support operations')
    st.caption('Your current session’s operational runs. Controlled safety tests and benchmark cases are kept separate.')
    records=[json.loads(r['payload']) for r in store.list_runs(limit=5000)]
    live=[r for r in records if r['execution_mode']!='offline_demo']
    a,b,c,d=st.columns(4)
    a.metric('Messages processed',len(records))
    b.metric('Waiting for review',len(store.list_runs('pending',limit=5000)))
    c.metric('Ready / simulated replies',sum(r['route'] in ['ready_to_send','auto_sent'] for r in records))
    d.metric('Live LLM runs',len(live))
    if records:
        operational=pd.DataFrame([{'brand':r['brand'],'intent':r.get('triage',{}).get('intent','unavailable'),
            'route':r['route'],'seconds':r['latency_ms']/1000,'rewrites':r.get('rewrite_count',0),
            'estimated_cost_usd':r['usage'].get('estimated_cost_usd'),
            'review_reason':r.get('review_reason','')} for r in records])
        left,right=st.columns(2)
        left.bar_chart(operational['intent'].value_counts())
        right.bar_chart(operational['route'].value_counts())
        st.dataframe(operational,hide_index=True,width='stretch')
    st.subheader('Business case calculator')
    st.caption('A what-if estimate, not measured labor savings. Enter assumptions and validate them with a timed human study.')
    left,right=st.columns(2)
    volume=left.number_input('Monthly support messages',min_value=1,value=10000,step=100)
    manual=left.number_input('Current handling minutes per message',min_value=0.0,value=5.0,step=0.5)
    rate=left.number_input('Loaded labor cost USD per hour',min_value=0.0,value=30.0,step=1.0)
    review=right.number_input('Minutes per escalated review',min_value=0.0,value=4.0,step=0.5)
    spotcheck=right.number_input('Minutes per routine reply check',min_value=0.0,value=1.0,step=0.5)
    measured_fraction=sum(r['route']=='human_review' for r in live)/len(live) if live else None
    fraction=right.slider('Assumed share requiring review',0.0,1.0,round(measured_fraction,2) if measured_fraction is not None else 1.0,0.01)
    known_costs=[r['usage']['estimated_cost_usd'] for r in live if r['usage'].get('estimated_cost_usd') is not None]
    unit_cost=right.number_input('Estimated API USD per message',min_value=0.0,value=sum(known_costs)/len(known_costs) if known_costs else 0.005,format='%.5f',step=0.001)
    scenario=roi_scenario(volume,manual,review,spotcheck,rate,fraction,unit_cost)
    x,y,z=st.columns(3)
    x.metric('Modeled hours saved / month',f"{scenario['hours_saved']:,.1f}")
    y.metric('Estimated API cost / month',f"${scenario['estimated_api_cost_usd']:,.2f}")
    z.metric('Modeled net value / month',f"${scenario['modeled_net_value_usd']:,.2f}")
    st.caption('Excludes infrastructure, training, rework, quality failures and deployment costs. Negative savings are possible.')

with safety_tab:
    st.subheader('Controlled safety experiments')
    st.write('Test risk escalation, instruction injection, and a critic’s response to an intentionally unsafe draft.')
    st.caption('These are synthetic challenge cases, reported separately from the real held-out evaluation. Passing them is not a general security guarantee.')
    selected_tests=st.multiselect('Safety cases',[c['id'] for c in SAFETY_CASES],default=[c['id'] for c in SAFETY_CASES])
    if st.button('Run selected safety tests',disabled=index is None):
        if not settings.api_key:
            st.error('Connect your API key to run live tests.')
        else:
            results=[]
            bar=st.progress(0,text='Starting controlled tests')
            for test in [c for c in SAFETY_CASES if c['id'] in selected_tests]:
                results.append(run_safety_case(settings,retriever,test))
                bar.progress(len(results)/max(1,len(selected_tests)),text=test['id'])
            st.session_state['safety_results']=results
    safety_results=st.session_state.get('safety_results',[])
    if safety_results:
        st.dataframe(pd.DataFrame([{'test':r['test_id'],'passed':r['passed'],'route':r['result']['route'],
            'drafts':len(r['result'].get('draft_history',[])),'rewrites':r['result'].get('rewrite_count',0)} for r in safety_results]),hide_index=True,width='stretch')
        for test in safety_results:
            with st.expander(test['test_id']+' — inspect evidence'):
                st.json(test)
        st.download_button('Download safety report',json.dumps(safety_results,indent=2),file_name='controlled_safety_report.json')
    with st.expander('Check a draft without a model call'):
        candidate=st.text_area('Draft to check',key='safety_candidate',placeholder='Paste a reply to check length, private-data requests, and unsupported promises.')
        if st.button('Run deterministic checks'):
            findings=validate_reply(candidate,[],[],[],[])
            st.write(findings if findings else 'No rule violations detected. A human must still check factual accuracy and relevance.')

with knowledge_tab:
    st.subheader('Policy knowledge and evidence quality')
    with st.expander('Advanced settings'):
        st.selectbox('Policy context',['Current support simulation','Historical replay'],key='policy_context_choice')
        st.caption('Current policy notes were checked November 20, 2026. Historical replay excludes them because their 2017 effective dates are unverified.')
        rates_left,rates_right=st.columns(2)
        rates_left.number_input('Input USD per million tokens',min_value=0.0,value=0.40,step=0.01,key='input_token_price')
        rates_right.number_input('Output USD per million tokens',min_value=0.0,value=1.60,step=0.01,key='output_token_price')
        st.caption('These rates are used for cost estimates. They are not a billing statement.')
    policy_data=json.loads((APP_ROOT/'policies/verified_notes.json').read_text(encoding='utf-8'))
    st.caption(policy_data['scope']+' '+policy_data['temporal_limit'])
    kb_brand=st.selectbox('Knowledge base brand',BRANDS)
    documents=[p for p in policy_data['documents'] if p['brand']==kb_brand]
    for document in documents:
        with st.expander(document['topic'],expanded=True):
            st.write(document['text'])
            st.markdown('[Official policy source]('+document['source_url']+')')
            st.caption(document['policy_id']+' · checked '+document['retrieved_at']+' · historical effective date unknown')
    question=st.text_input('Try a policy search',placeholder='Example: I was charged twice')
    if st.button('Search brand policy',disabled=index is None):
        st.session_state['policy_search_results']={'brand':kb_brand,'context':policy_context,
            'results':retriever.search_policy(kb_brand,question,policy_context)}
    if 'policy_search_results' in st.session_state:
        search=st.session_state['policy_search_results']
        if search['brand']==kb_brand and search['context']==policy_context:
            st.json(search['results'])
    st.info('Missing or irrelevant evidence should lead to clarification or human review. Historical customer replies cannot establish a current policy or private account fact.')
    with st.expander('Dataset quality and split audit'):
        audit=json.loads((APP_ROOT/'data_audit.json').read_text(encoding='utf-8'))
        st.dataframe(pd.DataFrame([{'Set':k,'Pairs':v} for k,v in audit['split_counts'].items()]),hide_index=True)
        st.write(audit['split_checks'])
        st.caption('Source data dates mostly to 2017. Conversation groups and exact normalized inputs are separated across splits; semantic similarity is audited separately.')

