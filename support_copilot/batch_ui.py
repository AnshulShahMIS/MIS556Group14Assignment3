import pandas as pd
import streamlit as st

from .batch import SAMPLE,parse_inbox,new_inbox,process_next,inbox_results,export_csv,spreadsheet_safe
from .engine import SupportEngine
from .provider import ProviderClient


def render_batch_inbox(settings,retriever,store,policy_context,index_ready):
    st.subheader('Batch support inbox')
    st.write('Upload support messages, review intake priorities, and prepare replies in small batches.')
    st.caption('CSV: brand and message are required; message_id and customer_vip are optional. '
               'Use Amazon, Apple, Uber, Spotify, or Delta. VIP values must be true or false and operator confirmed. '
               'Maximum 50 rows and 1 MB. Use synthetic messages; redaction cannot catch every personal detail.')
    st.download_button('Download sample inbox CSV',SAMPLE.encode(),file_name='sample_support_inbox.csv',mime='text/csv')
    upload=st.file_uploader('Upload support inbox CSV',type=['csv'],key='batch_upload',max_upload_size=1)
    load,sample=st.columns(2)
    uploaded=load.button('Load uploaded inbox',disabled=upload is None)
    sampled=sample.button('Try sample inbox')
    st.caption('Loading a different inbox replaces the active batch view. Existing run records remain in this session’s review queue and history.')
    if uploaded or sampled:
        try:
            rows,errors=parse_inbox(upload.getvalue() if uploaded else SAMPLE.encode())
            incoming=new_inbox(rows,errors,policy_context)
            current=st.session_state.get('batch_inbox')
            if current and current['fingerprint']==incoming['fingerprint'] and current['context']==policy_context:
                st.info('This inbox is already loaded. Existing progress was kept; completed messages will not run again.')
            else:
                st.session_state['batch_inbox']=incoming
        except ValueError as error:
            st.error(str(error))
    inbox=st.session_state.get('batch_inbox')
    if not inbox:
        st.info('Start with the sample inbox to preview the workflow without using an API key.')
        return
    st.caption('Batch '+inbox['batch_id'][:8]+' · Policy context fixed at import: '+inbox['context'].replace('_',' '))
    if inbox['errors']:
        st.warning(f"{len(inbox['errors'])} invalid row(s) excluded. Correct them in your source CSV before importing again.")
        st.dataframe(pd.DataFrame(inbox['errors']),hide_index=True,width='stretch')
        st.download_button('Download validation errors',export_csv(inbox['errors']),file_name='inbox_validation_errors.csv',mime='text/csv')
    if not inbox['rows']:
        st.info('No valid messages are available to process.')
        return
    waiting=sum(str(r['row']) not in inbox['claimed'] for r in inbox['rows'])
    preview=inbox_results(inbox,store)
    total,remaining,review=st.columns(3)
    total.metric('Valid messages',len(preview))
    remaining.metric('Waiting',waiting)
    review.metric('Needs human review',sum(r['status']=='pending' for r in preview))
    st.caption('Intake priority is a rule-based flag, not the model’s urgency classification. '
               'High-priority messages run first. All replies are drafts; nothing is posted or auto-sent.')
    st.dataframe(pd.DataFrame(preview)[['message_id','brand','message','priority','priority_reason','status']].map(spreadsheet_safe),hide_index=True,width='stretch')
    size=st.select_slider('Messages per processing action',options=[1,3,5,10],value=5)
    st.caption('Processing uses your OpenAI key and incurs API charges. Each message can make several model calls. '
               'Errors stop the current action; completed or interrupted requests are not automatically repeated.')
    if not settings.api_key:
        st.info('Connect your API key using the top-right button to process the inbox. Preview and CSV downloads work without one.')
    if st.button(f'Process next {min(size,waiting)} messages',type='primary',
                 disabled=not waiting or not index_ready or not settings.api_key):
        progress=st.progress(0,text='Processing high-priority messages first…')
        expected=min(size,waiting)
        process_next(inbox,lambda:SupportEngine(ProviderClient(settings),retriever,store),size,
                     lambda n:progress.progress(n/expected,text=f'Processed {n} of up to {expected} messages'))
        st.rerun()
    results=inbox_results(inbox,store)
    if inbox['claimed']:
        st.subheader('Batch results')
        if any(r['execution_error'] or r['status']=='interrupted' for r in results):
            st.warning('Some requests failed or were interrupted. Check your key/quota and the Review queue or run history before processing the remaining messages. Failed requests are not automatically retried.')
        status=st.selectbox('Filter batch results',['All','Needs human review','Ready','Approved','Rejected','Waiting','Errors / interrupted'])
        statuses={'Needs human review':{'pending'},'Ready':{'ready'},'Approved':{'approved'},
                  'Rejected':{'rejected'},'Waiting':{'waiting'}}
        shown=[r for r in results if status=='All' or r['status'] in statuses.get(status,set())
               or status=='Errors / interrupted' and (r['execution_error'] or r['status']=='interrupted')]
        st.dataframe(pd.DataFrame(shown).map(spreadsheet_safe),hide_index=True,width='stretch')
        st.caption('Approve or reject flagged cases in Review queue. Refresh this view after a review to include the latest edited reply and decision in your export.')
        st.button('Refresh review decisions',key='batch_refresh')
        st.download_button('Download batch results CSV',export_csv(results),file_name='support_inbox_results.csv',mime='text/csv')
        selected=st.selectbox('Inspect processed message',['']+[r['run_id'] for r in results if r['run_id']],
            format_func=lambda key:'Choose a message' if not key else next(r['message_id']+' · '+r['brand'] for r in results if r['run_id']==key))
        if selected:
            record=next(r for r in store.list_runs(limit=10000) if r['run_id']==selected)
            import json
            payload=json.loads(record['payload'])
            st.write(record['reply'])
            with st.expander('Agent trace and evidence'):
                st.json({'triage':payload.get('triage'),'trace':payload.get('trace'),
                         'cases':payload.get('cases'),'policies':payload.get('policies')})
    else:
        st.download_button('Download validated inbox',export_csv(results),file_name='validated_support_inbox.csv',mime='text/csv')
