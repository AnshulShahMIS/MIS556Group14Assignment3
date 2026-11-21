import time
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import BRANDS
from .privacy import redact, risk_signals, validate_reply
from .schemas import Critique, Draft, Triage
from .provider import ProviderError


class SupportState(TypedDict, total=False):
    run_id: str
    brand: str
    customer_text: str
    policy_context: str
    mode: str
    triage: dict
    cases: list[dict]
    policies: list[dict]
    policy_searched: bool
    searches: int
    research_done: bool
    draft: dict
    draft_history: list[dict]
    critic_history: list[dict]
    rewrite_count: int
    critic: dict
    trace: list[dict]
    review_reason: str
    route: str
    final_reply: str
    inject_bad_draft: bool


def event(state,stage,summary,**details):
    return state.get('trace',[]) + [{'stage':stage,'summary':summary,**details}]


class SupportEngine:
    def __init__(self,client,retriever,store=None,max_searches=4,max_rewrites=2):
        self.client = client
        self.retriever = retriever
        self.store = store
        self.max_searches = max_searches
        self.max_rewrites = max_rewrites
        graph = StateGraph(SupportState)
        for name in ['triage','research','draft','critic','review','finalize']:
            graph.add_node(name,getattr(self,name))
        graph.add_edge(START,'triage')
        graph.add_conditional_edges('triage',self.after_triage,{'research':'research','review':'review','ignored':'finalize'})
        graph.add_conditional_edges('research',lambda s:'draft' if s.get('research_done') else 'research')
        graph.add_edge('draft','critic')
        graph.add_conditional_edges('critic',self.after_critic,{'draft':'draft','review':'review','finalize':'finalize'})
        graph.add_edge('review','finalize')
        graph.add_edge('finalize',END)
        self.graph = graph.compile()

    def triage(self,state):
        result = self.client.json('triage',
            'Classify the support issue. High-risk concerns include credible legal threats, physical safety, account compromise and privacy exposure. '
            'Anger alone is not a safety emergency. Off-topic means clearly unrelated to customer support, not merely negative. '
            'Risk flags must be short category names and reason must be a brief evidence summary. Brand is trusted application metadata.',
            {'brand':state['brand'],'customer_text':state['customer_text']},Triage)
        hard_flags = risk_signals(state['customer_text'])
        if hard_flags:
            result['needs_human'] = True
            result['risk_flags'] = list(dict.fromkeys(result['risk_flags']+hard_flags))
        return {'triage':result,'trace':event(state,'triage',result['reason'],result=result)}

    def after_triage(self,state):
        if state['triage']['needs_human']:
            return 'review'
        if state['triage']['off_topic']:
            return 'ignored'
        return 'research'

    def research(self,state):
        if state['searches'] >= self.max_searches:
            return {'research_done':True,'trace':event(state,'research','Search budget reached; continue with available evidence.')}
        action = self.client.tool({'brand':state['brand'],'customer_text':state['customer_text'],
            'triage':state['triage'],'cases':state['cases'],'policies':state['policies'],
            'policy_searched':state['policy_searched'],'searches_used':state['searches']})
        name, arguments = action['name'], action.get('arguments',{})
        if name == 'finish_research':
            return {'research_done':True,'trace':event(state,'research',str(arguments.get('reason','Research complete')),tool=name)}
        query = str(arguments.get('query','')).strip()[:1500]
        if not query:
            query = state['customer_text']
        try:
            if name == 'search_cases':
                results = self.retriever.search_cases(state['brand'],query,5)
                return {'cases':results,'searches':state['searches']+1,
                        'trace':event(state,'retrieval',f'Retrieved {len(results)} same-brand cases.',tool=name,query=query,results=results)}
            if name == 'search_policy':
                results = self.retriever.search_policy(state['brand'],query,state['policy_context'])
                return {'policies':results,'policy_searched':True,'searches':state['searches']+1,
                        'trace':event(state,'policy',f'Retrieved {len(results)} source-verified policy notes.',tool=name,query=query,results=results)}
            raise ValueError('Unrecognized evidence tool')
        except (ValueError,RuntimeError,OSError) as error:
            # Failed attempts consume budget, making recovery bounded.
            return {'searches':state['searches']+1,
                    'trace':event(state,'tool_error',f'{name} failed: {type(error).__name__}. The agent can retry within its search budget.',tool=name)}

    def draft(self,state):
        feedback = state.get('critic',{})
        result = self.client.json('draft',
            'Write a concise empathetic public support reply under 280 Unicode characters in the selected brand voice. '
            'Historical replies are style examples, not current policy or private account facts. Use only relevant policy facts from the supplied notes. '
            'Never claim to have accessed an account, issued a refund, changed a booking, fixed an outage, or completed an action. '
            'Do not guarantee outcomes or request passwords, verification codes or payment secrets in any channel. '
            'Keep other personal information private. Use no placeholder tokens or copied historical URLs. '
            'Cite supporting policy_ids and case_ids separately from reply text. If evidence is insufficient, ask a safe clarification and state uncertainty internally. '
            'Apply specific critic feedback when present.',
            {'brand':state['brand'],'customer_text':state['customer_text'],'triage':state['triage'],
             'cases':state['cases'],'policies':state['policies'],'policy_context':state['policy_context'],
             'critic_feedback':feedback,'inject_bad_draft':False},Draft)
        controlled = state.get('inject_bad_draft') and not state['draft_history']
        if controlled:
            result = {**result,'reply':"I've refunded your payment and guarantee delivery tomorrow.",
                      'uncertainty':'Deliberately seeded draft for a controlled guardrail test'}
        return {'draft':result,'draft_history':state['draft_history']+[result],
                'rewrite_count':max(0,len(state['draft_history'])),
                'trace':event(state,'draft','Controlled unsafe draft seeded.' if controlled else 'Prepared a reply draft.',draft=result,controlled_test=bool(controlled))}

    def critic(self,state):
        draft = state['draft']
        issues = validate_reply(draft['reply'],draft['policy_ids'],draft['case_ids'],state['policies'],state['cases'])
        result = self.client.json('critic',
            'Review the reply against the customer issue and provided policy. Historical replies are not authoritative policy. '
            'Check helpfulness, tone, relevance, factual support, unsupported promises, credentials, privacy and length. '
            'Any deterministic issue makes passed=false. supported means factual claims are supported or the reply is a safe clarification with no unverified claims. '
            'Set needs_human for unresolved risk or a decision requiring account access. Do not mistake a polite request for future assistance for an action already completed.',
            {'brand':state['brand'],'customer_text':state['customer_text'],'draft':draft,
             'policies':state['policies'],'deterministic_issues':issues},Critique)
        result['issues'] = list(dict.fromkeys(issues+result['issues']))
        result['passed'] = bool(result['passed'] and not result['issues'] and result['supported'])
        if not state['policies'] and state['triage']['intent'] not in ['praise']:
            result['needs_human'] = True
            result['reason'] = 'No verified policy evidence is available for this issue and policy context.'
        return {'critic':result,'critic_history':state['critic_history']+[result],
                'trace':event(state,'critic',result['reason'],result=result)}

    def after_critic(self,state):
        result = state['critic']
        if result['needs_human']:
            return 'review'
        if result['passed']:
            return 'finalize'
        return 'draft' if state['rewrite_count'] < self.max_rewrites else 'review'

    def review(self,state):
        reason = state.get('review_reason') or state.get('critic',{}).get('reason') or state.get('triage',{}).get('reason') or 'Human judgment required.'
        return {'route':'human_review','review_reason':reason,
                'final_reply':state.get('draft',{}).get('reply') or 'Thanks for flagging this. Please contact support through the official app or website so your concern can be reviewed securely.',
                'trace':event(state,'human_review',reason)}

    def finalize(self,state):
        if state.get('route') == 'human_review':
            return {'trace':event(state,'finalize','Saved for human review.')}
        if state['triage']['off_topic']:
            return {'route':'ignored','final_reply':'','review_reason':state['triage']['reason'],
                    'trace':event(state,'finalize','Marked off-topic with an audit record.')}
        return {'route':'ready_to_send','final_reply':state['draft']['reply'],
                'review_reason':'','trace':event(state,'finalize','Reply passed checks and entered the local ready queue.')}

    def run(self,brand,text,*,case_id=None,policy_context='current_simulation',inject_bad_draft=False,on_step=None):
        if brand not in BRANDS or not text.strip():
            raise ValueError('Choose a supported brand and enter a customer message.')
        if len(text)>6000:
            raise ValueError('Use a customer message within 6,000 characters.')
        start = time.perf_counter()
        state = {'run_id':uuid.uuid4().hex,'brand':brand,'customer_text':redact(text),
                 'policy_context':policy_context,'mode':'full_agent','cases':[],'policies':[],
                 'policy_searched':False,'searches':0,'research_done':False,'draft_history':[],
                 'critic_history':[],'rewrite_count':0,'trace':[],'inject_bad_draft':inject_bad_draft}
        try:
            for update in self.graph.stream(state,stream_mode='updates',config={'recursion_limit':32}):
                for changes in update.values():
                    state.update(changes)
                    if on_step and changes.get('trace'):
                        on_step(changes['trace'][-1])
        except Exception as error:
            # Persist partial state for failures, without request headers or keys.
            detail = str(error) if isinstance(error,ProviderError) else type(error).__name__
            state['review_reason'] = f'Execution needs review: {detail}'
            state['trace'] = event(state,'execution_error',state['review_reason'])
            state.update(self.review(state))
        checks = {'policy_available':bool(state.get('policies')),
                  'critic_passed':bool(state.get('critic',{}).get('passed')),
                  'deterministic_checks_passed':bool(state.get('draft')) and not validate_reply(
                      state.get('final_reply',''),state.get('draft',{}).get('policy_ids',[]),
                      state.get('draft',{}).get('case_ids',[]),state.get('policies',[]),state.get('cases',[]))}
        state.update({'evidence_checks':checks,'evidence_check_score':sum(checks.values()),
                      'confidence_note':'Checklist score out of 3, not a calibrated probability of correctness.',
                      'case_id':case_id,'latency_ms':round((time.perf_counter()-start)*1000),
                      'usage':self.client.usage(),'execution_mode':self.client.usage()['provider'],
                      'evidence_status':'limited' if state['route']=='human_review' else 'checked',
                      'external_delivery':'No external delivery operation exists in this local prototype.'})
        if self.store:
            self.store.save(state)
        return state


def run_baseline(client,retriever,brand,text,mode):
    start = time.perf_counter()
    cases = retriever.search_cases(brand,text,5) if mode=='rag_only' else []
    result = client.json('draft',
        'Draft a helpful public support reply as the selected brand, under 280 characters. '
        'Protect private information. Do not request passwords or claim actions were completed. '
        'Use any supplied historical cases as examples. Return JSON including reply, policy_ids, case_ids and uncertainty.',
        {'brand':brand,'customer_text':redact(text),'cases':cases,'policies':[]},Draft)
    return {'brand':brand,'mode':mode,'route':'comparison_only','final_reply':result['reply'],
            'draft':result,'cases':cases,'policies':[],
            'deterministic_issues':validate_reply(result['reply'],result['policy_ids'],result['case_ids'],[],cases),
            'latency_ms':round((time.perf_counter()-start)*1000),'usage':client.usage()}
