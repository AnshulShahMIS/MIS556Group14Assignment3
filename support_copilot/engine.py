import time
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import BRANDS
from .privacy import redact, risk_signals, validate_reply
from .schemas import Critique, Draft, Triage, PolicyAssessment
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
    customer_vip: bool
    cases_searched: bool
    route_decision: str
    auto_send: bool
    confidence_threshold: int
    policy_assessment: dict


def event(state,stage,summary,**details):
    return state.get('trace',[]) + [{'stage':stage,'summary':summary,**details}]


def reply_confidence(state):
    """Transparent readiness heuristic; never a calibrated correctness probability."""
    draft=state.get('draft',{})
    critic=state.get('critic',{})
    policies=state.get('policies',[])
    cases=state.get('cases',[])
    checks={
        'same_brand_cases':bool(cases) and all(c.get('brand')==state['brand'] for c in cases),
        'cited_policy':bool(policies) and bool(draft.get('policy_ids')) and
                       set(draft.get('policy_ids',[])).issubset({p['policy_id'] for p in policies}),
        'critic_approved':bool(critic.get('passed') and critic.get('supported') and not critic.get('needs_human')),
        'deterministic_checks':bool(draft) and not validate_reply(draft.get('reply',''),
            draft.get('policy_ids',[]),draft.get('case_ids',[]),policies,cases)}
    return {'score':25*sum(checks.values()),'checks':checks,
            'method':'Four evidence/readiness checks, 25 points each. Not a probability of correctness.'}


class SupportEngine:
    def __init__(self,client,retriever,store=None,max_searches=4,max_rewrites=2):
        self.client = client
        self.retriever = retriever
        self.store = store
        self.max_searches = max_searches
        self.max_rewrites = max_rewrites
        graph = StateGraph(SupportState)
        for name in ['triage','router','research','policy_check','draft','critic','review','finalize']:
            graph.add_node(name,getattr(self,name))
        graph.add_edge(START,'triage')
        graph.add_edge('triage','router')
        graph.add_conditional_edges('router',lambda s:s['route_decision'],{'research':'research','review':'review','ignored':'finalize'})
        graph.add_conditional_edges('research',lambda s:'policy_check' if s.get('research_done') else 'research')
        graph.add_conditional_edges('policy_check',lambda s:'review' if s['policy_assessment']['requires_human'] else 'draft')
        graph.add_edge('draft','critic')
        graph.add_conditional_edges('critic',self.after_critic,{'draft':'draft','review':'review','finalize':'finalize'})
        graph.add_edge('review','finalize')
        graph.add_edge('finalize',END)
        self.graph = graph.compile()

    def triage(self,state):
        result = self.client.json('triage',
            'Classify the support issue. High-risk concerns include credible legal threats, physical safety, account compromise and privacy exposure. '
            'Anger alone is not a safety emergency. Identify unsolicited advertising as spam. Off-topic means clearly unrelated to customer support, not merely negative. '
            'Risk flags must be short category names and reason must be a brief evidence summary. Brand is trusted application metadata.',
            {'brand':state['brand'],'customer_text':state['customer_text']},Triage)
        hard_flags = risk_signals(state['customer_text'])
        if hard_flags:
            result['needs_human'] = True
            result['risk_flags'] = list(dict.fromkeys(result['risk_flags']+hard_flags))
        # Brand and VIP status come from the operator, never from a tweet's claims.
        result['brand']=state['brand']
        result['vip']=state.get('customer_vip',False)
        if result['vip']:
            result['needs_human']=True
            result['risk_flags']=list(dict.fromkeys(result['risk_flags']+['verified_vip']))
            result['reason']='Operator-marked VIP customer requires human support.'
        return {'triage':result,'trace':event(state,'triage',result['reason'],result=result)}

    def router(self,state):
        if state['triage']['needs_human']:
            decision='review'
        elif state['triage']['off_topic'] or state['triage'].get('spam'):
            decision='ignored'
        else:
            decision='research'
        return {'route_decision':decision,'trace':event(state,'router',
            {'review':'Risk/VIP flag: escalate to a human.',
             'ignored':'Spam or off-topic: no reply.',
             'research':'Support issue: retrieve cases and check policy.'}[decision],decision=decision)}

    def research(self,state):
        if state['searches'] >= self.max_searches:
            return {'research_done':True,'trace':event(state,'research','Search budget reached; continue with available evidence.')}
        action = self.client.tool({'brand':state['brand'],'customer_text':state['customer_text'],
            'triage':state['triage'],'cases':state['cases'],'policies':state['policies'],
            'policy_searched':state['policy_searched'],'searches_used':state['searches']})
        name, arguments = action['name'], action.get('arguments',{})
        missing=['search_cases'] if not state['cases_searched'] else []
        if not state['policy_searched']:
            missing.append('search_policy')
        # The model can choose query/order, but cannot skip required evidence.
        if missing and (name=='finish_research' or state['searches']==self.max_searches-1):
            name,arguments=missing[0],{'query':state['customer_text']}
        if name == 'finish_research':
            return {'research_done':True,'trace':event(state,'research',str(arguments.get('reason','Research complete')),tool=name)}
        query = str(arguments.get('query','')).strip()[:1500]
        if not query:
            query = state['customer_text']
        try:
            if name == 'search_cases':
                results = self.retriever.search_cases(state['brand'],query,5)
                return {'cases':results,'cases_searched':True,'searches':state['searches']+1,
                        'trace':event(state,'retrieval',f'Retrieved {len(results)} same-brand cases.',tool=name,query=query,results=results)}
            if name == 'search_policy':
                results = self.retriever.search_policy(state['brand'],query,state['policy_context'])
                return {'policies':results,'policy_searched':True,'searches':state['searches']+1,
                        'trace':event(state,'policy',f'Retrieved {len(results)} source-verified policy notes.',tool=name,query=query,results=results)}
            raise ValueError('Unrecognized evidence tool')
        except (ValueError,RuntimeError,OSError) as error:
            # Failed attempts consume budget, making recovery bounded.
            return {'searches':state['searches']+1,
                    'cases_searched':state['cases_searched'] or name=='search_cases',
                    'policy_searched':state['policy_searched'] or name=='search_policy',
                    'trace':event(state,'tool_error',f'{name} failed: {type(error).__name__}. The agent can retry within its search budget.',tool=name)}

    def policy_check(self,state):
        if not state['policies']:
            assessment={'relevant_policy_ids':[],'evidence_sufficient':False,'requires_human':True,
                        'reason':'No verified policy is available for this issue and policy context.'}
        else:
            assessment=self.client.json('policy_check',
                'Check which supplied official FAQ/policy notes apply to this customer issue. '
                'Return only relevant retrieved policy IDs. Historical cases are not policy. '
                'Decide whether public guidance or a safe clarification is supported. '
                'Do not infer private account facts, a refund entitlement, or a completed action. '
                'Require a human when the available notes do not support safe handling or an actual account decision is necessary. '
                'Public password-reset or troubleshooting guidance alone does not require private account access.',
                {'brand':state['brand'],'customer_text':state['customer_text'],
                 'triage':state['triage'],'policies':state['policies'],'policy_context':state['policy_context']},PolicyAssessment)
        known={p['policy_id'] for p in state['policies']}
        if not set(assessment['relevant_policy_ids']).issubset(known):
            assessment.update(requires_human=True,evidence_sufficient=False,
                              reason='Policy assessment cited a document that was not retrieved.')
        if not assessment['evidence_sufficient']:
            assessment['requires_human']=True
        selected=[p for p in state['policies'] if p['policy_id'] in assessment['relevant_policy_ids']]
        changes={'policies':selected,'policy_assessment':assessment,
                 'trace':event(state,'policy_check',assessment['reason'],result=assessment)}
        if assessment['requires_human']:
            changes['review_reason']=assessment['reason']
        return changes

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
             'policy_assessment':state.get('policy_assessment',{}),
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
        if state['triage']['off_topic'] or state['triage'].get('spam'):
            return {'route':'ignored','final_reply':'','review_reason':state['triage']['reason'],
                    'trace':event(state,'finalize','Marked off-topic with an audit record.')}
        confidence=reply_confidence(state)
        if confidence['score']<state['confidence_threshold']:
            reason=f"Readiness score {confidence['score']}/100 is below the {state['confidence_threshold']}-point release threshold."
            return {'route':'human_review','final_reply':state['draft']['reply'],'review_reason':reason,
                    'trace':event(state,'human_review',reason)}
        simulated=state.get('auto_send',False) and self.store is not None
        return {'route':'auto_sent' if simulated else 'ready_to_send','final_reply':state['draft']['reply'],
                'review_reason':'','trace':event(state,'finalize',
                    'Auto-send simulated in the local outbox.' if simulated else 'Reply passed checks and entered the local ready queue.')}

    def run(self,brand,text,*,case_id=None,policy_context='current_simulation',inject_bad_draft=False,on_step=None,
            customer_vip=False,auto_send=False,confidence_threshold=80):
        if brand not in BRANDS or not text.strip():
            raise ValueError('Choose a supported brand and enter a customer message.')
        if len(text)>6000:
            raise ValueError('Use a customer message within 6,000 characters.')
        if not 0<=confidence_threshold<=100:
            raise ValueError('Confidence threshold must be within 0–100.')
        start = time.perf_counter()
        state = {'run_id':uuid.uuid4().hex,'brand':brand,'customer_text':redact(text),
                 'policy_context':policy_context,'mode':'full_agent','cases':[],'policies':[],
                 'policy_searched':False,'cases_searched':False,'searches':0,'research_done':False,'draft_history':[],
                 'critic_history':[],'rewrite_count':0,'trace':[],'inject_bad_draft':inject_bad_draft}
        state.update(customer_vip=bool(customer_vip),auto_send=bool(auto_send),confidence_threshold=confidence_threshold)
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
        confidence=reply_confidence(state)
        state.update({'evidence_checks':checks,'evidence_check_score':sum(checks.values()),
                      'confidence_score':confidence['score'],'confidence_checks':confidence['checks'],
                      'confidence_note':confidence['method'],
                      'case_id':case_id,'latency_ms':round((time.perf_counter()-start)*1000),
                      'usage':self.client.usage(),'execution_mode':self.client.usage()['provider'],
                      'evidence_status':'limited' if state['route']=='human_review' else 'checked',
                      'external_delivery':'Local auto-send simulation only; no public tweets are posted.'})
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
