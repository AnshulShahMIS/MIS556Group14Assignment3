import json
from pathlib import Path

import pytest

from support_copilot.config import APP_ROOT
from support_copilot.engine import SupportEngine
from support_copilot.privacy import redact, validate_reply
from support_copilot.provider import DemoClient
from support_copilot.retrieval import read_jsonl
from support_copilot.storage import RunStore


class Evidence:
    def __init__(self, broken=False):
        self.brands = []
        self.broken = broken

    def search_cases(self, brand, query, top_k=5):
        self.brands.append(brand)
        if self.broken:
            raise RuntimeError('Unavailable')
        return [{'case_id':'case1','brand':brand,'customer_text':'My order is late',
                 'historical_reply':'Please check your tracking.'}]

    def search_policy(self, brand, query, context):
        if context == 'historical_replay':
            return []
        return [{'policy_id':'policy1','brand':brand,'text':'Check tracking.',
                 'source_url':'https://example.org/policy',
                 'demo_reply':'Sorry your order is late. Please check the tracking details in your account.'}]


def test_controlled_bad_draft_is_revised():
    result = SupportEngine(DemoClient(), Evidence()).run('AmazonHelp','My order is late',inject_bad_draft=True)
    assert result['route'] == 'ready_to_send'
    assert len(result['draft_history']) == 2
    assert not result['critic_history'][0]['passed']
    assert 'guarantee' not in result['final_reply']
    assert result['critic_history'][-1]['passed']


def test_high_risk_routes_before_retrieval():
    evidence = Evidence()
    result = SupportEngine(DemoClient(), evidence).run('AppleSupport','My account was hacked')
    assert result['route'] == 'human_review'
    assert evidence.brands == []


def test_historical_context_excludes_current_policy():
    result = SupportEngine(DemoClient(), Evidence()).run('AmazonHelp','Order is late',policy_context='historical_replay')
    assert result['policies'] == []
    assert result['route'] == 'human_review'


def test_retrieval_failure_is_bounded():
    result = SupportEngine(DemoClient(), Evidence(broken=True)).run('AmazonHelp','Order is late')
    assert result['searches'] == 4
    assert result['route'] == 'human_review'
    assert len(result['trace']) < 20


def test_rewrites_are_bounded():
    class FailingCritic(DemoClient):
        def json(self, stage, instruction, payload, schema):
            value = super().json(stage,instruction,payload,schema)
            if stage == 'critic':
                value.update(passed=False,issues=['Needs revision'])
            return value
    result = SupportEngine(FailingCritic(), Evidence()).run('AmazonHelp','Order is late')
    assert result['route'] == 'human_review'
    assert len(result['draft_history']) == 3
    assert result['rewrite_count'] == 2


def test_review_is_audited_and_cannot_be_overwritten(tmp_path):
    store = RunStore(tmp_path/'runs.sqlite')
    result = SupportEngine(DemoClient(), Evidence(),store).run('AppleSupport','My account was hacked')
    store.review(result['run_id'],'approved','Reviewer A',reply='Please use official account recovery through the support website.')
    store.save(result)
    assert store.list_runs()[0]['status'] == 'approved'
    with pytest.raises(ValueError):
        store.review(result['run_id'],'rejected','Reviewer B')
    with store.connection() as db:
        assert db.execute('SELECT count(*) FROM reviews').fetchone()[0] == 1


def test_unsafe_human_edit_cannot_be_approved(tmp_path):
    store = RunStore(tmp_path/'runs.sqlite')
    result = SupportEngine(DemoClient(), Evidence(),store).run('AppleSupport','My account was hacked')
    with pytest.raises(ValueError):
        store.review(result['run_id'],'approved','Reviewer',reply='Send us your password.')
    assert store.list_runs()[0]['status'] == 'pending'


def test_redaction_and_unsupported_citations():
    text = redact('Email jane@example.com or call 1234567890 @customer https://example.com')
    assert 'jane@example.com' not in text
    assert '1234567890' not in text
    assert 'https://' not in text
    assert validate_reply('Thanks for reaching out.',['invented'],[],[],[])


def test_conversation_splits_and_exact_queries_are_disjoint():
    sets = [read_jsonl(APP_ROOT/'data'/name) for name in
            ['index_pairs.jsonl','development_pairs.jsonl','test_pairs.jsonl']]
    assert [len(s) for s in sets] == [50000,100,200]
    for key in ['conversation_id','customer_text','case_id']:
        values = [{r[key] for r in rows} for rows in sets]
        assert not (values[0]&values[1] or values[0]&values[2] or values[1]&values[2])


def test_generation_never_receives_test_reference():
    class Spy(DemoClient):
        def json(self,stage,instruction,payload,schema):
            assert 'historical_reference' not in payload
            assert 'SECRET_HELDOUT_REPLY' not in json.dumps(payload)
            return super().json(stage,instruction,payload,schema)
    result = SupportEngine(Spy(), Evidence()).run('AmazonHelp','Order is late')
    assert result['route'] == 'ready_to_send'


def test_verified_vip_escalates_before_retrieval_and_never_auto_sends():
    evidence=Evidence()
    result=SupportEngine(DemoClient(),evidence).run('Delta','Please help with a flight.',customer_vip=True,auto_send=True)
    assert result['route']=='human_review'
    assert result['triage']['vip'] and result['triage']['brand']=='Delta'
    assert evidence.brands==[]
    assert any(s['stage']=='router' for s in result['trace'])


@pytest.mark.parametrize('flag',['spam','off_topic'])
def test_spam_and_off_topic_are_ignored_without_retrieval(flag):
    class SpamClient(DemoClient):
        def json(self,stage,instruction,payload,schema):
            value=super().json(stage,instruction,payload,schema)
            if stage=='triage':
                value[flag]=True
            return value
    evidence=Evidence()
    result=SupportEngine(SpamClient(),evidence).run('AmazonHelp','Unrelated advertisement.',auto_send=True)
    assert result['route']=='ignored' and result['final_reply']==''
    assert result['confidence_score']==0 and evidence.brands==[]


def test_model_cannot_skip_required_case_and_policy_tools():
    class EarlyFinish(DemoClient):
        def tool(self,payload):
            return {'name':'finish_research','arguments':{'reason':'No search needed'}}
    result=SupportEngine(EarlyFinish(),Evidence()).run('AmazonHelp','Order is late')
    assert result['cases_searched'] and result['policy_searched']
    assert result['searches']==2
    assert {s['stage'] for s in result['trace']} >= {'triage','router','retrieval','policy','draft','critic','finalize'}


def test_low_confidence_cannot_be_released():
    class UncitedDraft(DemoClient):
        def json(self,stage,instruction,payload,schema):
            value=super().json(stage,instruction,payload,schema)
            if stage=='draft':
                value['policy_ids']=[]
            return value
    result=SupportEngine(UncitedDraft(),Evidence()).run('AmazonHelp','Order is late',auto_send=True)
    assert result['confidence_score']==75
    assert result['route']=='human_review'
    assert 'release threshold' in result['review_reason']


def test_auto_send_uses_idempotent_local_outbox(tmp_path):
    store=RunStore(tmp_path/'runs.sqlite')
    result=SupportEngine(DemoClient(),Evidence(),store).run('AmazonHelp','Order is late',auto_send=True)
    assert result['route']=='auto_sent' and result['confidence_score']==100
    assert 'Not a probability' in result['confidence_note']
    store.save(result)
    assert store.list_runs('simulated_sent')[0]['reply']==result['final_reply']
    with store.connection() as db:
        assert db.execute('SELECT count(*) FROM outbox').fetchone()[0]==1


def test_policy_agent_rejects_unknown_policy_before_drafting():
    class WrongPolicy(DemoClient):
        def json(self,stage,instruction,payload,schema):
            value=super().json(stage,instruction,payload,schema)
            if stage=='policy_check':
                value['relevant_policy_ids']=['nonexistent']
            return value
    result=SupportEngine(WrongPolicy(),Evidence()).run('AmazonHelp','Order is late')
    assert result['route']=='human_review'
    assert not result['draft_history']
    assert result['policy_assessment']['requires_human']


def test_auto_send_without_store_does_not_claim_delivery():
    result=SupportEngine(DemoClient(),Evidence()).run('AmazonHelp','Order is late',auto_send=True)
    assert result['route']=='ready_to_send'
