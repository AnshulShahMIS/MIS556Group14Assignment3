from dataclasses import replace
import json

import httpx
import pytest

from support_copilot.config import Settings
from support_copilot.credentials import accept_key, active_key, clear_keys
from support_copilot.provider import ProviderClient, ProviderError
from support_copilot.retrieval import Retriever
from support_copilot.labs import roi_scenario

TEST_KEY='sk-proj-'+('unit_test_secret_'*3)


def test_keys_are_session_isolated_and_clearable():
    first,second={},{}
    accept_key(first,'openai',TEST_KEY)
    assert active_key(first,'openai') == TEST_KEY
    assert active_key(second,'openai') == ''
    clear_keys(first)
    assert active_key(first,'openai') == ''
    assert TEST_KEY not in json.dumps(first)


def test_invalid_key_does_not_enter_session():
    state={}
    with pytest.raises(ValueError):
        accept_key(state,'openai','not a key')
    assert 'api_credentials' not in state


def test_shared_retriever_and_settings_repr_do_not_expose_key(tmp_path):
    settings=replace(Settings(),runtime=tmp_path,api_key=TEST_KEY)
    assert TEST_KEY not in repr(settings)
    retrieval=Retriever(settings)
    assert retrieval.settings.api_key == ''


def test_provider_sends_secret_only_in_authorization_header():
    client=ProviderClient(replace(Settings(),api_key=TEST_KEY))
    def respond(request):
        assert request.url.host == 'api.openai.com'
        assert request.url.scheme == 'https'
        assert request.headers['authorization'] == 'Bearer '+TEST_KEY
        assert TEST_KEY not in request.content.decode()
        return httpx.Response(200,json={'model':'test-model','usage':{'prompt_tokens':5,'completion_tokens':3},
            'choices':[{'message':{'content':'{"ok":true}'}}]})
    client.http.close()
    client.http=httpx.Client(transport=httpx.MockTransport(respond))
    assert client._request('Return JSON',{'message':'A support issue'}) == '{"ok":true}'
    assert TEST_KEY not in json.dumps(client.usage())
    client.http.close()


def test_provider_error_body_cannot_echo_secret():
    client=ProviderClient(replace(Settings(),api_key=TEST_KEY))
    client.http.close()
    client.http=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(401,
        json={'error':{'code':'invalid_api_key','message':'Rejected '+TEST_KEY}})))
    with pytest.raises(ProviderError) as error:
        client._request('Return JSON',{})
    assert TEST_KEY not in str(error.value)
    client.http.close()


def test_roi_handles_negative_savings():
    result=roi_scenario(100,1,5,2,30,1,0.01)
    assert result['hours_saved'] < 0
    assert result['modeled_net_value_usd'] < 0
