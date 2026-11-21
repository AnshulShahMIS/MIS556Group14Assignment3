"""In-memory BYOK state helpers. No filesystem, caches, or environment writes."""
import re


def accept_key(state,provider,candidate):
    candidate = candidate.strip()
    # Format check only. Authentication/quota are checked on the first request.
    if not re.fullmatch(r'sk-[A-Za-z0-9_-]{20,}',candidate):
        raise ValueError('Enter an API secret key in the expected format.')
    keys = dict(state.get('api_credentials',{}))
    keys[provider] = candidate
    state['api_credentials'] = keys


def clear_keys(state):
    state.pop('api_credentials',None)
    state.pop('credential_error',None)
    state['byok_input'] = ''


def active_key(state,provider):
    return state.get('api_credentials',{}).get(provider,'')
