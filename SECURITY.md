# Credential handling and deployment boundaries

The web application uses bring-your-own-key credentials. It never uses the
operator's environment key as a fallback. CLI scripts separately support a
local `.env` file for reproducible private experiments.

The key dialog masks input, disables autocomplete hints and keeps accepted keys
in Streamlit Session State. Successful submission clears the entry widget.
Cancel/dismiss clears pending input; Clear API key removes accepted session
credentials. No app code writes them to browser storage, cookies, URLs, files,
SQLite, analytics, evaluation records or downloads. Provider calls use fixed
HTTPS endpoints and authorization headers. Model prompts never receive the key.
Shared embedding/retrieval resources explicitly strip credentials. Settings
representations omit the secret, and provider errors omit response messages.

Streamlit is a server application: the browser and server process handle the
key in memory. It is not a browser-only vault. Reloading creates a new session;
disconnected sessions have a 15-second TTL configured before they are eligible
for cleanup. Cleanup timing and physical memory erasure are not guaranteed.
Clearing the UI cannot recall a request already in flight. Browser extensions,
password managers, OS access, server operators and compromised dependencies are
outside these app-level controls. Only use a key on a computer/server you trust.

For local use, bind explicitly to 127.0.0.1. Provider traffic is HTTPS; the
browser-to-localhost connection is HTTP. Cloud configuration lets the hosting
platform select the bind address. Streamlit Community Cloud supplies HTTPS/WSS;
the intended app is publicly accessible, with no shared operator key. CORS and
XSRF protection remain enabled and static serving is disabled. See DEPLOYMENT.md
for the release checklist. Do not expose a local development server directly.

Review queues, labels, and user-created evaluations are scoped to a random
session workspace. Redacted run records persist on the server under `.runtime`;
keys do not. A browser reload loses access to that workspace in the UI.
Read-only bundled dataset/benchmark reports can be shared across sessions.
On Community Cloud, these files are temporary and can disappear on restart or
redeployment. Do not enter real customer secrets in messages. Regex redaction
is incomplete. Public hosting must be verified before the app is declared live.

Tests check isolated key state, clearing, masked-dialog behavior, absence of
owner-key fallback, credential-free cached settings, HTTPS header use, sanitized
provider errors and exports/records. These are implementation checks, not a
penetration test or a guarantee against every possible disclosure.

If a key is posted in chat, source control or an export, revoke and replace it.
Set appropriate spending limits in your provider account. Never share a group
member's personal key in the submission package.

References:

- [OpenAI API key safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- [Streamlit Session State](https://docs.streamlit.io/develop/concepts/architecture/session-state)
