# Support Copilot

## Assignment report

[Download the six-page assignment report](Support-Copilot-Assignment-3-Report.pdf).
The report covers the business problem, agent workflow, real dataset, seven
workspace tabs, batch processing, security, measured results, and pros and cons.
The 200-case results describe an earlier workflow revision; the current live
benchmark and independent human ratings remain pending.

Public demo: https://support-copilot-mis556-group14.streamlit.app/?embed=true

## Batch support inbox

The Batch inbox tab accepts UTF-8 CSV files with `brand,message` and optional
`message_id,customer_vip` columns. Up to 50 messages and 1 MB are accepted.
Download the five-brand sample or select **Try sample inbox** for a free preview.
VIP must be explicitly operator confirmed; values are `true` or `false`.
Unsupported columns and malformed rows are rejected with row-level errors.
Messages are redacted before processing, but redaction is incomplete: use
synthetic examples without real customer secrets.

Risk signals and VIP rows receive High intake priority before LLM triage.
Process 1, 3, 5, or 10 messages per action using the current visitor's key.
Replies remain drafts; batch processing disables simulated auto-send.
Flagged runs appear in the existing Review queue, where edits and decisions
are reflected in batch exports after refreshing. Results include agent urgency,
readiness, estimated cost, latency, run identifiers, and evidence traces.
CSV exports neutralize spreadsheet formula prefixes.

Loading the same validated inbox preserves progress. Claimed/completed rows
are not automatically repeated; an error stops the current processing action.
Interrupted rows remain marked for manual inspection in run history to avoid
silently charging twice. A different import replaces the active batch view;
existing session run records remain available. Batch progress stays in Session
State and is lost on reload or a new session. Export results before leaving.

## Purpose

Help customer-support teams handle incoming public tweets by understanding the
issue, retrieving real same-brand support conversations, checking official
policy, drafting an appropriate reply, and escalating cases requiring a human.
TWCS provides real historical customer/brand conversations; the application
uses five brands from that dataset. Historical replies provide examples and
evaluation references, not authoritative policy or proof of resolution.

## Required workflow

| Stage | Implemented behavior |
|---|---|
| Triage | Intent, sentiment, urgency, trusted brand and operator-confirmed VIP metadata |
| Python router | Ignore spam/off-topic; escalate legal/safety/account risks and VIPs |
| Retrieval | Search five same-brand historical customer/reply pairs |
| Policy agent | Assess relevance and sufficiency of retrieved brand FAQ/policy notes before drafting |
| Draft | Brand-appropriate public reply within the 280-character prototype limit |
| Critic | LLM tone/support review plus deterministic promise, privacy, citation and length checks |
| Revision | At most two rewrites after the original draft; unresolved failure goes to review |
| Confidence | Four transparent readiness checks, 25 points each; default release threshold 80 |
| Delivery | Passing replies can enter an idempotent local simulated auto-send outbox; risky/uncertain cases enter human review |

The model can choose search order and queries, but cannot finish research before
attempting both evidence tools. Tool failures consume a four-search budget.
VIP status must be supplied by a support operator from known customer records;
the dataset does not establish VIP status. The confidence score is a heuristic
readiness indicator, not a calibrated probability of correctness.

The web app simulates auto-send locally and never posts to Twitter/X. Live public
posting requires a separate authorized platform integration. If simulation is
turned off, passing replies enter the local ready queue instead. Human review
supports edited approval/rejection and an audit trail. The original 200-case
benchmark remains a reference for the earlier workflow revision; it must be
rerun to measure these updated routing/evidence/confidence behaviors.

A local customer-support agent for MIS 556. Python 3.12, LangGraph, Streamlit,
Chroma and sentence-transformers. Five brands: AmazonHelp, AppleSupport,
Uber_Support, SpotifyCares and Delta.

## Quick start (Windows PowerShell)

Open PowerShell in this extracted folder. Install Python 3.12 if needed.
requirements-local.txt includes local evaluation and test dependencies.
requirements.txt is reserved for the Linux Streamlit Cloud deployment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
Copy-Item .env.example .env
```

The web app has a **Use your own API key** dialog. It accepts a key for your
session without saving it to disk. A `.env` key is used only by CLI experiments,
never as a fallback for web visitors. See [SECURITY.md](SECURITY.md) for the
precise memory and deployment boundaries. Then run:

```powershell
.\.venv\Scripts\python.exe -m support_copilot.cli build-index
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The first index build downloads a pinned MiniLM
embedding model and embeds 50,000 examples on your CPU. An internet connection
is required for installation, the initial model download and live model calls.
Indexing needs no OpenAI key. The CLI's explicitly labeled `--demo` mode
uses rules; it is not an LLM and is excluded from measured model evaluations.

The web UI uses `gpt-4.1-mini` with an OpenAI key and hides model/provider controls.
Provider/model remain configurable for CLI experiments. Token rates and policy
context are under Knowledge base → Advanced settings. There is no sidebar.
OpenAI and Anthropic native function calls are supported by the backend; only
OpenAI is live-tested in this delivery. Anthropic requires its own API key.

## Workflow and genuine tool decisions

The API-key button is in the top-right header. It opens a masked session-key
dialog with connect, clear, and cancel actions. The drafting screen offers
brand-specific examples; controlled draft injection is in Safety lab.

The seven tabs are Draft reply, Batch inbox, Review queue, Evaluation lab,
Operations & ROI, Safety lab, and Knowledge base. Operations shows session-scoped intent/routing
analytics and a transparent what-if labor-cost calculator. Safety lab runs
separate synthetic risk and unsafe-draft tests. Knowledge base supports source
inspection and brand-scoped policy searches. The previous Project evidence
tab has been removed.

LangGraph stores structured triage, retrieved evidence, drafts, critic feedback,
usage and an observable event trace. The model chooses native `search_cases`,
`search_policy` and `finish_research` tool calls, can change its query based on
observations, and operates within four searches. Brand filtering is enforced by
the application. Retrieved customer text is treated as untrusted data.

Risk routing precedes retrieval. A draft receives deterministic checks plus an
LLM critic. It may be rewritten twice before human review. Tool failures consume
the search budget; model/execution failures preserve partial state for review.
The review queue supports edit, approve and reject with a SQLite audit record.
There is no external tweet-posting integration. `ready_to_send` is a local queue
status, not proof of factual correctness or delivery.

## Data and evidence

- `data/index_pairs.jsonl`: 50,000 historical pairs, 10,000 per brand.
- `data/development_pairs.jsonl`: 100 development cases, 20 per brand.
- `data/test_pairs.jsonl`: 200 held-out cases, 40 per brand.
- `data/test_inputs.jsonl`: held-out input view without reference answers.
- `data/triage_annotation_template.jsonl`: 100 cases awaiting human labels.
- `data_audit.json`: full-source counts and preprocessing details.
- `policies/verified_notes.json`: 11 paraphrased notes with official source URLs.

Source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
The supplied `twcs.csv` was authoritative. The supplied sample has incompatible
tweet-ID/text mappings and was used only for parser compatibility checks.
Conversation components and exact normalized customer queries are separated
across splits. The index contains training examples only. Reference replies
are supplied to the evaluation judge, never to the generating agent for that
case. Regex redaction is imperfect; use synthetic messages in public demos.
Respect the dataset's source terms before redistributing it outside your group.

Historical replies are examples of brand style and past handling. They are not
gold-standard answers or authoritative current policy. Current official notes
have limited topic coverage and no verified 2017 effective dates. The default
**current support simulation** applies current evidence to historical customer
issues. **Historical replay** excludes those notes and generally needs review.
Do not describe current-policy results as historically accurate 2017 answers.

## Reproduce evaluation

The `materials` folder contains two 60-case human-rater assignments with a
20-case shared overlap covering 100 unique tweets. Upload each JSON in the
Evaluation lab's human-labeling section. Read `materials/LABELING_GUIDE.txt`,
work independently, and export before leaving the session. Labels are blank.

After both raters finish:

```powershell
.\.venv\Scripts\python.exe tools/compare_human_labels.py --rater-a rater_A_completed.json --rater-b rater_B_completed.json --output adjudication
```

The tool reports field-level agreement and Cohen's kappa, and creates a
`consensus_draft.json` with unresolved/missing cases left blank for adjudication.
It never fabricates a human label. After the current live benchmark, use
`tools/prepare_reply_review.py --report <benchmark-folder> --output reply_review`
to generate balanced anonymous reply-quality forms. Keep its private system key
away from raters. The timed demo and validation order are also in `materials`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m support_copilot.cli evaluate --split development --limit 5
.\.venv\Scripts\python.exe -m support_copilot.cli evaluate --split test --limit 200 --workers 4
.\.venv\Scripts\python.exe -m support_copilot.cli bertscore
```

Evaluation makes paid API calls.
For CLI calls, first put your own key in your local `.env`. The web dialog's
session key is deliberately unavailable to CLI processes.

Runs resume completed cases under a fingerprint
of model, code, corpus and policy. Results are stored in `.runtime/evaluations`.
Baseline and RAG use the same model and basic privacy instructions; the full
agent adds triage, tools, current-policy evidence, critique and routing. Their
comparison measures the overall package; it does not isolate the causal effect
of each component. Model settings, actual returned model IDs, token usage,
estimated costs and per-case traces are retained. Prices are estimates, with
cached tokens counted at the full input rate; they are not billing records.

The judge scores anonymized, randomized candidates on helpfulness, factual
support and tone. It is the same model family, so self-evaluation bias remains.
Historical references can themselves be generic or incorrect. BERTScore uses
DistilBERT and measures reply similarity, not factual correctness or safety.
Means and descriptive confidence intervals do not establish significance.
Do not assume the agent outperforms the baselines; report the measured results.

Use the Evaluation tab to label intent, urgency and human-review need on the
100-case template. Have two teammates independently label a shared subset,
resolve disagreements and record the rubric. Until labels exist, triage
accuracy and escalation precision/recall/F1 are unmeasured. Do not use model
labels as human ground truth. Reviewed natural cases and clearly labeled
controlled risk tests should be reported separately.

## Report and live demonstration

1. Public support problem, business benefit and why tool decisions are useful.
2. Graph, tool choices, memory/state and bounded recovery.
3. Live case: evidence, reply, critic; then controlled unsafe-draft revision.
4. Held-out comparison: quality, review rate, latency and estimated cost.
5. Pros/cons: human control, limited policies, dated data, model bias and ROI.

The assignment explicitly requires a real application, working agentic system,
real-world dataset/effectiveness demonstration, and pros/cons. The ablation-style
comparison and detailed evaluation strengthen graduate-level depth; they do not
replace your group's analysis, manual annotations or final report.

## File map

`app.py` is the UI. `support_copilot/engine.py` is the graph; `provider.py` handles
model calls; `retrieval.py` builds and searches evidence; `privacy.py` applies
checks; `storage.py` persists review decisions; `evaluation.py` runs benchmarks.
Runtime databases, model caches, labels and local credentials are excluded from
the source package. User-created web runs, labels and evaluations are scoped to
a random session workspace under `.runtime/sessions`. Reloading starts a new
workspace; export important results first. CLI evaluation uses
`.runtime/evaluations`. Back up your own labels and review decisions locally.
