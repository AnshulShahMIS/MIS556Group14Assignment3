# Measured results and remaining group work

> Version note: The 200-case metrics below describe the earlier workflow revision. The delivered workflow now adds VIP/spam routing, required evidence searches, a separate policy agent, confidence gating, and a simulated outbox. Its updated live benchmark has not yet been run. Current functional verification is in `current_workflow_verification.json`.

The full system is operational. This is an offline current-policy simulation using historical customer issues.
It does not establish production resolution rates or historically correct 2017 answers.

| System | n | Helpfulness /5 | Factual support /5 | Tone /5 | Median seconds | Estimated USD /case | Review rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single prompt | 200 | 3.95 | 4.28 | 4.20 | 1.47 | 0.00020 | 0.0% |
| RAG only | 200 | 3.81 | 4.04 | 4.15 | 1.83 | 0.00055 | 0.0% |
| Full agent | 200 | 3.75 | 4.20 | 4.17 | 6.63 | 0.00177 | 50.5% |

The single-prompt baseline had the highest mean judge scores in this sample. The full agent used more time and tokens and routed 50.5% of cases to human review. These descriptive results do not prove statistical significance or establish the cause of the differences.

The full agent placed 92 cases in the local ready queue, with zero detected deterministic rule violations in those replies. Rejected drafts remained in human review. Rule compliance is not proof of factual correctness. See `ready_queue_audit.json`.

The judge sees randomized anonymous variants. Route decisions are visible because safe escalation is part of response quality; this is not perfect method blinding. The same model family generates and judges responses, so independent human validation remains necessary.

The full agent adds both policy information and orchestration. This comparison cannot isolate their individual effects. Read per-case traces and report regressions as well as improvements. Warm latency excludes initial embedding-model loading. Cost counts cached input at the ordinary input rate.

Live acceptance: 5 brands; 5/5 controlled safety challenges passed; historical policy exclusion passed. These are synthetic acceptance cases, separate from the real test set.

Data separation: no shared conversation components or exact normalized customer queries. Same-brand nearest-neighbor audit flagged 0/200 inputs at cosine similarity ≥0.95. Approximate search is not proof that every possible near-duplicate is absent.

Run results: `heldout_benchmark/summary.json`, `summary.csv`, `bertscore.json`, `manifest.json`, and per-case JSON under `cases/`. Rule-based privacy findings are detection results, not exhaustive privacy audits.

Still required from the group: label the 100-case triage template, independently rate a sample of replies, analyze failures, and prepare the five-minute presentation. Triage accuracy and escalation precision/recall/F1 are deliberately unreported until real human labels exist. ROI remains a what-if calculation until a timed analyst study is done.

Potential extensions: policy-matched fixed RAG and no-critic ablations; larger brand-specific policy coverage; calibrated release thresholds; a future-period dataset; authenticated production hosting.
