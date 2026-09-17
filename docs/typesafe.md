# Optional TypeSafe judgments

TypeSafe is a hosted service, not a local generation model. Enabling a feature
sends bounded excerpts to `https://api.typesafe.ai/v1/systemone`. Source text is
never rewritten by the model. All features default off.

Set `TYPESAFE_API_KEY` in the server/worker environment (or the repository `.env`),
never in committed YAML. Configure `typesafe` in the normal ScribeBase config:

```yaml
typesafe:
  structure_enabled: true
  reranking_enabled: true
  classification_enabled: true
  classification_confidence: 0.85
  instruction_exclusion_probability: 0.85
  candidate_pool_size: 36
  passage_max_chars: 6000
  model: jev-latest
  api_key_env: TYPESAFE_API_KEY
  timeout_seconds: 60
  batch_size: 16
  structure_confidence: 0.85
```

## Structure recovery

Before chunking, short standalone blocks are judged as chapter, section, or other
using neighboring text. Code fences and generated page labels are excluded.
Explicit chapter headings and source chapter metadata remain authoritative.
Recovered titles are copied verbatim (with existing heading cleanup); chapter
numbers are never invented. Uncertain judgments do not create new boundaries.
The 0.85 confidence threshold is a conservative starting policy, not a validated
accuracy guarantee. Evaluate against representative books before tuning it.

Content-addressed `*.structure-<hash>.json` sidecars next to the Markdown retain
raw judgments and returned model IDs. Identical input/model/prompt reuses them,
even if the hosted alias changes; threshold changes reuse the raw judgments.
Changed text or configured model creates a new sidecar. Back these up with the
canonical Markdown. Invalid sidecars fail explicitly rather than reclassifying.
No existing sources are automatically rebuilt when enabling the feature.

Missing credentials, transport errors, malformed answers, and exhausted service
errors fail the operation. Rate-limit/overload responses get two bounded retries.
Structure assessment runs before vector-store mutations, so failure preserves
the published index. Successful annotation caches may remain after a later
embedding/indexing failure and are safe to reuse.

## Query-aware reranking

Hybrid retrieval still applies the original filters and embedding-profile checks.
When enabled it retrieves `candidate_pool_size` candidates, assesses their relevance
to the query on a four-level Score rubric (0–3), then returns the requested top-k.
Requests larger than the pool fail explicitly; raise the pool (maximum 100) if
needed. Equal semantic scores preserve hybrid order. There is no low-confidence
filter: confidence is reported separately, not confused with relevance.

The original `score` and `explain_score` remain unchanged. Search JSON adds an
`assessment` with semantic score, confidence, model, raw answers, and a visible
truncation flag. Only the first `passage_max_chars` characters of a passage are
assessed; returned source text and citations remain intact. Query and title are
also sent to TypeSafe. Empty result sets and disabled features make no calls.
Service failure fails the search; it does not return an unannounced hybrid fallback.

## Passage screening for context packs

Classification shares each assessment request with reranking when both are enabled.
It also works independently, preserving hybrid ordering. A Choice labels evidence,
contradiction of the query's premise, background, or irrelevant material. A separate
Noul estimates whether the passage attempts to instruct the consuming AI.

Context packs withhold text for likely instruction attempts (probability at least
`instruction_exclusion_probability`) or confidently irrelevant material (confidence
at least `classification_confidence`). Excluded chunk IDs and reasons remain visible.
Uncertain and background passages stay in context, with their role and confidence.
Contradictions are explicitly labeled and retained unless the independent instruction
screen excludes them. If everything is excluded, the pack says no usable context remains.
Search results still expose all selected passages and raw assessments for auditing.
Filtering occurs after top-k selection, so a pack can contain fewer than top-k passages;
excluded passages are not silently replaced by lower-ranked candidates.

This is semantic screening, **not a prompt-injection security boundary**. A passage
can be misclassified, and only its assessed prefix is screened when truncation occurs.
Thresholds require evaluation on your material; unit tests establish workflow behavior,
not model accuracy. No answering/generation model is introduced.

## Mac mini rollout

After pulling the merged changes, run `uv sync --extra server --extra dev`. Existing
dependencies suffice (HTTP calls use httpx). Set the key in the environment loaded by
both launchd services and enable the three flags in the deployment config. Restart
only ScribeBase server and worker; embeddings, OCR, and Weaviate do not need restarts
or schema migration. Do not rebuild the library just to enable these features.

Smoke-test an isolated Markdown fixture through chunking and check its structure
sidecar; then exercise search and context with a known query. Check that original
hybrid scores/citations coexist with assessments, contradictions remain labeled,
and excluded passage text is absent from context packs. Disabling the three flags
restores the original path; it does not remove cached annotations or alter source text.
