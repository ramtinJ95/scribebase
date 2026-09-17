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
