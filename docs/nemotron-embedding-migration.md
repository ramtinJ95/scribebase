# Migration plan: Qwen3-Embedding-4B → Nemotron-3-Embed-1B-BF16

Status: COMPLETE (2026-08-15). All phases done. Parity PASS (min cosine
0.999927), full reindex of 8 sources / 5,643 chunks promoted to alias `Chunk`,
manifests stamped `Nemotron-3-Embed-1B-BF16` @ 2048 dims, baseline query
comparison showed no regressions (kubelet-eviction query improved). Follow-up
planned: reduce embedding server `--ctx-size` 32768 → 8192 to cut steady-state
memory. Reminder: launchd plist edits need `launchctl bootout` + `bootstrap`;
`kickstart -k` restarts with the cached job definition.

## Phase 0 results and required workarounds

Verified on the Mac mini with llama.cpp release b10437 (installed side by side
at `~/.local/llama.cpp-b10437`, the running b9902 services untouched).
Artifacts live in `~/staging/nemotron/` on the mini: the converted
`Nemotron-3-Embed-1B-BF16.gguf` (2.1 GB), the HF checkpoint, and
`baseline_qwen_searches.txt` (5 queries against the live Qwen index for
post-cutover comparison).

Three workarounds were needed; all are load-bearing for Phase 2/3:

1. **Converter does not know the embed architecture.** The checkpoint declares
   `architectures: ["Ministral3Model"]`; llama.cpp's conversion registry only
   maps `Ministral3ForCausalLM`. Fix: temporarily patch `config.json` to
   `Ministral3ForCausalLM` for the conversion (identical backbone weights).
   The converter still reads `is_causal: false` from the config and writes
   `mistral3.attention.causal = False` into the GGUF, so the served model is
   self-describingly bidirectional — `--attention non-causal` is NOT needed
   (and llama-server does not accept that flag anyway). Worth filing upstream:
   register `Ministral3Model` in `conversion/__init__.py`.
2. **Converter venv needs current transformers.** The pinned requirements are
   too old for the checkpoint's tokenizer format (`TokenizersBackend`,
   list-valued `extra_special_tokens`). `uv pip install -U transformers
   tokenizers` (5.15.0 worked) before converting.
3. **BOS must be disabled at serve time.** llama.cpp defaults to prepending
   `<s>`; the HF reference tokenizer does not. Under mean pooling this skewed
   every vector (parity min cosine 0.977, worst on short texts). Serve with
   `--override-kv tokenizer.ggml.add_bos_token=bool:false` → parity min cosine
   0.999927. This flag MUST be in the launchd plist; without it embeddings are
   silently degraded.

4. **Micro-batch must cover the longest input.** Non-causal embedding requires
   each input to fit within one physical batch; llama-server otherwise clamps
   to `n_ubatch=512` and returns HTTP 500 for longer chunks ("input (N tokens)
   is too large to process"). This surfaced during the first `rebuild-index
   --all`, not during parity testing (short texts). Serve with
   `--ubatch-size 4096 --batch-size 4096`.

Non-issue confirmed: the config's `apply_yarn_scaling: false` is ignored by
transformers, but llama.cpp and transformers apply YaRN identically, so rope
needs no overrides (`--rope-scaling none` is NOT needed).

Verified serve command:

```bash
llama-server \
  --model ./models/Nemotron-3-Embed-1B-BF16.gguf \
  --embedding \
  --pooling mean \
  --override-kv tokenizer.ggml.add_bos_token=bool:false \
  --ctx-size 32768 \
  -ngl 99 \
  --alias Nemotron-3-Embed-1B-BF16 \
  --ubatch-size 4096 \
  --batch-size 4096 \
  --port 8080
```

## Goal

Replace the embedding model with
[nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
while keeping the BYO-vectors architecture: ScribeBase computes vectors through a
local llama.cpp `/v1/embeddings` server and pushes them into Weaviate as
self-provided vectors. No Weaviate module changes are needed or wanted; Weaviate
never computes embeddings in this design.

## Model differences that drive the changes

| | Qwen3-Embedding-4B (current) | Nemotron-3-Embed-1B-BF16 (target) |
|---|---|---|
| Attention | causal | bidirectional |
| llama.cpp pooling | `--pooling last` | `--pooling mean` |
| Dimensions | 2560 | 2048 (Matryoshka-truncatable) |
| Query prefix | Qwen instruct template (`Instruct: ...\nQuery: `) | `query: ` |
| Document prefix | none | `passage: ` |
| Max context | 32k | 32k |

Prefixes are applied transiently in the embedding client only. Stored chunk text
and the BM25 half of hybrid search stay prefix-free.

## Key risk

There is no official GGUF. We convert the BF16 safetensors ourselves. If
llama.cpp runs this bidirectional encoder with causal attention, it produces
plausible-looking but wrong vectors — a silent failure. Phase 0 therefore gates
everything on a numeric parity check against the sentence-transformers reference
implementation. Fallback runtimes if llama.cpp fails: MLX community builds, or a
thin OpenAI-compatible sentence-transformers wrapper (the app only speaks
`/v1/embeddings` HTTP, so either slots in without code changes).

## Phase 0 — Feasibility gate (on the Mac mini)

1. Verify llama.cpp build is recent enough for the architecture (model released
   July 2026): `llama-server --version`. Upgrade if older.
2. Download `nvidia/Nemotron-3-Embed-1B-BF16` (~2.3 GB safetensors) and convert:
   `convert_hf_to_gguf.py --outtype bf16` → `models/Nemotron-3-Embed-1B-BF16.gguf`.
   If the converter rejects the architecture: stop, fall back (see above).
3. Serve it:

   ```bash
   llama-server \
     --model ./models/Nemotron-3-Embed-1B-BF16.gguf \
     --embedding \
     --pooling mean \
     --ctx-size 32768 \
     -ngl 99 \
     --alias Nemotron-3-Embed-1B-BF16 \
  --ubatch-size 4096 \
  --batch-size 4096 \
     --port 8080
   ```

   `--alias` gives a stable model name for the config-match guard.
4. Parity check with `scripts/verify_embedding_parity.py`: embed ~10 diverse
   texts and prefixed queries through both the llama-server endpoint and
   sentence-transformers; require cosine ≥ 0.99 per pair, 2048 dims, and a
   retrieval sanity check (each `query: ...` ranks its own `passage: ...`
   first). If BF16 fails parity, try Q8_0 before abandoning llama.cpp.
5. Capture a baseline: run ~5 known-good searches against the current Qwen index
   and save results for post-cutover comparison.

## Phase 1 — Code changes

6. `src/scribebase/config.py` (`EmbeddingConfig`): default `model` →
   `"Nemotron-3-Embed-1B-BF16"`, default `query_instruction` → `"query: "`,
   new field `document_instruction: str = "passage: "`. `normalize: True`
   stays; the unused `dimension` field stays untouched.
7. `src/scribebase/embeddings/llamacpp_client.py`: make the raw HTTP embed
   private; `embed_query` applies `query_instruction`, `embed_batches` applies
   `document_instruction`, `detect_dimension` uses raw text. The restructure
   exists to make double-prefixing impossible (`embed_query` currently funnels
   through `embed_texts`).
8. `indexing.py` keeps passing raw chunk text; prefixing happens inside the
   client only.
9. Tests (`tests/test_embedding_client.py` and indexing tests): document prefix
   applied on batch embed, query prefix on query embed, dimension probe
   unprefixed, upserted properties contain unprefixed text.

## Phase 2 — Docs and service definitions

10. Update README (Quickstart command; "`--pooling last` is required for Qwen"
    → "`--pooling mean` is required for Nemotron"), `docs/macmini-deployment.md`,
    `docs/launchd/com.scribebase.embedding.plist.example` (model path,
    `--pooling mean`, `--alias`).

## Phase 3 — Cutover on the Mac mini

11. Install the updated launchd plist and restart the embedding service.
12. Update the live `config.yaml` on the mini: `embedding.model`,
    `query_instruction`, new `document_instruction`. The deployed file has the
    old Qwen instruct string explicitly set, so new defaults alone do not fix it.
13. `scribebase doctor`, then `scribebase rebuild-index --all`. The model-name
    and dimension guards (2560 → 2048) force the rebuild; the alias-based
    blue/green rebuild keeps search up, and a failed build leaves the live
    index untouched.
14. Re-run the Phase 0 baseline queries and compare quality.

## Rollback

Keep the Qwen GGUF in `models/`. Revert = old plist + old config values +
`rebuild-index --all`. The mismatch guards make a half-migrated state impossible
rather than quietly degraded.

## Decisions locked in

- BYO vectors stay; no Weaviate vectorizer modules (Docker on Apple Silicon has
  no GPU passthrough — module inference would be CPU-only in a VM).
- BF16 as specified; Q8_0 only as a parity-failure fallback.
- Repo defaults flip to Nemotron rather than keeping Qwen as documented default.
