#!/usr/bin/env python3
"""Verify a llama.cpp embedding endpoint against the sentence-transformers reference.

Used for the Nemotron-3-Embed migration (docs/nemotron-embedding-migration.md).
Embeds a fixed set of passages and queries through both the OpenAI-compatible
/v1/embeddings endpoint and a local sentence-transformers load of the same
checkpoint, then requires:

  1. cosine(server, reference) >= --threshold for every input, and
  2. each query ranks its own passage first among all passages (both backends).

Exit code 0 only if both hold. Run on the machine serving the model:

  .venv/bin/python verify_embedding_parity.py \
    --endpoint http://127.0.0.1:8090/v1 \
    --model-name Nemotron-3-Embed-1B-BF16 \
    --reference-model ~/staging/nemotron/hf
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Query i must retrieve passage i. Deliberately diverse: prose, technical,
# multilingual, code-ish, short, long.
PAIRS: list[tuple[str, str]] = [
    (
        "how does the kubelet evict pods under memory pressure",
        "The kubelet monitors node memory and, when the eviction threshold is "
        "crossed, ranks pods by quality-of-service class and usage above "
        "requests, terminating the highest-ranked pods first.",
    ),
    (
        "what did the study find about working memory capacity",
        "The experiment showed that participants could reliably hold about four "
        "chunks in working memory, and that rehearsal strategies increased "
        "effective capacity without changing the underlying limit.",
    ),
    (
        "warum ist der Himmel blau",
        "Rayleigh-Streuung an den Molekülen der Atmosphäre streut kurzwelliges "
        "blaues Licht stärker als langwelliges rotes Licht, daher erscheint der "
        "Himmel tagsüber blau.",
    ),
    (
        "python function to deduplicate a list while preserving order",
        "def dedupe(items):\n    seen = set()\n    out = []\n    for x in items:\n"
        "        if x not in seen:\n            seen.add(x)\n            out.append(x)\n"
        "    return out",
    ),
    (
        "capital of japan",
        "Tokyo is the capital of Japan and its largest metropolitan area.",
    ),
    (
        "how are hybrid search scores combined in weaviate",
        "Hybrid search fuses a BM25 keyword score and a vector similarity score "
        "using an alpha weight; alpha 1.0 is pure vector search and alpha 0.0 is "
        "pure keyword search, with fusion applied over the two ranked lists.",
    ),
    (
        "symptoms of vitamin b12 deficiency",
        "Deficiency in vitamin B12 commonly causes fatigue, numbness or tingling "
        "in the hands and feet, memory difficulties, and megaloblastic anemia.",
    ),
    (
        "what is the plot of the odyssey",
        "After the fall of Troy, Odysseus spends ten years trying to return home "
        "to Ithaca, surviving the Cyclops, Circe, and the Sirens, while his wife "
        "Penelope fends off suitors until he returns and reclaims his household.",
    ),
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


def embed_server(endpoint: str, model: str, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts, "encoding_format": "float"})
    req = urllib.request.Request(
        f"{endpoint}/embeddings",
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    rows = sorted(payload["data"], key=lambda d: d["index"])
    return [row["embedding"] for row in rows]


def embed_reference(model_path: str, texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=4)
    return [v.tolist() for v in vectors]


def retrieval_ok(queries: list[list[float]], passages: list[list[float]], label: str) -> bool:
    ok = True
    for i, q in enumerate(queries):
        scores = [cosine(q, p) for p in passages]
        best = max(range(len(scores)), key=scores.__getitem__)
        if best != i:
            print(f"FAIL [{label}] query {i} retrieved passage {best} (score {scores[best]:.4f} vs own {scores[i]:.4f})")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8090/v1")
    parser.add_argument("--model-name", required=True, help="model name the server expects")
    parser.add_argument("--reference-model", required=True, help="HF repo id or local checkpoint dir")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument(
        "--expected-dimension",
        type=int,
        default=2048,
        help="required embedding dimension (default: 2048)",
    )
    args = parser.parse_args()

    queries = [QUERY_PREFIX + q for q, _ in PAIRS]
    passages = [PASSAGE_PREFIX + p for _, p in PAIRS]
    texts = queries + passages

    server_vecs = embed_server(args.endpoint, args.model_name, texts)
    ref_vecs = embed_reference(args.reference_model, texts)

    dim_server, dim_ref = len(server_vecs[0]), len(ref_vecs[0])
    print(f"dimensions: server={dim_server} reference={dim_ref}")
    if dim_server != args.expected_dimension or dim_ref != args.expected_dimension:
        print(
            "FAIL dimension mismatch: "
            f"expected={args.expected_dimension} server={dim_server} reference={dim_ref}"
        )
        return 1

    sims = [cosine(s, r) for s, r in zip(server_vecs, ref_vecs)]
    for text, sim in zip(texts, sims):
        marker = "ok  " if sim >= args.threshold else "FAIL"
        print(f"{marker} cos={sim:.6f}  {text[:60]!r}")
    parity = min(sims) >= args.threshold
    print(f"min cosine: {min(sims):.6f} (threshold {args.threshold})")

    n = len(PAIRS)
    retrieval = retrieval_ok(server_vecs[:n], server_vecs[n:], "server")
    retrieval &= retrieval_ok(ref_vecs[:n], ref_vecs[n:], "reference")
    if retrieval:
        print("retrieval sanity: every query ranks its own passage first (both backends)")

    if parity and retrieval:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
