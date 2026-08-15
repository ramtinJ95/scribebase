from scribebase.config import EmbeddingConfig
from scribebase.embeddings.profile import embedding_profile_fingerprint


def test_embedding_profile_fingerprint_is_stable() -> None:
    config = EmbeddingConfig()

    assert embedding_profile_fingerprint(config, 2048) == embedding_profile_fingerprint(
        config, 2048
    )


def test_embedding_profile_fingerprint_covers_vector_semantics() -> None:
    config = EmbeddingConfig()
    baseline = embedding_profile_fingerprint(config, 2048)
    variants = [
        config.model_copy(update={"provider": "other"}),
        config.model_copy(update={"model": "other-model"}),
        config.model_copy(update={"query_instruction": "search: "}),
        config.model_copy(update={"document_instruction": "document: "}),
        config.model_copy(update={"normalize": False}),
    ]

    assert all(embedding_profile_fingerprint(variant, 2048) != baseline for variant in variants)
    assert embedding_profile_fingerprint(config, 1024) != baseline
