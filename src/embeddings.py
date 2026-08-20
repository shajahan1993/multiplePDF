"""Dense (sentence-transformers) and sparse (BM25) embedding helpers."""

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple word-level tokenizer: lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized dense embeddings (so inner product == cosine similarity)."""
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized_corpus)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two arbitrary (non-pre-normalized) vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
