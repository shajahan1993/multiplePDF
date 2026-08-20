"""FAISS dense index + BM25 sparse index, combined via weighted hybrid scoring."""

from dataclasses import dataclass

import faiss
import numpy as np

from src.chunker import Chunk
from src.config import DENSE_WEIGHT, SPARSE_WEIGHT, TOP_K
from src.embeddings import build_bm25_index, embed_query, embed_texts, tokenize


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class HybridIndex:
    """Holds a FAISS dense index and a BM25 sparse index over the same chunk set."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.dense_index = None
            self.bm25_index = None
            return

        vectors = embed_texts([c.text for c in chunks])
        dimension = vectors.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        self.dense_index.add(vectors)
        self.bm25_index = build_bm25_index([c.text for c in chunks])

    def is_empty(self) -> bool:
        return not self.chunks

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        if self.is_empty():
            return []

        n = len(self.chunks)
        k = min(top_k, n)

        query_vector = embed_query(query).reshape(1, -1)
        dense_scores_all, dense_indices_all = self.dense_index.search(query_vector, n)
        dense_scores = _score_array_from_search(dense_scores_all[0], dense_indices_all[0], n)

        bm25_scores = np.asarray(self.bm25_index.get_scores(tokenize(query)), dtype="float32")

        dense_norm = _min_max_normalize(dense_scores)
        sparse_norm = _min_max_normalize(bm25_scores)

        hybrid_scores = DENSE_WEIGHT * dense_norm + SPARSE_WEIGHT * sparse_norm

        top_indices = np.argsort(-hybrid_scores)[:k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(hybrid_scores[i])) for i in top_indices]


def _score_array_from_search(scores: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """FAISS search over the full corpus returns every index once; realign into position order."""
    full = np.zeros(n, dtype="float32")
    for score, idx in zip(scores, indices):
        if idx >= 0:
            full[idx] = score
    return full


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)
