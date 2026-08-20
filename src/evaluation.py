"""RAG evaluation: computes metrics from real pipeline runs against a Q&A test dataset.

Every metric below is calculated from actual retrieval/generation results for the
dataset in EVAL_DATASET_PATH -- nothing here is a hard-coded score.
"""

import json
import time
from dataclasses import dataclass, field

from src.chunker import chunk_pages
from src.config import EVAL_DATASET_PATH, EVAL_PDFS_DIR, GROUNDEDNESS_PASS_THRESHOLD
from src.embeddings import cosine_similarity, embed_texts
from src.pdf_processor import validate_and_extract
from src.rag import RagAnswer, generate_answer
from src.retriever import HybridIndex


@dataclass
class EvalMetrics:
    context_precision: float
    context_recall: float
    answer_relevance: float
    faithfulness: float
    citation_accuracy: float
    retrieval_hit_rate: float
    hallucination_rate: float
    average_latency_seconds: float
    overall_score: int
    per_question: list[dict] = field(default_factory=list)


def load_eval_dataset() -> list[dict]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_index(pdf_dir=EVAL_PDFS_DIR) -> HybridIndex:
    """Build a dedicated index from the bundled sample PDFs used by the eval dataset."""
    chunks = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pages = validate_and_extract(pdf_path.name, pdf_path.read_bytes())
        chunks.extend(chunk_pages(pages))
    return HybridIndex(chunks)


def _answer_relevance(question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    vectors = embed_texts([question, answer])
    return max(0.0, cosine_similarity(vectors[0], vectors[1]))


def _faithfulness(answer: str, context_text: str) -> float:
    if not answer.strip() or not context_text.strip():
        return 0.0
    vectors = embed_texts([answer, context_text])
    return max(0.0, cosine_similarity(vectors[0], vectors[1]))


def _context_precision_recall(
    retrieved_pairs: set[tuple[str, int]], expected_pairs: set[tuple[str, int]]
) -> tuple[float, float]:
    if not retrieved_pairs and not expected_pairs:
        return 1.0, 1.0
    if not retrieved_pairs:
        return 0.0, 0.0
    if not expected_pairs:
        return 0.0, 1.0
    overlap = retrieved_pairs & expected_pairs
    precision = len(overlap) / len(retrieved_pairs)
    recall = len(overlap) / len(expected_pairs)
    return precision, recall


def run_evaluation(index: HybridIndex | None = None, dataset: list[dict] | None = None) -> EvalMetrics:
    dataset = dataset if dataset is not None else load_eval_dataset()
    index = index if index is not None else build_eval_index()

    precisions, recalls, relevances, faithfulnesses = [], [], [], []
    citation_hits, citation_total = 0, 0
    retrieval_hits = 0
    latencies = []
    hallucination_flags = []
    per_question: list[dict] = []

    for item in dataset:
        question: str = item["question"]
        expected_pairs = {
            (source["document"], source["page"]) for source in item.get("expected_sources", [])
        }

        start = time.perf_counter()
        result: RagAnswer = generate_answer(question, index)
        latency = time.perf_counter() - start
        latencies.append(latency)

        retrieved_pairs = {
            (c.chunk.document_name, c.chunk.page_number) for c in result.retrieved_chunks
        }
        precision, recall = _context_precision_recall(retrieved_pairs, expected_pairs)
        precisions.append(precision)
        recalls.append(recall)

        if expected_pairs & retrieved_pairs:
            retrieval_hits += 1

        relevance = _answer_relevance(question, result.answer)
        relevances.append(relevance)

        context_text = "\n\n".join(c.chunk.text for c in result.retrieved_chunks)
        faithfulness = _faithfulness(result.answer, context_text)
        faithfulnesses.append(faithfulness)
        hallucination_flags.append(faithfulness < GROUNDEDNESS_PASS_THRESHOLD)

        cited_pairs = {(c.document_name, c.page_number) for c in result.citations}
        citation_total += len(cited_pairs)
        citation_hits += len(cited_pairs & retrieved_pairs)

        per_question.append(
            {
                "question": question,
                "answer": result.answer,
                "context_precision": precision,
                "context_recall": recall,
                "answer_relevance": relevance,
                "faithfulness": faithfulness,
                "latency_seconds": latency,
            }
        )

    n = max(len(dataset), 1)
    context_precision = sum(precisions) / n
    context_recall = sum(recalls) / n
    answer_relevance = sum(relevances) / n
    faithfulness_avg = sum(faithfulnesses) / n
    citation_accuracy = (citation_hits / citation_total) if citation_total else 1.0
    retrieval_hit_rate = retrieval_hits / n
    hallucination_rate = sum(hallucination_flags) / n
    average_latency = sum(latencies) / n if latencies else 0.0

    overall_score = round(
        100
        * (
            0.2 * context_precision
            + 0.15 * context_recall
            + 0.2 * answer_relevance
            + 0.25 * faithfulness_avg
            + 0.1 * citation_accuracy
            + 0.1 * retrieval_hit_rate
        )
    )

    return EvalMetrics(
        context_precision=context_precision,
        context_recall=context_recall,
        answer_relevance=answer_relevance,
        faithfulness=faithfulness_avg,
        citation_accuracy=citation_accuracy,
        retrieval_hit_rate=retrieval_hit_rate,
        hallucination_rate=hallucination_rate,
        average_latency_seconds=average_latency,
        overall_score=overall_score,
        per_question=per_question,
    )
