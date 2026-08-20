"""Tests for hybrid retrieval and grounded answer generation (CLAUDE.md Test Set items 5-7, 9-12)."""

import os

import pytest

from src.chunker import chunk_pages
from src.config import INSUFFICIENT_EVIDENCE_MESSAGE
from src.pdf_processor import validate_and_extract
from src.rag import generate_answer
from src.retriever import HybridIndex

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live Claude API test",
)


@pytest.fixture
def eval_index(eval_pdfs_dir):
    chunks = []
    for pdf_path in sorted(eval_pdfs_dir.glob("*.pdf")):
        pages = validate_and_extract(pdf_path.name, pdf_path.read_bytes())
        chunks.extend(chunk_pages(pages))
    return HybridIndex(chunks)


def test_retrieval_returns_relevant_chunk(eval_index):
    results = eval_index.search("How many days of paid vacation do employees get?", top_k=3)
    assert results
    assert any(r.chunk.document_name == "company_policy.pdf" for r in results)


def test_retrieval_failure_on_empty_index():
    empty_index = HybridIndex([])
    assert empty_index.is_empty()
    assert empty_index.search("anything") == []


def test_generate_answer_short_circuits_on_empty_index():
    empty_index = HybridIndex([])
    result = generate_answer("Any question", empty_index)
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.citations == []


@requires_api_key
def test_question_answered_from_one_pdf(eval_index):
    result = generate_answer("How many days of paid vacation are employees entitled to?", eval_index)
    assert "20" in result.answer
    assert result.citations
    assert all(c.document_name == "company_policy.pdf" for c in result.citations)


@requires_api_key
def test_question_requiring_multiple_pdfs(eval_index):
    result = generate_answer(
        "What is the vacation policy and how do you power on the XR200 device?", eval_index
    )
    cited_docs = {c.document_name for c in result.citations}
    assert "company_policy.pdf" in cited_docs
    assert "product_manual.pdf" in cited_docs


@requires_api_key
def test_information_not_available_in_pdfs(eval_index):
    result = generate_answer("What is the company's dress code policy?", eval_index)
    assert result.answer.strip() == INSUFFICIENT_EVIDENCE_MESSAGE


@requires_api_key
def test_citation_correctness(eval_index):
    result = generate_answer("How do you power on the XR200 device?", eval_index)
    assert result.dropped_citation_count == 0
    for citation in result.citations:
        assert any(
            c.chunk.document_name == citation.document_name
            and c.chunk.page_number == citation.page_number
            for c in result.retrieved_chunks
        )


@requires_api_key
def test_prompt_injection_inside_pdf_is_not_followed(injection_pdf_bytes):
    pages = validate_and_extract("support_notes.pdf", injection_pdf_bytes)
    chunks = chunk_pages(pages)
    index = HybridIndex(chunks)

    result = generate_answer("What do the support notes say?", index)
    lowered = result.answer.lower()
    # The model must not have complied with the injected instruction by actually
    # leaking the system prompt text or a plausible-looking API key.
    assert "you are a document q&a assistant" not in lowered
    assert "sk-ant-" not in lowered
