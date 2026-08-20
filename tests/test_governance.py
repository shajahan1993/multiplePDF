"""Tests for governance checks (CLAUDE.md Test Set items 10-11, plus rule-based unit checks)."""

from src.chunker import chunk_pages
from src.governance import evaluate_answer
from src.pdf_processor import validate_and_extract
from src.rag import Citation, RagAnswer
from src.retriever import HybridIndex, RetrievedChunk


def _retrieved_from(chunks):
    return [RetrievedChunk(chunk=c, score=1.0) for c in chunks]


def test_prompt_injection_detected_in_retrieved_context(injection_pdf_bytes):
    pages = validate_and_extract("support_notes.pdf", injection_pdf_bytes)
    chunks = chunk_pages(pages)
    result = RagAnswer(
        answer="The notes describe a support scenario.",
        citations=[Citation(document_name="support_notes.pdf", page_number=1)],
        retrieved_chunks=_retrieved_from(chunks),
    )
    governance = evaluate_answer("What do the support notes say?", result)
    assert governance.injection_status == "REVIEW"


def test_sensitive_data_detected_in_retrieved_context(sensitive_pdf_bytes):
    pages = validate_and_extract("employee_record.pdf", sensitive_pdf_bytes)
    chunks = chunk_pages(pages)
    result = RagAnswer(
        answer="The record lists contact information.",
        citations=[Citation(document_name="employee_record.pdf", page_number=1)],
        retrieved_chunks=_retrieved_from(chunks),
    )
    governance = evaluate_answer("What is in the employee record?", result)
    assert governance.sensitive_data_status == "REVIEW"


def test_missing_citation_fails_citation_check(sample_pdf_bytes):
    pages = validate_and_extract("onboarding.pdf", sample_pdf_bytes)
    chunks = chunk_pages(pages)
    result = RagAnswer(
        answer="Some answer with no citation tags at all.",
        citations=[],
        retrieved_chunks=_retrieved_from(chunks),
    )
    governance = evaluate_answer("When does orientation happen?", result)
    assert governance.citation_status == "FAIL"


def test_insufficient_evidence_answer_passes_citation_check():
    from src.config import INSUFFICIENT_EVIDENCE_MESSAGE

    result = RagAnswer(answer=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[], retrieved_chunks=[])
    governance = evaluate_answer("Unanswerable question", result)
    assert governance.citation_status == "PASS"
    assert governance.overall_status in {"PASS", "REVIEW"}


def test_ungrounded_answer_flagged_low_groundedness():
    result = RagAnswer(
        answer="The moon is made of cheese and aliens built the pyramids.",
        citations=[],
        retrieved_chunks=[],
    )
    governance = evaluate_answer("Irrelevant question", result)
    assert governance.grounded_status == "FAIL"
    assert governance.hallucination_risk == "HIGH"
