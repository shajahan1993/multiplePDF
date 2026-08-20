"""Tests for PDF validation/extraction and chunking (CLAUDE.md Test Set items 1-4, 8, 9)."""

import pytest

from src.chunker import chunk_pages
from src.pdf_processor import PDFValidationError, validate_and_extract


def test_single_pdf_upload(sample_pdf_bytes):
    pages = validate_and_extract("onboarding.pdf", sample_pdf_bytes)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "orientation" in pages[0].text.lower()
    assert pages[1].page_number == 2


def test_multiple_pdf_upload(sample_pdf_bytes):
    pages_a = validate_and_extract("doc_a.pdf", sample_pdf_bytes)
    pages_b = validate_and_extract("doc_b.pdf", sample_pdf_bytes)
    chunks = chunk_pages(pages_a) + chunk_pages(pages_b)
    document_names = {c.document_name for c in chunks}
    assert document_names == {"doc_a.pdf", "doc_b.pdf"}


def test_corrupted_pdf_raises(corrupted_pdf_bytes):
    with pytest.raises(PDFValidationError):
        validate_and_extract("corrupted.pdf", corrupted_pdf_bytes)


def test_empty_pdf_raises(empty_pdf_bytes):
    with pytest.raises(PDFValidationError):
        validate_and_extract("empty.pdf", empty_pdf_bytes)


def test_unsupported_file_raises(unsupported_file_bytes):
    with pytest.raises(PDFValidationError):
        validate_and_extract("fake.pdf", unsupported_file_bytes)


def test_chunk_preserves_metadata(sample_pdf_bytes):
    pages = validate_and_extract("onboarding.pdf", sample_pdf_bytes)
    chunks = chunk_pages(pages)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.document_name == "onboarding.pdf"
        assert chunk.page_number in (1, 2)
        assert chunk.chunk_id
        assert chunk.text
