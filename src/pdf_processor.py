"""PDF validation and text/metadata extraction."""

from dataclasses import dataclass

import pymupdf as fitz

from src.config import MAX_PDF_PAGES, MAX_PDF_SIZE_MB


class PDFValidationError(Exception):
    """Raised when an uploaded file fails validation or extraction. Message is safe to show to users."""


@dataclass
class PageContent:
    document_name: str
    page_number: int  # 1-indexed
    text: str


def validate_and_extract(document_name: str, file_bytes: bytes) -> list[PageContent]:
    """Validate a PDF's type/size/content and extract per-page text.

    Raises PDFValidationError with a user-safe message on any failure.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise PDFValidationError(
            f"'{document_name}' is {size_mb:.1f} MB, which exceeds the {MAX_PDF_SIZE_MB} MB limit."
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFValidationError(
            f"'{document_name}' could not be opened — it may be corrupted or not a valid PDF."
        ) from exc

    try:
        if doc.is_encrypted:
            raise PDFValidationError(f"'{document_name}' is password-protected and cannot be processed.")

        if doc.page_count == 0:
            raise PDFValidationError(f"'{document_name}' has no pages.")

        if doc.page_count > MAX_PDF_PAGES:
            raise PDFValidationError(
                f"'{document_name}' has {doc.page_count} pages, which exceeds the {MAX_PDF_PAGES}-page limit."
            )

        pages: list[PageContent] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text().strip()
            pages.append(PageContent(document_name=document_name, page_number=page_index + 1, text=text))
    finally:
        doc.close()

    if not any(page.text for page in pages):
        raise PDFValidationError(
            f"'{document_name}' contains no extractable text (it may be a scanned image with no OCR)."
        )

    return pages
