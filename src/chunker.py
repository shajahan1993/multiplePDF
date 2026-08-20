"""Fixed-length, word-level chunking. Chunks never cross a page boundary,
so every chunk keeps one exact page_number for citation purposes."""

from dataclasses import dataclass

from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from src.pdf_processor import PageContent


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str


def chunk_page(page: PageContent, chunk_index_offset: int = 0) -> list[Chunk]:
    words = page.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    start = 0
    index = chunk_index_offset
    while start < len(words):
        window = words[start : start + CHUNK_SIZE_WORDS]
        chunk_id = f"{page.document_name}_p{page.page_number}_c{index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_name=page.document_name,
                page_number=page.page_number,
                text=" ".join(window),
            )
        )
        index += 1
        start += step
    return chunks


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks
