"""Central configuration constants for the MultiPDF AI pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL: str = "claude-opus-5"

# PDF validation
MAX_PDF_SIZE_MB: int = 50
MAX_PDF_PAGES: int = 500

# Chunking (fixed-length, word-level tokenization)
CHUNK_SIZE_WORDS: int = 200
CHUNK_OVERLAP_WORDS: int = 50

# Embeddings
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
TOP_K: int = 5
DENSE_WEIGHT: float = 0.5
SPARSE_WEIGHT: float = 0.5

# Governance thresholds
GROUNDEDNESS_PASS_THRESHOLD: float = 0.45
GROUNDEDNESS_REVIEW_THRESHOLD: float = 0.25

# Evaluation
_FIXTURES_DIR: Path = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
EVAL_DATASET_PATH: Path = _FIXTURES_DIR / "qa_eval_set.json"
EVAL_PDFS_DIR: Path = _FIXTURES_DIR / "sample_pdfs"

# Logging
LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
GOVERNANCE_LOG_PATH: Path = LOG_DIR / "governance_log.jsonl"

INSUFFICIENT_EVIDENCE_MESSAGE: str = (
    "I could not find sufficient information in the uploaded PDFs."
)
