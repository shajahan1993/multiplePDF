"""Rule-based governance checks: groundedness, citation, prompt-injection, sensitive data."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from src.config import (
    GOVERNANCE_LOG_PATH,
    GROUNDEDNESS_PASS_THRESHOLD,
    GROUNDEDNESS_REVIEW_THRESHOLD,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    LOG_DIR,
)
from src.embeddings import cosine_similarity, embed_texts
from src.rag import RagAnswer

Status = str  # "PASS" | "REVIEW" | "FAIL"

_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal (your|the) (system prompt|instructions|api key)",
    r"act as (if )?you (are|were)",
    r"new instructions?:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_SENSITIVE_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b(?:\d[ -]*?){13,16}\b",  # credit card-ish
    r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",  # email
    r"\b\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b",  # phone
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS))


@dataclass
class GovernanceResult:
    grounded_status: Status
    citation_status: Status
    injection_status: Status
    sensitive_data_status: Status
    hallucination_risk: str  # LOW | MEDIUM | HIGH
    overall_status: Status
    groundedness_score: float
    details: dict = field(default_factory=dict)


def _overall(statuses: list[Status]) -> Status:
    if "FAIL" in statuses:
        return "FAIL"
    if "REVIEW" in statuses:
        return "REVIEW"
    return "PASS"


def _check_injection(context_text: str) -> Status:
    return "REVIEW" if _INJECTION_RE.search(context_text) else "PASS"


def _check_sensitive_data(text: str) -> Status:
    return "REVIEW" if _SENSITIVE_RE.search(text) else "PASS"


def _groundedness(answer: str, context_text: str) -> float:
    if not answer.strip() or not context_text.strip():
        return 0.0
    vectors = embed_texts([answer, context_text])
    return cosine_similarity(vectors[0], vectors[1])


def evaluate_answer(query: str, result: RagAnswer) -> GovernanceResult:
    context_text = "\n\n".join(item.chunk.text for item in result.retrieved_chunks)

    is_insufficient_evidence = result.answer.strip() == INSUFFICIENT_EVIDENCE_MESSAGE

    if is_insufficient_evidence:
        # Correctly declining to answer is not a hallucination -- there is no
        # context to be grounded in, but nothing was fabricated either.
        score = 1.0
        grounded_status, hallucination_risk = "PASS", "LOW"
    else:
        score = _groundedness(result.answer, context_text)
        if score >= GROUNDEDNESS_PASS_THRESHOLD:
            grounded_status, hallucination_risk = "PASS", "LOW"
        elif score >= GROUNDEDNESS_REVIEW_THRESHOLD:
            grounded_status, hallucination_risk = "REVIEW", "MEDIUM"
        else:
            grounded_status, hallucination_risk = "FAIL", "HIGH"

    citation_status: Status = "PASS" if (is_insufficient_evidence or result.citations) else "FAIL"

    injection_status = _check_injection(context_text)
    sensitive_data_status = _check_sensitive_data(context_text + "\n" + result.answer)

    overall_status = _overall(
        [grounded_status, citation_status, injection_status, sensitive_data_status]
    )

    governance = GovernanceResult(
        grounded_status=grounded_status,
        citation_status=citation_status,
        injection_status=injection_status,
        sensitive_data_status=sensitive_data_status,
        hallucination_risk=hallucination_risk,
        overall_status=overall_status,
        groundedness_score=score,
        details={"dropped_citation_count": result.dropped_citation_count},
    )

    _log(query, result, governance)
    return governance


def _log(query: str, result: RagAnswer, governance: GovernanceResult) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_sources": [
            {"document_name": item.chunk.document_name, "page_number": item.chunk.page_number}
            for item in result.retrieved_chunks
        ],
        "answer": result.answer,
        "governance_result": {
            "grounded_status": governance.grounded_status,
            "citation_status": governance.citation_status,
            "injection_status": governance.injection_status,
            "sensitive_data_status": governance.sensitive_data_status,
            "hallucination_risk": governance.hallucination_risk,
            "overall_status": governance.overall_status,
        },
    }
    with open(GOVERNANCE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
