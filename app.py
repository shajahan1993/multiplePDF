"""MultiPDF AI — Intelligent PDF Q&A Assistant (Streamlit UI)."""

import streamlit as st

from src.chunker import chunk_pages
from src.config import ANTHROPIC_API_KEY
from src.evaluation import EvalMetrics, run_evaluation
from src.governance import GovernanceResult, evaluate_answer
from src.pdf_processor import PDFValidationError, validate_and_extract
from src.rag import generate_answer
from src.retriever import HybridIndex

st.set_page_config(page_title="MultiPDF AI", page_icon="📄", layout="wide")

STATUS_COLOR = {"PASS": "🟢", "REVIEW": "🟡", "FAIL": "🔴"}


def _init_state() -> None:
    defaults = {
        "documents": {},  # name -> {"pages": int, "bytes": bytes}
        "chunks": [],
        "index": None,
        "chat_history": [],  # list of {"query", "result", "governance"}
        "eval_metrics": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _rebuild_index() -> None:
    st.session_state.index = HybridIndex(st.session_state.chunks)


def _handle_uploads(uploaded_files) -> None:
    for uploaded in uploaded_files:
        if uploaded.name in st.session_state.documents:
            continue
        file_bytes = uploaded.getvalue()
        with st.spinner(f"Validating and extracting '{uploaded.name}'..."):
            try:
                pages = validate_and_extract(uploaded.name, file_bytes)
            except PDFValidationError as exc:
                st.error(str(exc))
                continue
            new_chunks = chunk_pages(pages)
        st.session_state.documents[uploaded.name] = {
            "pages": len(pages),
            "bytes": file_bytes,
        }
        st.session_state.chunks.extend(new_chunks)
        st.success(f"Indexed '{uploaded.name}' ({len(pages)} pages, {len(new_chunks)} chunks).")

    if uploaded_files:
        with st.spinner("Building hybrid FAISS + BM25 index..."):
            _rebuild_index()


def _suggested_questions() -> list[str]:
    if not st.session_state.documents:
        return []
    questions = [
        "What are the main topics covered across all uploaded documents?",
        "Summarize the key points from these documents.",
    ]
    for doc_name in list(st.session_state.documents)[:2]:
        questions.append(f"What does '{doc_name}' cover?")
    return questions


def _run_query(query: str) -> None:
    if not query.strip():
        return
    if st.session_state.index is None or st.session_state.index.is_empty():
        st.warning("Upload at least one PDF before asking a question.")
        return
    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file to generate answers.")
        return

    with st.spinner("Retrieving relevant context and generating a grounded answer..."):
        try:
            result = generate_answer(query, st.session_state.index)
        except Exception as exc:  # surfaced to user without internals
            st.error(f"Failed to generate an answer: {exc}")
            return
        governance = evaluate_answer(query, result)

    st.session_state.chat_history.append(
        {"query": query, "result": result, "governance": governance}
    )


def _render_header() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            padding: 1.25rem 1.5rem; border-radius: 12px;
            background: linear-gradient(90deg, #1d4ed8, #2563eb);
            color: white; margin-bottom: 1.25rem;
        }
        .main-header h1 { margin: 0; font-size: 1.6rem; }
        .main-header p { margin: 0.25rem 0 0; opacity: 0.9; font-size: 0.95rem; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem; }
        </style>
        <div class="main-header">
            <h1>📄 MultiPDF AI</h1>
            <p>Intelligent, grounded Q&amp;A across multiple PDF documents — with citations, governance, and evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_upload_and_documents() -> None:
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drag and drop one or more PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        _handle_uploads(uploaded_files)

    if st.session_state.documents:
        st.subheader("Uploaded Documents")
        for name, meta in st.session_state.documents.items():
            st.markdown(f"- **{name}** — {meta['pages']} pages")


def _render_pdf_preview() -> None:
    st.subheader("PDF Preview")
    if not st.session_state.documents:
        st.info("Upload a PDF to preview it here.")
        return

    doc_name = st.selectbox("Select a document", list(st.session_state.documents.keys()))
    meta = st.session_state.documents[doc_name]
    page_number = st.number_input(
        "Page", min_value=1, max_value=meta["pages"], value=1, step=1
    )

    import fitz

    doc = fitz.open(stream=meta["bytes"], filetype="pdf")
    try:
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(dpi=120)
        st.image(pixmap.tobytes("png"), use_container_width=True)
    finally:
        doc.close()


def _render_chat() -> None:
    st.subheader("Ask a Question")

    suggestions = _suggested_questions()
    if suggestions:
        st.caption("Suggested questions")
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                _run_query(suggestion)

    query = st.chat_input("Ask a question about your uploaded PDFs...")
    if query:
        _run_query(query)

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["result"].answer)
            if turn["result"].citations:
                sources = ", ".join(
                    f"{c.document_name} (p. {c.page_number})" for c in turn["result"].citations
                )
                st.caption(f"Sources: {sources}")
            _render_governance_inline(turn["governance"])


def _render_governance_inline(governance: GovernanceResult) -> None:
    icon = STATUS_COLOR.get(governance.overall_status, "⚪")
    st.caption(f"{icon} Governance: {governance.overall_status}")


def _render_governance_panel() -> None:
    st.subheader("Governance Status")
    if not st.session_state.chat_history:
        st.info("Ask a question to see governance results for that answer.")
        return

    latest = st.session_state.chat_history[-1]["governance"]
    rows = [
        ("Grounded Answer", latest.grounded_status),
        ("Source Citation", latest.citation_status),
        ("Prompt Injection", latest.injection_status),
        ("Sensitive Data", latest.sensitive_data_status),
    ]
    for label, status in rows:
        st.markdown(f"{label} `{'.' * (24 - len(label))}` {STATUS_COLOR.get(status, '⚪')} {status}")
    st.markdown(f"**Hallucination Risk** `{'.' * 12}` {latest.hallucination_risk}")
    st.markdown(f"**Overall Governance** `{'.' * 10}` {STATUS_COLOR.get(latest.overall_status, '⚪')} **{latest.overall_status}**")


def _render_evaluation_panel() -> None:
    st.subheader("Evaluation Metrics")
    if st.button("Run Evaluation"):
        if st.session_state.index is None or st.session_state.index.is_empty():
            st.warning("Upload PDFs and build the index before running evaluation.")
        elif not ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file to run evaluation.")
        else:
            with st.spinner("Running evaluation against the test Q&A dataset..."):
                st.session_state.eval_metrics = run_evaluation(st.session_state.index)

    metrics: EvalMetrics | None = st.session_state.eval_metrics
    if metrics is None:
        st.info("Run the evaluation to see real, computed RAG quality metrics.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Context Precision", f"{metrics.context_precision:.2f}")
        st.metric("Context Recall", f"{metrics.context_recall:.2f}")
        st.metric("Answer Relevance", f"{metrics.answer_relevance:.2f}")
        st.metric("Faithfulness", f"{metrics.faithfulness:.2f}")
    with col2:
        st.metric("Citation Accuracy", f"{metrics.citation_accuracy:.2f}")
        st.metric("Retrieval Hit Rate", f"{metrics.retrieval_hit_rate * 100:.0f}%")
        st.metric("Hallucination Rate", f"{metrics.hallucination_rate * 100:.0f}%")
        st.metric("Average Latency", f"{metrics.average_latency_seconds:.1f} sec")

    st.markdown(f"### Overall RAG Score: {metrics.overall_score}/100")


def _render_controls() -> None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            st.session_state.documents = {}
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.chat_history = []
            st.session_state.eval_metrics = None
            st.rerun()


def main() -> None:
    _init_state()
    _render_header()

    if not ANTHROPIC_API_KEY:
        st.warning(
            "ANTHROPIC_API_KEY is not set — PDF upload/indexing works, but answer "
            "generation and evaluation require a key in your .env file."
        )

    left, right = st.columns([1, 1.4])
    with left:
        _render_upload_and_documents()
        _render_pdf_preview()
        _render_controls()
    with right:
        _render_chat()
        _render_governance_panel()
        _render_evaluation_panel()


if __name__ == "__main__":
    main()
