"""PolicyPulse Streamlit frontend — chat UI with telemetry sidebar and XAI inspector."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120.0
METADATA_TIMEOUT_SECONDS = 5.0
UPLOAD_TIMEOUT_SECONDS = 300.0
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "suku-pulse")


def get_backend_url() -> str:
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_upload_key" not in st.session_state:
        st.session_state.last_upload_key = None
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False


def score_to_match_percent(score: float) -> int:
    """Convert a cosine-style retrieval score into a 0-100 match percentage."""
    normalized = max(0.0, min(float(score), 1.0))
    return int(round(normalized * 100))


def score_to_progress_value(score: float) -> float:
    """Normalize retrieval score for Streamlit progress widgets."""
    return max(0.0, min(float(score), 1.0))


def fetch_metadata(backend_url: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=METADATA_TIMEOUT_SECONDS) as client:
            response = client.get(f"{backend_url}/metadata")
            response.raise_for_status()
            return response.json(), None
    except httpx.ConnectError:
        return None, "Backend server is unreachable. Start FastAPI with `uvicorn app.main:app --port 8000`."
    except httpx.TimeoutException:
        return None, "Metadata request timed out."
    except httpx.HTTPStatusError as error:
        return None, f"Metadata request failed ({error.response.status_code})."
    except Exception as error:
        return None, f"Unexpected metadata error: {error}"


def upload_policy_pdf(
    backend_url: str,
    uploaded_file: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=UPLOAD_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{backend_url}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
            response.raise_for_status()
            return response.json(), None
    except httpx.ConnectError:
        return None, "Backend server is unreachable. Ensure FastAPI is running on port 8000."
    except httpx.TimeoutException:
        return None, "Upload and indexing timed out. Try again with a smaller PDF."
    except httpx.HTTPStatusError as error:
        detail = error.response.text
        try:
            detail = error.response.json().get("detail", detail)
        except Exception:
            pass
        return None, f"Upload failed ({error.response.status_code}): {detail}"
    except Exception as error:
        return None, f"Unexpected upload error: {error}"


def query_agent(backend_url: str, query: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(f"{backend_url}/agent", json={"query": query})
            response.raise_for_status()
            return response.json(), None
    except httpx.ConnectError:
        return None, "Backend server is unreachable. Ensure FastAPI is running on port 8000."
    except httpx.TimeoutException:
        return None, "Agent request timed out. Try a shorter question or retry."
    except httpx.HTTPStatusError as error:
        detail = error.response.text
        try:
            detail = error.response.json().get("detail", detail)
        except Exception:
            pass
        return None, f"Agent request failed ({error.response.status_code}): {detail}"
    except Exception as error:
        return None, f"Unexpected agent error: {error}"


def render_policy_uploader(backend_url: str) -> None:
    st.sidebar.subheader("Upload Policy")
    st.sidebar.caption("Add a PDF to rebuild the local FAISS index instantly (max 10MB).")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a policy PDF (Max 10MB)",
        type=["pdf"],
        accept_multiple_files=False,
        help="Backend enforces a strict 10MB file size limit.",
    )

    if uploaded_file is None:
        return

    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.last_upload_key == upload_key:
        return

    with st.sidebar.spinner("Processing and Indexing..."):
        payload, error = upload_policy_pdf(backend_url, uploaded_file)

    if error:
        st.sidebar.error(error)
        return

    assert payload is not None
    st.session_state.last_upload_key = upload_key
    st.sidebar.success(
        f"Indexed `{payload.get('filename')}` — "
        f"{payload.get('chunk_count', 0)} chunks, "
        f"{payload.get('vector_count', 0)} vectors."
    )
    st.rerun()


def render_sidebar(backend_url: str) -> dict[str, Any] | None:
    st.sidebar.title("System Telemetry")
    st.sidebar.caption("Live deployment and runtime metrics")

    render_policy_uploader(backend_url)
    st.sidebar.divider()

    metadata, error = fetch_metadata(backend_url)

    if error:
        st.sidebar.error(error)
        st.sidebar.metric("Backend", "Offline")
        return None

    assert metadata is not None
    st.sidebar.success("Backend connected")
    st.sidebar.metric("Deployment", metadata.get("deployment_target", "unknown").upper())

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    st.sidebar.write(f"**Provider:** `{metadata.get('model_provider', 'n/a')}`")
    st.sidebar.write(f"**Chat model:** `{metadata.get('model_name', 'n/a')}`")
    st.sidebar.write(f"**Embedding model:** `{metadata.get('embedding_model', 'n/a')}`")

    st.sidebar.divider()
    st.sidebar.subheader("Policy Index")
    index_ready = bool(metadata.get("index_ready"))
    st.sidebar.metric("Index status", "Ready" if index_ready else "Not loaded")
    st.sidebar.metric("Chunks", metadata.get("chunk_count", 0))
    st.sidebar.metric("Vectors", metadata.get("vector_count", 0))

    built_at = metadata.get("index_built_at")
    if built_at:
        st.sidebar.caption(f"Last built: {built_at}")

    index_error = metadata.get("index_error")
    if index_error:
        st.sidebar.warning(index_error)

    st.sidebar.divider()
    st.sidebar.subheader("Latency")
    last_latency = metadata.get("latency_last_ms")
    p50_latency = metadata.get("latency_p50_ms")
    trace_count = metadata.get("recent_trace_count", 0)

    st.sidebar.metric("Last request", f"{last_latency} ms" if last_latency is not None else "—")
    st.sidebar.metric("p50 (recent)", f"{p50_latency} ms" if p50_latency is not None else "—")
    st.sidebar.caption(f"Based on {trace_count} recent trace(s)")

    st.sidebar.divider()
    st.sidebar.caption(f"API: `{backend_url}`")

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return metadata


def render_xai_inspector(trace: dict[str, Any]) -> None:
    with st.expander("XAI Inspector", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Trace ID", trace.get("trace_id", "n/a")[:8] + "…")
        cols[1].metric("Latency", f"{trace.get('latency_ms', 0)} ms")
        cols[2].metric("Tool calls", len(trace.get("tool_calls", [])))

        raw_intent = trace.get("raw_intent")
        if raw_intent:
            st.markdown("**Raw intent**")
            st.info(raw_intent)

        tool_calls = trace.get("tool_calls", [])
        if tool_calls:
            st.markdown("**Selected tools**")
            for index, tool_call in enumerate(tool_calls, start=1):
                st.code(
                    f"{index}. {tool_call.get('name')} "
                    f"(latency: {tool_call.get('latency_ms', 0)} ms)\n"
                    f"args: {tool_call.get('args', {})}",
                    language="json",
                )

        chunks = trace.get("retrieved_chunks", [])
        st.markdown(f"**Retrieved chunks ({len(chunks)})**")
        if not chunks:
            st.caption("No FAISS passages were retrieved for this response.")
            return

        for index, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "n/a")
            score = float(chunk.get("score", 0.0))
            match_percent = score_to_match_percent(score)
            text = chunk.get("text", "")
            preview = text if len(text) <= 500 else text[:500] + "…"

            st.markdown(
                f"**Passage {index}** — `{source}` · page {page} · **{match_percent}% Match**"
            )
            st.progress(score_to_progress_value(score), text=f"Retrieval confidence: {match_percent}%")
            st.text(preview)
            if index < len(chunks):
                st.divider()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if role == "assistant" and message.get("trace"):
                render_xai_inspector(message["trace"])


def handle_user_input(backend_url: str, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and synthesizing answer…"):
            payload, error = query_agent(backend_url, prompt)

        if error:
            st.error(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Error:** {error}", "trace": None}
            )
            return

        assert payload is not None
        answer = payload.get("answer", "").strip() or "_No answer returned by the agent._"
        trace = payload.get("trace")

        st.markdown(answer)
        if trace:
            render_xai_inspector(trace)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "trace": trace}
        )


def render_password_gateway() -> bool:
    """Render a simple password gate for public demo protection."""
    if st.session_state.password_correct:
        return True

    st.title("PolicyPulse Demo Access")
    st.caption(
        "PolicyPulse Demo — Password protected to prevent unauthorized LLM token consumption."
    )
    password = st.text_input("Enter demo password", type="password")

    if password == DEMO_PASSWORD:
        st.session_state.password_correct = True
        st.rerun()
    elif password:
        st.error("Incorrect password. Please try again.")

    return False


def main() -> None:
    st.set_page_config(
        page_title="PolicyPulse",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    backend_url = get_backend_url()

    if not render_password_gateway():
        return

    st.title("PolicyPulse")
    st.caption("Enterprise Policy Copilot — grounded answers from your indexed policy documents")

    render_sidebar(backend_url)

    render_chat_history()

    prompt = st.chat_input("Ask a question about company policies…")
    if prompt:
        handle_user_input(backend_url, prompt.strip())


if __name__ == "__main__":
    main()
