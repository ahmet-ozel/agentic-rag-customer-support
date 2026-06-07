"""AgentDesk Gradio Frontend - Chat, document upload, and system info UI.

Run standalone:
    python frontend/gradio_app.py

Connects to the AgentDesk FastAPI backend (default: http://localhost:8000).

Validates: Requirements 12.1, 12.2, 12.3, 12.4
"""

from __future__ import annotations

import os
import uuid

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("AGENTDESK_BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT = float(os.environ.get("AGENTDESK_HTTP_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _client() -> httpx.Client:
    """Return a synchronous httpx client (Gradio callbacks are sync)."""
    return httpx.Client(base_url=BACKEND_URL, timeout=HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------


def chat_send(
    user_message: str,
    chat_history: list[dict],
    session_id: str | None,
) -> tuple[list[dict], str, str, str]:
    """Send a message to /api/v1/chat and return updated state.

    Returns:
        (chat_history, session_id, citations_md, traces_md)
    """
    if not user_message.strip():
        return chat_history, session_id or "", "", ""

    # Append user message to history display
    chat_history = chat_history + [{"role": "user", "content": user_message}]

    payload: dict = {"message": user_message}
    if session_id:
        payload["session_id"] = session_id

    try:
        with _client() as client:
            resp = client.post("/api/v1/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        err_msg = f" Backend error: {exc.response.status_code}"
        chat_history = chat_history + [{"role": "assistant", "content": err_msg}]
        return chat_history, session_id or "", "", ""
    except Exception as exc:
        err_msg = f" Connection error: {exc}"
        chat_history = chat_history + [{"role": "assistant", "content": err_msg}]
        return chat_history, session_id or "", "", ""

    assistant_msg = data.get("response", "")
    new_session_id = data.get("session_id", session_id or "")

    chat_history = chat_history + [{"role": "assistant", "content": assistant_msg}]

    # Format citations
    citations_md = _format_citations(data.get("citations", []))

    # Format tool traces
    traces_md = _format_traces(data.get("tool_traces", []))

    return chat_history, new_session_id, citations_md, traces_md


def _format_citations(citations: list[dict]) -> str:
    """Render citations as Markdown."""
    if not citations:
        return "_No citations._"
    lines: list[str] = []
    for i, c in enumerate(citations, 1):
        source = c.get("source", "unknown")
        page = c.get("page")
        text = c.get("text", "")
        score = c.get("score", 0.0)
        page_str = f", p.{page}" if page else ""
        lines.append(f"**[{i}]** {source}{page_str} (score: {score:.2f})")
        lines.append(f"> {text}")
        lines.append("")
    return "\n".join(lines)


def _format_traces(traces: list[dict]) -> str:
    """Render tool traces as Markdown."""
    if not traces:
        return "_No tool calls._"
    lines: list[str] = []
    for t in traces:
        name = t.get("tool_name", "?")
        server = t.get("server_name", "?")
        duration = t.get("duration_ms", 0.0)
        summary = t.get("result_summary", "")
        lines.append(f" **{name}** ({server}) - {duration:.0f} ms")
        if summary:
            lines.append(f"> {summary}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document upload logic
# ---------------------------------------------------------------------------


def upload_document(file) -> str:
    """Upload a file to /api/v1/documents and return status text."""
    if file is None:
        return "No file selected."

    try:
        with _client() as client:
            with open(file, "rb") as f:
                filename = os.path.basename(file)
                resp = client.post(
                    "/api/v1/documents",
                    files={"file": (filename, f)},
                )
                resp.raise_for_status()
                data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f" Upload failed: {exc.response.status_code} - {exc.response.text}"
    except Exception as exc:
        return f" Connection error: {exc}"

    doc_id = data.get("document_id", "?")
    status = data.get("status", "?")
    message = data.get("message", "")
    filename = data.get("filename", "?")
    return (
        f" **{filename}** uploaded successfully\n\n"
        f"- Document ID: `{doc_id}`\n"
        f"- Status: {status}\n"
        f"- {message}"
    )


def list_documents() -> str:
    """Fetch document list from /api/v1/documents."""
    try:
        with _client() as client:
            resp = client.get("/api/v1/documents")
            resp.raise_for_status()
            docs = resp.json()
    except Exception as exc:
        return f" Error: {exc}"

    if not docs:
        return "_No documents uploaded yet._"

    lines: list[str] = []
    for d in docs:
        name = d.get("filename", "?")
        status = d.get("status", "?")
        doc_id = d.get("document_id", "?")
        lines.append(f"- **{name}** - {status} (`{doc_id}`)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Settings / info panel logic
# ---------------------------------------------------------------------------


def fetch_system_info() -> str:
    """Fetch system info from /info endpoint."""
    try:
        with _client() as client:
            resp = client.get("/info")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return f" Error: {exc}"

    version = data.get("version", "?")
    llm_provider = data.get("active_llm_provider", "?")
    llm_model = data.get("active_llm_model", "?")
    vector_store = data.get("active_vector_store", "?")
    mcp_servers = data.get("enabled_mcp_servers", [])

    return (
        f"**Version:** {version}\n\n"
        f"**LLM Provider:** {llm_provider}\n\n"
        f"**LLM Model:** {llm_model}\n\n"
        f"**Vector Store:** {vector_store}\n\n"
        f"**Enabled MCP Servers:** {', '.join(mcp_servers) if mcp_servers else 'None'}"
    )


def fetch_mcp_status() -> str:
    """Fetch MCP server statuses from /api/v1/mcp/status."""
    try:
        with _client() as client:
            resp = client.get("/api/v1/mcp/status")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return f" Error: {exc}"

    servers = data.get("servers", {})
    if not servers:
        return "_No MCP servers configured._"

    lines: list[str] = []
    for name, info in servers.items():
        status = info.get("status", "?")
        transport = info.get("transport", "?")
        uptime = info.get("uptime_seconds")
        error = info.get("error_message")

        icon = {"running": "", "stopped": "", "error": ""}.get(status, "")
        line = f"{icon} **{name}** - {status} ({transport})"
        if uptime is not None:
            line += f" - uptime {uptime:.0f}s"
        if error:
            line += f"\n  >  {error}"
        lines.append(line)
    return "\n".join(lines)


def fetch_stats() -> str:
    """Fetch system stats from /api/v1/stats."""
    try:
        with _client() as client:
            resp = client.get("/api/v1/stats")
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return f" Error: {exc}"

    convos = data.get("total_conversations", 0)
    tokens = data.get("total_tokens", {})
    avg_ms = data.get("average_response_time_ms", 0.0)
    tool_dist = data.get("tool_call_distribution", {})

    tool_lines = "\n".join(
        f"  - {k}: {v}" for k, v in tool_dist.items()
    ) if tool_dist else "  _None_"

    return (
        f"**Total Conversations:** {convos}\n\n"
        f"**Total Tokens:** {tokens.get('total_tokens', 0)} "
        f"(prompt: {tokens.get('prompt_tokens', 0)}, "
        f"completion: {tokens.get('completion_tokens', 0)})\n\n"
        f"**Avg Response Time:** {avg_ms:.1f} ms\n\n"
        f"**Tool Call Distribution:**\n{tool_lines}"
    )


# ---------------------------------------------------------------------------
# Gradio UI (Blocks layout)
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(
        title="AgentDesk - RAG Assistant",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("#  AgentDesk - RAG Customer Support Assistant")

        # Hidden state for session_id
        session_state = gr.State(value="")

        with gr.Tabs():
            # ---- Chat Tab ----
            with gr.Tab(" Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    height=450,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message…",
                        show_label=False,
                        scale=8,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Accordion(" Source Citations", open=False):
                    citations_display = gr.Markdown("_No citations yet._")

                with gr.Accordion(" Tool Traces", open=False):
                    traces_display = gr.Markdown("_No tool traces yet._")

                # Wire up chat events
                chat_inputs = [msg_input, chatbot, session_state]
                chat_outputs = [chatbot, session_state, citations_display, traces_display]

                send_btn.click(
                    fn=chat_send,
                    inputs=chat_inputs,
                    outputs=chat_outputs,
                ).then(fn=lambda: "", outputs=msg_input)

                msg_input.submit(
                    fn=chat_send,
                    inputs=chat_inputs,
                    outputs=chat_outputs,
                ).then(fn=lambda: "", outputs=msg_input)

            # ---- Documents Tab ----
            with gr.Tab(" Documents"):
                gr.Markdown("### Upload Documents")
                with gr.Row():
                    file_input = gr.File(label="Select file to upload")
                    upload_btn = gr.Button("Upload", variant="primary")
                upload_status = gr.Markdown("_No upload yet._")

                upload_btn.click(
                    fn=upload_document,
                    inputs=file_input,
                    outputs=upload_status,
                )

                gr.Markdown("### Uploaded Documents")
                doc_list_display = gr.Markdown("_Click refresh to load._")
                refresh_docs_btn = gr.Button(" Refresh List")
                refresh_docs_btn.click(
                    fn=list_documents,
                    outputs=doc_list_display,
                )

            # ---- Settings Tab ----
            with gr.Tab(" Settings & Info"):
                gr.Markdown("### System Information")
                sys_info_display = gr.Markdown("_Click refresh to load._")
                refresh_info_btn = gr.Button(" Refresh System Info")
                refresh_info_btn.click(
                    fn=fetch_system_info,
                    outputs=sys_info_display,
                )

                gr.Markdown("### MCP Server Status")
                mcp_status_display = gr.Markdown("_Click refresh to load._")
                refresh_mcp_btn = gr.Button(" Refresh MCP Status")
                refresh_mcp_btn.click(
                    fn=fetch_mcp_status,
                    outputs=mcp_status_display,
                )

                gr.Markdown("### Statistics")
                stats_display = gr.Markdown("_Click refresh to load._")
                refresh_stats_btn = gr.Button(" Refresh Stats")
                refresh_stats_btn.click(
                    fn=fetch_stats,
                    outputs=stats_display,
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
