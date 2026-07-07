"""Native while-loop agent orchestration for PolicyPulse."""

from __future__ import annotations

import time

from app.agents.schemas import (
    AgentRequest,
    AgentResponse,
    AgentState,
    ChatMessage,
    LLMClient,
    MessageRole,
)
from app.agents.tools import SYSTEM_TOOL_DEFINITIONS, execute_tool
from app.config import get_settings
from app.observability.tracer import (
    ExecutionTrace,
    ToolCallTrace,
    new_trace_id,
    utc_now_iso,
    write_trace,
)
from app.services.llm_factory import get_llm_client

SYSTEM_PROMPT = (
    "You are PolicyPulse, an enterprise policy copilot. "
    "For factual questions about company policies, call retrieve_policy_context before answering. "
    "Ground answers in retrieved passages and cite the source file and page when possible. "
    "If no relevant policy text is found, say so clearly."
)


def run_agent(
    request: AgentRequest | str,
    *,
    llm_client: LLMClient | None = None,
    write_log: bool = True,
) -> AgentResponse:
    """Execute the tool-calling loop until the LLM returns a final answer."""
    settings = get_settings()
    llm = llm_client or get_llm_client()
    query = request.query if isinstance(request, AgentRequest) else request

    state = AgentState(
        trace_id=new_trace_id(),
        query=query,
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=query),
        ],
    )

    started_at = time.perf_counter()
    answer = ""
    completion = None

    while state.iterations < settings.max_tool_iterations:
        state.iterations += 1
        completion = llm.complete(state.messages, tools=SYSTEM_TOOL_DEFINITIONS)

        if completion.raw_intent and state.raw_intent is None:
            state.raw_intent = completion.raw_intent
        elif completion.content and state.raw_intent is None and not completion.tool_calls:
            state.raw_intent = completion.content

        if not completion.tool_calls:
            answer = (completion.content or completion.raw_intent or "").strip()
            break

        state.messages.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=completion.content,
                tool_calls=completion.tool_calls,
            )
        )

        for tool_call in completion.tool_calls:
            tool_started_at = time.perf_counter()
            tool_result = execute_tool(tool_call.name, tool_call.arguments)
            tool_latency_ms = int((time.perf_counter() - tool_started_at) * 1000)

            state.tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                    "latency_ms": tool_latency_ms,
                }
            )
            state.retrieved_chunks.extend(tool_result.retrieved_chunks)

            state.messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=tool_result.output,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                )
            )
    else:
        final_content = None
        if completion is not None:
            final_content = completion.content or completion.raw_intent
        answer = (final_content or "").strip() or (
            "Unable to complete the request within the configured tool iteration limit."
        )

    if not answer and state.tool_calls:
        answer = (
            "I retrieved relevant policy passages but could not synthesize a final answer. "
            "Please retry your question."
        )

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    trace = ExecutionTrace(
        trace_id=state.trace_id,
        timestamp=utc_now_iso(),
        query=query,
        raw_intent=state.raw_intent,
        tool_calls=[
            ToolCallTrace(
                name=str(record["name"]),
                args=dict(record["args"]),
                latency_ms=int(record["latency_ms"]),
            )
            for record in state.tool_calls
        ],
        retrieved_chunks=state.retrieved_chunks,
        latency_ms=latency_ms,
        model_provider=llm.provider,
        model_name=llm.model_name,
    )

    if write_log:
        write_trace(trace)

    return AgentResponse(answer=answer, trace=trace)
