"""Unit tests for the agent orchestrator loop."""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.orchestrator import run_agent
from app.agents.schemas import (
    ChatMessage,
    LLMClient,
    LLMCompletionResult,
    MessageRole,
    ToolDefinition,
    UnifiedToolCall,
)
from app.agents.tools import SYSTEM_TOOL_DEFINITIONS, execute_tool, get_gemini_tool_declarations
from app.observability.tracer import RetrievedChunkTrace


class ScriptedLLMClient:
    """Deterministic LLM stub that emits one tool call then a final answer."""

    provider = "test"
    model_name = "scripted-llm"

    def __init__(self, script: list[LLMCompletionResult]) -> None:
        self._script = script
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMCompletionResult:
        self.calls.append(messages)
        if not self._script:
            raise AssertionError("ScriptedLLMClient ran out of scripted responses.")
        return self._script.pop(0)


def test_gemini_tool_declarations_exclude_unsupported_schema_fields() -> None:
    declarations = get_gemini_tool_declarations()
    retrieve_tool = next(item for item in declarations if item["name"] == "retrieve_policy_context")
    parameters = retrieve_tool["parameters"]

    assert parameters["type"] == "object"
    assert "query" in parameters["properties"]
    assert "top_k" in parameters["properties"]
    assert parameters["required"] == ["query"]

    for property_schema in parameters["properties"].values():
        assert "default" not in property_schema
        assert set(property_schema.keys()).issubset({"type", "description", "enum", "items", "properties", "required"})

    canonical = SYSTEM_TOOL_DEFINITIONS[0].parameters
    assert "default" not in str(canonical)


def test_execute_tool_retrieve_policy_context(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunks = [
        RetrievedChunkTrace(
            source="handbook.pdf",
            page=2,
            text="Employees may work remotely up to three days per week.",
            score=0.91,
        )
    ]

    class FakeStore:
        def search(self, query: str, top_k: int = 4, embedder: Any = None) -> list[RetrievedChunkTrace]:
            assert query == "remote work"
            return fake_chunks

    monkeypatch.setattr("app.agents.tools.get_policy_index_store", lambda: FakeStore())

    result = execute_tool("retrieve_policy_context", {"query": "remote work", "top_k": 2})

    assert "handbook.pdf" in result.output
    assert result.retrieved_chunks == fake_chunks


def test_orchestrator_runs_tool_loop_then_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunks = [
        RetrievedChunkTrace(
            source="handbook.pdf",
            page=2,
            text="Employees may work remotely up to three days per week.",
            score=0.91,
        )
    ]

    class FakeStore:
        def search(self, query: str, top_k: int = 4, embedder: Any = None) -> list[RetrievedChunkTrace]:
            return fake_chunks

    monkeypatch.setattr("app.agents.tools.get_policy_index_store", lambda: FakeStore())
    monkeypatch.setattr("app.agents.orchestrator.write_trace", lambda trace: None)

    llm = ScriptedLLMClient(
        [
            LLMCompletionResult(
                content="I will search the policy documents.",
                raw_intent="policy_lookup",
                tool_calls=[
                    UnifiedToolCall(
                        id="call_1",
                        name="retrieve_policy_context",
                        arguments={"query": "remote work policy", "top_k": 2},
                    )
                ],
            ),
            LLMCompletionResult(
                content="Remote work is allowed up to three days per week (handbook.pdf, page 2).",
            ),
        ]
    )

    response = run_agent("What is the remote work policy?", llm_client=llm, write_log=False)

    assert len(llm.calls) == 2
    assert llm.calls[1][-1].role == MessageRole.TOOL
    assert "handbook.pdf" in llm.calls[1][-1].content
    assert "three days per week" in response.answer
    assert response.trace.raw_intent == "policy_lookup"
    assert len(response.trace.tool_calls) == 1
    assert response.trace.tool_calls[0].name == "retrieve_policy_context"
    assert len(response.trace.retrieved_chunks) == 1


def test_orchestrator_returns_direct_answer_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agents.orchestrator.write_trace", lambda trace: None)

    llm = ScriptedLLMClient(
        [LLMCompletionResult(content="Hello! How can I help with company policies today?")]
    )

    response = run_agent("Hi there", llm_client=llm, write_log=False)

    assert len(llm.calls) == 1
    assert response.answer.startswith("Hello!")
    assert response.trace.tool_calls == []
