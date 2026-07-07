"""Unit tests for Gemini LLM adapter message and response handling."""

from __future__ import annotations

from types import SimpleNamespace

from app.agents.schemas import ChatMessage, MessageRole, UnifiedToolCall
from app.services.llm_factory import GeminiLLMClient


def test_to_gemini_contents_batches_tool_responses_as_user_role() -> None:
    client = object.__new__(GeminiLLMClient)
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is the remote work policy?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I will check the policy documents.",
            tool_calls=[
                UnifiedToolCall(id="call_1", name="retrieve_policy_context", arguments={"query": "remote work"}),
                UnifiedToolCall(id="call_2", name="retrieve_policy_context", arguments={"query": "hybrid work"}),
            ],
        ),
        ChatMessage(role=MessageRole.TOOL, content="Passage one", name="retrieve_policy_context", tool_call_id="call_1"),
        ChatMessage(role=MessageRole.TOOL, content="Passage two", name="retrieve_policy_context", tool_call_id="call_2"),
    ]

    _, contents = client._to_gemini_contents(messages)

    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[2]["role"] == "user"
    assert len(contents[2]["parts"]) == 2
    assert contents[2]["parts"][0]["function_response"]["response"]["content"] == "Passage one"


def test_from_gemini_response_uses_text_fallback_without_tool_calls() -> None:
    client = object.__new__(GeminiLLMClient)
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="Remote work is allowed up to three days per week.", function_call=None)]
                )
            )
        ],
        text="Remote work is allowed up to three days per week.",
    )

    result = client._from_gemini_response(response)

    assert result.tool_calls == []
    assert result.content == "Remote work is allowed up to three days per week."


def test_from_gemini_response_prefers_tool_calls_over_text() -> None:
    client = object.__new__(GeminiLLMClient)
    function_call = SimpleNamespace(name="retrieve_policy_context", args={"query": "pto"})
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="Checking policies.", function_call=None),
                        SimpleNamespace(text=None, function_call=function_call),
                    ]
                )
            )
        ],
        text="Checking policies.",
    )

    result = client._from_gemini_response(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "retrieve_policy_context"
    assert result.content == "Checking policies."
