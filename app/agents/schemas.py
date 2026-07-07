"""Unified agent schemas and LLM message contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.observability.tracer import ExecutionTrace, RetrievedChunkTrace


class MessageRole(str, Enum):
    """Normalized chat roles used inside the orchestrator."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class UnifiedToolCall(BaseModel):
    """Provider-agnostic tool invocation emitted by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A single message in the orchestrator conversation history."""

    role: MessageRole
    content: str | None = None
    tool_calls: list[UnifiedToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinition(BaseModel):
    """JSON-schema tool definition exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class LLMCompletionResult(BaseModel):
    """Normalized response from any LLM provider."""

    content: str | None = None
    tool_calls: list[UnifiedToolCall] = Field(default_factory=list)
    raw_intent: str | None = None


class ToolExecutionResult(BaseModel):
    """Result returned by a Python tool implementation."""

    output: str
    retrieved_chunks: list[RetrievedChunkTrace] = Field(default_factory=list)


class AgentRequest(BaseModel):
    """Inbound request to the agent orchestrator."""

    query: str = Field(min_length=1)


class AgentResponse(BaseModel):
    """Final agent answer plus the execution trace."""

    answer: str
    trace: ExecutionTrace


class AgentState(BaseModel):
    """Mutable state tracked while the orchestrator loop runs."""

    trace_id: str
    query: str
    messages: list[ChatMessage] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkTrace] = Field(default_factory=list)
    raw_intent: str | None = None
    iterations: int = 0


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat + tool-calling interface implemented by provider adapters."""

    provider: str
    model_name: str

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMCompletionResult:
        """Run one LLM turn and return normalized content and tool calls."""
