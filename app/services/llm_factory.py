"""Provider-agnostic LLM client factory for the agent orchestrator."""

from __future__ import annotations

import json
from typing import Any

from app.agents.schemas import (
    ChatMessage,
    LLMClient,
    LLMCompletionResult,
    MessageRole,
    ToolDefinition,
    UnifiedToolCall,
)
from app.agents.tools import get_gemini_tool_declarations, get_openai_tool_schemas
from app.config import Settings, get_settings


class GeminiLLMClient:
    """Gemini chat client with normalized tool-call handling."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when MODEL_PROVIDER=gemini")

        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        self._genai = genai
        self.provider = "gemini"
        self.model_name = self.settings.resolved_model_name

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMCompletionResult:
        system_instruction, contents = self._to_gemini_contents(messages)
        tool_config = None
        if tools:
            tool_config = [{"function_declarations": get_gemini_tool_declarations()}]

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
            tools=tool_config,
        )
        response = model.generate_content(contents)
        return self._from_gemini_response(response)

    def _to_gemini_contents(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []
        index = 0

        while index < len(messages):
            message = messages[index]

            if message.role == MessageRole.SYSTEM:
                system_instruction = message.content
                index += 1
                continue

            if message.role == MessageRole.USER:
                contents.append({"role": "user", "parts": [message.content or ""]})
                index += 1
                continue

            if message.role == MessageRole.ASSISTANT:
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append(message.content)
                for tool_call in message.tool_calls:
                    parts.append(
                        {
                            "function_call": {
                                "name": tool_call.name,
                                "args": tool_call.arguments,
                            }
                        }
                    )
                contents.append({"role": "model", "parts": parts})
                index += 1
                continue

            if message.role == MessageRole.TOOL:
                # Gemini expects function responses in a single user turn.
                function_response_parts: list[dict[str, Any]] = []
                while index < len(messages) and messages[index].role == MessageRole.TOOL:
                    tool_message = messages[index]
                    function_response_parts.append(
                        {
                            "function_response": {
                                "name": tool_message.name or "tool",
                                "response": {"content": tool_message.content or ""},
                            }
                        }
                    )
                    index += 1
                contents.append({"role": "user", "parts": function_response_parts})
                continue

            index += 1

        return system_instruction, contents

    def _from_gemini_response(self, response: object) -> LLMCompletionResult:
        text_parts: list[str] = []
        tool_calls: list[UnifiedToolCall] = []

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None

            if parts:
                for part_index, part in enumerate(parts):
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

                    function_call = getattr(part, "function_call", None)
                    if function_call is not None and getattr(function_call, "name", None):
                        args = dict(function_call.args) if function_call.args else {}
                        tool_calls.append(
                            UnifiedToolCall(
                                id=f"call_{part_index}",
                                name=function_call.name,
                                arguments=args,
                            )
                        )

        if not tool_calls:
            aggregated_text = self._extract_gemini_text(response)
            if aggregated_text:
                text_parts = [aggregated_text]

        content = "\n".join(text_parts).strip() or None
        return LLMCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_intent=content,
        )

    @staticmethod
    def _extract_gemini_text(response: object) -> str | None:
        """Use the SDK text aggregator when part-level parsing returns no text."""
        try:
            text = getattr(response, "text", None)
        except ValueError:
            return None

        if not text:
            return None
        return str(text).strip() or None


class OpenAILLMClient:
    """OpenAI chat client with normalized tool-call handling."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")

        from openai import OpenAI

        self._client = OpenAI(api_key=self.settings.openai_api_key)
        self.provider = "openai"
        self.model_name = self.settings.resolved_model_name

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMCompletionResult:
        payload = {
            "model": self.model_name,
            "messages": self._to_openai_messages(messages),
        }
        if tools:
            payload["tools"] = get_openai_tool_schemas()

        response = self._client.chat.completions.create(**payload)
        return self._from_openai_response(response)

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                converted.append({"role": "system", "content": message.content or ""})
                continue

            if message.role == MessageRole.USER:
                converted.append({"role": "user", "content": message.content or ""})
                continue

            if message.role == MessageRole.ASSISTANT:
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments),
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
                converted.append(entry)
                continue

            if message.role == MessageRole.TOOL:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content or "",
                    }
                )

        return converted

    def _from_openai_response(self, response: object) -> LLMCompletionResult:
        message = response.choices[0].message
        tool_calls: list[UnifiedToolCall] = []

        for tool_call in message.tool_calls or []:
            raw_args = tool_call.function.arguments or "{}"
            tool_calls.append(
                UnifiedToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(raw_args),
                )
            )

        content = message.content
        return LLMCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_intent=content,
        )


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create the configured LLM client for the active provider."""
    settings = settings or get_settings()

    if settings.model_provider == "gemini":
        return GeminiLLMClient(settings=settings)
    if settings.model_provider == "openai":
        return OpenAILLMClient(settings=settings)

    raise ValueError(f"Unsupported MODEL_PROVIDER: {settings.model_provider}")
