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

        from google import genai

        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        self.provider = "gemini"
        self.model_name = self.settings.resolved_model_name

    def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMCompletionResult:
        from google.genai import types
        
        system_instruction, contents = self._to_gemini_contents(messages)
        
        config_dict: dict[str, Any] = {}
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
        if tools:
            config_dict["tools"] = [{"function_declarations": get_gemini_tool_declarations()}]
        
        config = types.GenerateContentConfig(**config_dict) if config_dict else None
        
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return self._from_gemini_response(response)

    def _to_gemini_contents(self, messages: list[ChatMessage]) -> tuple[str | None, list[Any]]:
        """Convert ChatMessages to new SDK format.
        
        The new SDK automatically handles thought_signature when we pass
        Content objects back unchanged.
        """
        system_instruction: str | None = None
        contents: list[Any] = []
        index = 0

        while index < len(messages):
            message = messages[index]

            if message.role == MessageRole.SYSTEM:
                system_instruction = message.content
                index += 1
                continue

            if message.role == MessageRole.USER:
                contents.append({"role": "user", "parts": [{"text": message.content or ""}]})
                index += 1
                continue

            if message.role == MessageRole.ASSISTANT:
                if message.gemini_raw_content:
                    # Pass the SDK's Content object back unchanged
                    # This preserves thought_signature automatically
                    contents.append(message.gemini_raw_content)
                    index += 1
                    continue

                # Fallback for text-only responses (no tool calls)
                if message.content:
                    contents.append({"role": "model", "parts": [{"text": message.content}]})
                index += 1
                continue

            if message.role == MessageRole.TOOL:
                # Gemini expects function responses in a single user turn
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
        """Parse new google-genai SDK response.
        
        The new SDK automatically preserves thought_signature when we store
        response.candidates[0].content and pass it back in the next request.
        """
        text_parts: list[str] = []
        tool_calls: list[UnifiedToolCall] = []
        raw_content = None

        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            
            # Store the entire Content object for thought_signature preservation
            if content is not None:
                raw_content = content
            
            parts = getattr(content, "parts", None) if content is not None else None

            if parts:
                for part_index, part in enumerate(parts):
                    # Extract text
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

                    # Extract tool calls
                    function_call = getattr(part, "function_call", None)
                    if function_call is not None:
                        name = getattr(function_call, "name", None)
                        if name:
                            args = getattr(function_call, "args", {})
                            # Convert args to dict if needed
                            if hasattr(args, "items"):
                                args = dict(args)
                            elif not isinstance(args, dict):
                                args = {}
                            
                            tool_calls.append(
                                UnifiedToolCall(
                                    id=f"call_{part_index}",
                                    name=name,
                                    arguments=args,
                                )
                            )

        # For text-only responses, don't store raw_content
        if not tool_calls:
            text = getattr(response, "text", None)
            if text:
                text_parts = [text]
            raw_content = None

        content = "\n".join(text_parts).strip() or None
        return LLMCompletionResult(
            content=content,
            tool_calls=tool_calls,
            raw_intent=content,
            gemini_raw_content=raw_content if tool_calls else None,
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
