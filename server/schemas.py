"""OpenAI-compatible request/response models.

These mirror the shapes of api.openai.com/v1/chat/completions closely enough
that standard OpenAI clients (and the web app) work unmodified.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "transformer"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False
    tools_enabled: bool = True
    # Accepted for compatibility, currently ignored by the demo backend:
    tools: Optional[list] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Usage()


# --- streaming shapes ---


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]

    @staticmethod
    def start(model: str) -> "ChatCompletionChunk":
        return ChatCompletionChunk(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=model,
            choices=[StreamChoice(delta=Delta(role="assistant"))],
        )
