"""FastAPI app: an OpenAI-compatible /v1/chat/completions endpoint.

Run:  uvicorn server.app:app --reload

Until a trained checkpoint exists (out/1m-prototype/ckpt.pt), the server
answers with the demo backend so the whole stack — streaming, agent loop,
tools, apps — is exercisable from day one.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import DemoBackend, ModelBackend, run_agent
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    StreamChoice,
    Delta,
    Usage,
)
from tools.registry import default_registry

app = FastAPI(title="Transformer", version="0.1.0")

# The web client is a static PWA that may be served from anywhere (Netlify,
# localhost, a file). Allow all origins: the server is stateless and stores
# nothing, so there is no session data to leak.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = default_registry()

# Backend selection: a trained checkpoint wins; otherwise the demo backend.
_CKPT_DIR = os.environ.get("TRANSFORMER_CKPT_DIR", "out/1m-prototype")
try:
    backend = ModelBackend(_CKPT_DIR)
except Exception:  # no checkpoint (or torch missing) -> demo mode, not a crash
    backend = DemoBackend()


def _completion_id() -> str:
    return f"chatcmpl-{os.urandom(6).hex()}"


@app.get("/health")
def health():
    return {"status": "ok", "backend": backend.name, "tools": [s.name for s in registry.specs()]}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": backend.name, "object": "model", "owned_by": "transformer"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(_stream_reply(req), media_type="text/event-stream")
    return _reply(req)


def _reply(req: ChatCompletionRequest) -> ChatCompletionResponse:
    text_parts: list[str] = []
    for event in run_agent(backend, req.messages, registry, req.tools_enabled, req.temperature, req.max_tokens):
        if event.type == "delta":
            text_parts.append(event.text)
    text = "".join(text_parts)
    return ChatCompletionResponse(
        id=_completion_id(),
        model=backend.name,
        choices=[Choice(message=ChoiceMessage(content=text))],
        usage=Usage(completion_tokens=len(text.split())),
    )


def _stream_reply(req: ChatCompletionRequest):
    chunk_id = _completion_id()

    def sse(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    first = ChatCompletionChunk(
        id=chunk_id, model=backend.name, choices=[StreamChoice(delta=Delta(role="assistant"))]
    )
    yield sse(first)

    for event in run_agent(backend, req.messages, registry, req.tools_enabled, req.temperature, req.max_tokens):
        if event.type == "delta":
            chunk = ChatCompletionChunk(id=chunk_id, model=backend.name, choices=[StreamChoice(delta=Delta(content=event.text))])
            yield sse(chunk)

    last = ChatCompletionChunk(id=chunk_id, model=backend.name, choices=[StreamChoice(delta=Delta(), finish_reason="stop")])
    yield sse(last)
    yield "data: [DONE]\n\n"
