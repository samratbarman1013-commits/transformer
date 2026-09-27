"""The agent loop: plan -> tool call -> observe -> respond.

Backends produce plain text (the model's job). This module owns the tool
protocol: the system prompt advertises available tools, model output that
contains a fenced ```tool block is parsed as a JSON tool call, executed via the
registry, and the result is appended as an observation for the next iteration.
Hard caps everywhere: max iterations, tool timeouts, and a graceful fallback
message when a tool fails.
"""
from __future__ import annotations

import datetime
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

from tools.registry import ToolRegistry

from .schemas import ChatMessage

MAX_ITERATIONS = 4

TOOL_BLOCK_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.DOTALL)

SYSTEM_PROMPT_TEMPLATE = """You are Transformer, a helpful assistant. Today: {date}.

You can call tools. To call one, emit exactly:

```tool
{{"name": "<tool>", "arguments": {{...}}}}
```

After a tool call you will receive an observation starting with [tool_result].
Use it to answer the user. If a tool fails, say so plainly and answer from what you know.

Available tools:
{tools}"""


@dataclass
class AgentEvent:
    """What the agent emits as it runs — either streamed text or a tool status."""

    type: str  # "delta" (assistant text) | "tool_status" (a tool is running)
    text: str = ""
    tool: str = ""


class DemoBackend:
    """Streams a canned response until a real model checkpoint exists."""

    name = "transformer-demo"

    def generate(self, messages, temperature: float, max_tokens: int) -> Iterator[str]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        reply = (
            "Hi! I'm the Transformer demo backend — the 1M-parameter model hasn't been "
            "trained yet, so this is a placeholder streaming response so you can see the "
            "full pipeline (API, streaming, agent loop, apps) working end to end.\n\n"
            f"You said: \"{last_user}\"\n\n"
            "Train the model with `python -m model.train --config config/1m_prototype.yaml "
            "--data <corpus.txt>` and restart the server to talk to the real thing."
        )
        for word in reply.split(" "):
            yield word + " "


class ModelBackend:
    """Loads a trained checkpoint and generates with it."""

    name = "transformer-1m"

    def __init__(self, ckpt_dir: str) -> None:
        import torch  # noqa: F401 (fail loudly with a clear error below if missing)

        from model.sample import load_model
        from model.tokenizer import load_tokenizer

        ckpt_path, tok_path = f"{ckpt_dir}/ckpt.pt", f"{ckpt_dir}/tokenizer.json"
        self.model, _ = load_model(__import__("pathlib").Path(ckpt_path))
        self.tok = load_tokenizer(__import__("pathlib").Path(tok_path))

    def _chat_context(self, messages) -> list:
        out = []
        for m in messages:
            label = "User" if m.role == "user" else "Assistant" if m.role == "assistant" else "System"
            out.append(f"{label}: {m.content}")
        out.append("Assistant:")
        return "\n".join(out)

    def generate(self, messages, temperature: float, max_tokens: int) -> Iterator[str]:
        import torch

        ids = self.tok.encode(self._chat_context(messages)).ids
        x = torch.tensor([ids], dtype=torch.long)
        y = self.model.generate(x, max_tokens, temperature=temperature, top_k=50)
        for token_id in y[0, len(ids) :].tolist():
            piece = self.tok.decode([token_id])
            yield piece
            if "User:" in piece or "System:" in piece:
                return


def run_agent(
    backend,
    messages,
    registry: ToolRegistry,
    tools_enabled: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Iterator[AgentEvent]:
    """Run the plan -> tool call -> observe -> respond loop, streaming events."""
    system = SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.datetime.now(tz=datetime.UTC).date().isoformat(),
        tools=registry.specs_json() if tools_enabled else "(none)",
    )
    conversation = [ChatMessage(role="system", content=system)] + list(messages)

    for _iteration in range(MAX_ITERATIONS):
        reply = ""
        for piece in backend.generate(conversation, temperature, max_tokens):
            reply += piece
            yield AgentEvent(type="delta", text=piece)

        # Does the reply contain a tool call? (only check the whole reply, once assembled)
        match = TOOL_BLOCK_RE.search(reply)
        if not match or not tools_enabled:
            return  # plain answer, we're done

        try:
            call = json.loads(match.group(1))
            name, arguments = call["name"], call.get("arguments", {})
        except (json.JSONDecodeError, KeyError):
            yield AgentEvent(type="delta", text="\n\n(I produced a malformed tool call; ignoring it.)")
            return

        yield AgentEvent(type="tool_status", tool=name)
        result = registry.execute(name, arguments)
        observation = json.dumps(result.data if result.ok else {"error": result.error})

        # Record the model's tool call and its observation, then iterate.
        conversation.append(ChatMessage(role="assistant", content=reply))
        conversation.append(ChatMessage(role="user", content=f"[tool_result] {observation[:4000]}"))

    yield AgentEvent(type="delta", text="\n\n(Stopped after reaching the maximum number of tool calls.)")
