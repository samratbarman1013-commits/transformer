# Architecture

```
                       ┌─────────────────────────────────────────────┐
                       │                  SERVER                     │
                       │                                             │
  web app (PWA) ──▶    │  FastAPI  server/app.py                     │
  /app/web             │    │                                        │
                       │    ▼                                        │
                       │  Agent loop  server/agent.py                │
                       │    plan → tool call → observe → respond     │
                       │    │                        │               │
                       │    ▼                        ▼               │
                       │  Backend                 Tool registry      │
                       │  (model/ or demo)        tools/registry.py  │
                       │                          ├ web_search       │
                       │                          ├ run_python      │
                       │                          └ memory (local)  │
                       └─────────────────────────────────────────────┘
```

## Components

**model/** — a GPT-style decoder-only transformer written from scratch
(nanoGPT as reference, but every component explicit: token embedding →
N × [masked self-attention + MLP] → tied LM head). All hyperparameters live in
`config/*.yaml`. Scaling to 500M is a config change plus data and compute.

**Tokenizer** — a small ByteLevel BPE (2,048 tokens for the prototype) trained
on the corpus, so the vocabulary — and therefore the embedding budget — is ours
to choose.

**server/agent.py** — the agent loop. Backends produce plain text; the agent
owns the tool protocol. Tool calls are fenced JSON blocks
(``` ```tool {"name": ..., "arguments": ...} ``` ```), parsed and executed via
the registry; results come back as `[tool_result]` observations. Caps: max 4
iterations per request, tool timeouts, graceful fallback text on failure.

**tools/registry.py** — typed tool registry. Every tool declares a JSON schema
for its arguments, gets validated, and failures return a structured result
(an observation for the model) rather than crashing the request.

**server/app.py** — OpenAI-compatible API (`/v1/chat/completions`, SSE
streaming, `/v1/models`, `/health`). Stateless per request. Runs the demo
backend until `out/1m-prototype/ckpt.pt` exists, then the real model
automatically.

**app/web/** — static PWA client. Talks to the API over SSE; stores all
history on-device (localStorage); shows tool status ("searching…") while the
agent works. No server-side session state exists at all.

## Request lifecycle (chat)

1. Client POSTs messages to `/v1/chat/completions` (stream: true).
2. Server builds the system prompt (tool specs + date) and runs the agent.
3. Backend streams tokens; agent streams `delta` events; tool calls emit
   `tool_status` events before execution.
4. Server serializes everything as OpenAI-style SSE chunks + `[DONE]`.
5. Client renders token by token, keeps history in localStorage only.
