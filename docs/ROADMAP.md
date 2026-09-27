# Roadmap

The prototype's job is to prove the architecture, training loop, tool system, and apps all
work — not to be smart yet.

## Phase 1 — Model (current)

Train a small decoder-only transformer (GPT-style), ~1M params, from scratch.

- [x] Config-driven model code (`model/`, `config/*.yaml`)
- [x] Training script with overfit smoke mode (`model/train.py --smoke`)
- [x] Small BPE tokenizer builder (`model/tokenizer.py`)
- [ ] Train on TinyStories (download into `data/corpus.txt`, see `data/README.md`)
- [ ] **Exit criterion:** generates coherent short text

## Phase 2 — Capabilities

1. Chat with context memory (multi-turn).
2. Online research via tool calling — the model learns the *format*; a router/orchestrator layer handles execution.
3. Programming assistance — code generation + sandboxed executor (`tools/code_runner.py`).
4. URL/document summarization, local memory store, multilingual input (optional, later).

## Phase 3 — Agent system

- [x] Tool manager: registry with typed inputs/outputs, JSON tool calls (`tools/registry.py`)
- [x] Agent loop: plan → tool call → observe → respond, max iterations, graceful fallback (`server/agent.py`)
- [x] Streaming responses (`server/app.py`, SSE)
- [ ] Safety guardrails beyond the prototype blocklist (real sandboxing, rate limits)
- [ ] Structured per-run logging

## Phase 4 — Server & deployment

- [x] FastAPI server with OpenAI-compatible `/v1/chat/completions`
- [x] CI (lint + tests) via GitHub Actions
- [ ] Host a demo (HF Spaces / Render / Fly.io)
- [ ] Quantized (int8/4-bit) model files for on-device use

## Phase 5 — Applications

- [x] Web chat UI (PWA: streaming, tool status, on-device history) — `app/web/`
- [ ] Android APK (WebView wrapper) — see `app/android/README.md`

## Phase 6 — App polish

- [x] Clean chat UI, light/dark, mobile-first
- [x] Privacy-first: history only on device; stateless server
- [x] Export/import chat history as JSON; settings panel; offline notice

## Deliverables, in order

1. [x] Repo skeleton + README + roadmap *(this)*
2. [ ] Working training script that overfits a tiny dataset *(code exists; validate on your machine)*
3. [x] Chat inference API + minimal web UI *(demo backend until the model is trained)*
4. [x] Tool system + search tool *(interface; needs an API key wired)*
5. [ ] Android APK
6. [ ] Scale-up plan for the 500M version *(written: `docs/SCALING.md`; execution later)*
