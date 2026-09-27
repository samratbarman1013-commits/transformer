# Transformer

A small, open-source, end-to-end LLM assistant: a GPT-style language model trained from
scratch, a tool-calling agent loop, an OpenAI-compatible API server, and a
privacy-first web/PWA client.

**Live app:** https://samratbarman1013-commits.github.io/transformer/

[![CI](https://github.com/samratbarman1013-commits/transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/samratbarman1013-commits/transformer/actions/workflows/ci.yml)
[![Deploy](https://github.com/samratbarman1013-commits/transformer/actions/workflows/pages.yml/badge.svg)](https://github.com/samratbarman1013-commits/transformer/actions/workflows/pages.yml)

**Status: Phase 1 (prototype).** The 1M-parameter prototype exists to validate the full
pipeline — architecture, training loop, tool system, apps — not to be smart yet.

## Quickstart

```bash
# 1. Install
pip install -e .[dev]

# 2. Smoke-test the training loop (overfits a tiny file in ~a minute on CPU)
python -m model.train --config config/1m_prototype.yaml --data data/smoke.txt --smoke

# 3. Generate text from the smoke checkpoint
python -m model.sample --config config/1m_prototype.yaml --prompt "once upon a time"

# 4. Serve the OpenAI-compatible API (demo backend until a real checkpoint exists)
uvicorn server.app:app --reload

# 5. Open the web client — live at
#    https://samratbarman1013-commits.github.io/transformer/ (or open app/web/index.html)
```

## Repository layout

```
config/        YAML experiment configs — scaling is a config change, not a rewrite
model/         GPT-style decoder-only transformer, tokenizer, training & sampling
server/        FastAPI app exposing an OpenAI-compatible /v1/chat/completions API
tools/         Typed tool registry: web_search, code_runner, memory
app/web/       PWA chat client (streaming, on-device history, light/dark)
app/android/   Notes & plan for the Android APK (WebView wrapper)
docs/          Roadmap, architecture, privacy promise, 1M -> 500M scaling plan
tests/         pytest suite (runs in CI)
data/          Corpora. data/smoke.txt is a tiny built-in overfit target.
```

## Config-driven scaling

| Config | Params | Layers | Hidden | Context | Vocab | Purpose |
|---|---|---|---|---|---|---|
| `1m_prototype` | ~1.0M | 4 | 128 | 256 | 2,048 | validate pipeline end-to-end |
| `500m` | ~500M | 30 | 1152 | 1024 | 32,768 | the real assistant (see `docs/SCALING.md`) |

Both configs run through the same `model/` code. The only differences are numbers
(and training data/compute — see `docs/SCALING.md` for the honest requirements).

> Note: the spec suggested hidden dim 256-512 for the prototype, but with a small
> (2k) BPE vocabulary that lands at 3-4M params. To actually hit ~1M we use
> n_embd=128. The parameter budget wins; bump it in the config whenever you want.

## Privacy

All chat history is stored **only on the client device** (localStorage in the web
client). The server is stateless per request and stores nothing. See
[docs/PRIVACY.md](docs/PRIVACY.md).

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) — six phases, each with exit criteria.

## License

**Not chosen yet (deliberately).** Until a license is added, all rights are
reserved by the author. Leading candidates:

- **MIT** — short, permissive, most common for small LLM projects
- **Apache 2.0** — permissive + explicit patent grant, better if accepting contributions
- **GPL v3** — forces derivatives to stay open-source

Pick one, drop the corresponding text into `LICENSE`, and delete this section.

## Development

```bash
ruff check .     # lint (same as CI)
pytest -q        # tests (same as CI)
```
