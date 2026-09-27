# Transformer web client (PWA)

**Live:** https://samratbarman1013-commits.github.io/transformer/

A static, dependency-free chat client, auto-deployed to GitHub Pages on every
push that touches `app/web/` (see `.github/workflows/pages.yml`).

## Modes

- **Demo mode** (default): no server needed. A local simulation exercises the
  full UI — streaming, tool status pill, multi-chat history, export/import —
  so the app is testable from day one.
- **API mode**: Settings ⚙ → paste your server address (default
  `http://localhost:8000`, served by `uvicorn server.app:app` from the repo
  root). Streams from the OpenAI-compatible `/v1/chat/completions` endpoint.

## Privacy

All conversations live in this browser's localStorage. Export (⬇) gives you
the raw JSON; deleting site data erases everything. The server you connect to
is stateless per request.

## Install as an app (Android / desktop)

Open the live URL, then: Chrome → menu → **Install app**. The manifest +
service worker make it a standalone, offline-capable installable app — this
is the PWA path described in `app/android/README.md`.

## Files

- `index.html` / `styles.css` / `app.js` — the app (no build step)
- `manifest.webmanifest` + `sw.js` — installability + offline shell
- `icons/` — generated app icons
