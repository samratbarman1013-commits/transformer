/* Transformer web client.
 *
 * Two modes:
 *  - API mode:  set an API endpoint in Settings; streams from an
 *               OpenAI-compatible /v1/chat/completions (SSE).
 *  - Demo mode: no endpoint set; a local simulation so the UI is fully
 *               exercisable (streaming, tool pill, history) with no server.
 *
 * All state is on-device (localStorage). Nothing is sent anywhere unless an
 * API endpoint is explicitly configured.
 */
"use strict";

const STORE_KEY = "transformer.v1";

const state = {
  chats: [],          // [{id, title, created, messages:[{role, content}]}]
  activeId: null,
  settings: { api: "", temperature: 0.7, tools: true },
  busy: false,
};

/* ---------- persistence ---------- */

function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    state.chats = raw.chats || [];
    state.activeId = raw.activeId || null;
    state.settings = Object.assign(state.settings, raw.settings || {});
  } catch { /* corrupt storage: start fresh rather than crash */ }
}

function save() {
  localStorage.setItem(STORE_KEY, JSON.stringify({
    chats: state.chats, activeId: state.activeId, settings: state.settings,
  }));
}

/* ---------- tiny DOM helpers ---------- */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, ms);
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

/* ---------- chats ---------- */

function activeChat() { return state.chats.find((c) => c.id === state.activeId); }

function newChat() {
  const chat = { id: crypto.randomUUID(), title: "New chat", created: Date.now(), messages: [] };
  state.chats.unshift(chat);
  state.activeId = chat.id;
  save();
  renderSidebar();
  renderMessages();
}

function deleteChat(id, ev) {
  ev.stopPropagation();
  state.chats = state.chats.filter((c) => c.id !== id);
  if (state.activeId === id) state.activeId = state.chats[0]?.id || null;
  save();
  renderSidebar();
  renderMessages();
}

function renderSidebar() {
  const list = $("#chat-list");
  list.replaceChildren();
  for (const chat of state.chats) {
    const item = el("button", "chat-item" + (chat.id === state.activeId ? " active" : ""));
    item.textContent = chat.title;
    item.title = chat.title;
    const del = el("span", "del", "✕");
    del.addEventListener("click", (e) => deleteChat(chat.id, e));
    item.prepend(del);
    item.addEventListener("click", () => {
      state.activeId = chat.id;
      save();
      renderSidebar();
      renderMessages();
      closeSidebar();
    });
    list.append(item);
  }
}

function renderMessages() {
  const box = $("#messages");
  box.replaceChildren();
  const chat = activeChat();
  $("#chat-title").textContent = chat ? chat.title : "New chat";
  if (!chat || chat.messages.length === 0) {
    const w = el("div", "welcome");
    w.innerHTML =
      '<img src="icons/icon-512.png" alt="" class="welcome-logo" />' +
      "<h2>Hi, I'm Transformer</h2>" +
      "<p>An open-source assistant prototype — small model, big pipeline. " +
      "Everything you write stays on this device.</p>";
    box.append(w);
    return;
  }
  for (const m of chat.messages) box.append(renderMessage(m.role, m.content));
  box.scrollTop = box.scrollHeight;
}

function renderMessage(role, content) {
  const wrap = el("div", "msg " + role);
  const avatar = el("div", "avatar", role === "user" ? "You"[0] : "T");
  const bubble = el("div", "bubble");
  bubble.textContent = content;
  wrap.append(avatar, bubble);
  return wrap;
}

/* ---------- tool pill ---------- */

function setToolPill(shown, label) {
  $("#tool-status").hidden = !shown;
  if (label) $("#tool-label").textContent = label;
}
const TOOL_LABELS = { web_search: "searching the web…", run_python: "running code…", memory_get: "reading memory…", memory_set: "saving a preference…" };

/* ---------- send / stream ---------- */

async function send() {
  if (state.busy) return;
  const input = $("#input");
  const text = input.value.trim();
  if (!text || !navigator.onLine) {
    if (!navigator.onLine) toast("You're offline — can't send right now.");
    return;
  }

  let chat = activeChat();
  if (!chat) { newChat(); chat = activeChat(); }
  chat.messages.push({ role: "user", content: text });
  if (chat.messages.length === 1) chat.title = text.slice(0, 44) + (text.length > 44 ? "…" : "");
  input.value = "";
  autosize(input);
  save();
  renderSidebar();
  renderMessages();
  setBusy(true);

  const assistantBubble = addStreamingBubble();
  try {
    const full = await streamReply(chat, text, assistantBubble);
    chat.messages.push({ role: "assistant", content: full });
  } catch (err) {
    const msg = fullText(assistantBubble) + (fullText(assistantBubble) ? "\n\n" : "") +
      `(error: ${err.message})`;
    chat.messages.push({ role: "assistant", content: msg });
  }
  setToolPill(false);
  save();
  renderMessages();
  setBusy(false);
}

function addStreamingBubble() {
  const box = $("#messages");
  const wrap = el("div", "msg assistant");
  wrap.append(el("div", "avatar", "T"));
  const bubble = el("div", "bubble");
  const cursor = el("span", "cursor");
  bubble.append(cursor);
  wrap.append(bubble);
  box.append(wrap);
  box.scrollTop = box.scrollHeight;
  return bubble;
}

function fullText(bubble) {
  return bubble.textContent.replace(/\u2588$/, ""); // strip cursor glyph if any
}

async function streamReply(chat, text, bubble) {
  const { api, temperature, tools } = state.settings;

  if (!api) {
    return demoStream(chat, bubble, tools);
  }

  const res = await fetch(normalizeEndpoint(api), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "transformer",
      messages: chat.messages.map((m) => ({ role: m.role, content: m.content })),
      temperature,
      stream: true,
      tools_enabled: tools,
    }),
  });
  if (!res.ok) throw new Error(`server returned ${res.status}`);

  // Parse SSE: lines of "data: {...}" terminated by "data: [DONE]".
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "", full = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return full;
      try {
        const chunk = JSON.parse(payload);
        const delta = chunk.choices?.[0]?.delta;
        if (delta?.content) { full += delta.content; appendTo(bubble, delta.content); }
      } catch { /* keep-alive or partial line; skip */ }
    }
  }
  return full;
}

function normalizeEndpoint(api) {
  let base = api.trim().replace(/\/+$/, "");
  if (/\/chat\/completions$/.test(base)) return base;
  if (/\/v1$/.test(base)) return base + "/chat/completions";
  return base + "/v1/chat/completions";
}

function appendTo(bubble, text) {
  const cursor = bubble.querySelector(".cursor");
  if (cursor) cursor.before(document.createTextNode(text));
  else bubble.append(text);
  const box = $("#messages");
  if (box.scrollHeight - box.scrollTop - box.clientHeight < 140) box.scrollTop = box.scrollHeight;
}

/* ---------- demo mode (no server) ---------- */

async function demoStream(chat, bubble, tools) {
  const lastUser = [...chat.messages].reverse().find((m) => m.role === "user")?.content || "";

  // Show the tool pill briefly so the UI state is exercisable without a server.
  if (tools && /\b(search|find|look up|latest|news|weather|who is|what time)\b/i.test(lastUser)) {
    setToolPill(true, TOOL_LABELS.web_search);
    await sleep(1400);
    setToolPill(false);
  } else {
    setToolPill(true, "thinking…");
    await sleep(500);
    setToolPill(false);
  }

  const reply =
    "This is demo mode — no model is connected yet, so this reply is generated " +
    "locally to show the full experience: streaming, tool status, on-device history.\n\n" +
    `You said: “${lastUser}”\n\n` +
    "To talk to the real model:\n" +
    "1. Run the repo's server: uvicorn server.app:app\n" +
    "2. Open Settings (⚙) and paste the address (e.g. http://localhost:8000)\n\n" +
    "Your messages never leave this device unless you configure an endpoint.";

  let full = "";
  for (const word of reply.split(/(\s+)/)) {
    full += word;
    appendTo(bubble, word);
    await sleep(18);
  }
  return full;
}

/* ---------- settings ---------- */

function openSettings() {
  $("#set-api").value = state.settings.api;
  $("#set-temp").value = state.settings.temperature;
  $("#temp-out").value = state.settings.temperature;
  $("#set-tools").checked = state.settings.tools;
  $("#settings-dialog").showModal();
}

function saveSettings() {
  state.settings.api = $("#set-api").value.trim();
  state.settings.temperature = parseFloat($("#set-temp").value) || 0.7;
  state.settings.tools = $("#set-tools").checked;
  save();
  updateBackendBadge();
}

function updateBackendBadge() {
  const badge = $("#backend-badge");
  if (state.settings.api) {
    badge.textContent = "live api";
    badge.classList.add("live");
    badge.title = "Connected to: " + state.settings.api;
  } else {
    badge.textContent = "demo";
    badge.classList.remove("live");
    badge.title = "Demo backend — connect an API in Settings";
  }
}

/* ---------- export / import ---------- */

function exportChats() {
  const data = { app: "transformer", version: 1, exported: new Date().toISOString(), chats: state.chats };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `transformer-chats-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function importChats(file) {
  file.text().then((text) => {
    const data = JSON.parse(text);
    if (!Array.isArray(data.chats)) throw new Error("not a Transformer export");
    const existing = new Set(state.chats.map((c) => c.id));
    const added = data.chats.filter((c) => !existing.has(c.id));
    state.chats.push(...added);
    save();
    renderSidebar();
    toast(`Imported ${added.length} conversation${added.length === 1 ? "" : "s"}.`);
  }).catch(() => toast("Import failed — is that a Transformer export file?"));
}

/* ---------- theme ---------- */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("transformer.theme", theme);
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || "light";
  applyTheme(current === "dark" ? "light" : "dark");
}

/* ---------- misc ---------- */

function setBusy(busy) {
  state.busy = busy;
  $("#send-btn").disabled = busy;
  $("#input").disabled = busy;
  if (!busy) $("#input").focus();
}

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  document.querySelector("#sidebar-overlay")?.remove();
}

function toggleSidebar() {
  const sidebar = $("#sidebar");
  if (sidebar.classList.contains("open")) { closeSidebar(); return; }
  sidebar.classList.add("open");
  const overlay = el("div");
  overlay.id = "sidebar-overlay";
  overlay.addEventListener("click", closeSidebar);
  document.body.append(overlay);
}

/* ---------- PWA ---------- */

let deferredInstall = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredInstall = e;
  toast("This app can be installed — use your browser's 'Install app' option.", 4000);
});

if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

/* ---------- wire up ---------- */

function init() {
  load();
  const theme = localStorage.getItem("transformer.theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  applyTheme(theme);

  if (state.chats.length === 0) newChat(); else renderSidebar(), renderMessages();
  updateBackendBadge();

  $("#send-btn").addEventListener("click", send);
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $("#input").addEventListener("input", (e) => autosize(e.target));
  $("#new-chat").addEventListener("click", () => { newChat(); closeSidebar(); $("#input").focus(); });
  $("#menu-btn").addEventListener("click", toggleSidebar);
  $("#theme-btn").addEventListener("click", toggleTheme);
  $("#open-settings").addEventListener("click", openSettings);
  $("#set-temp").addEventListener("input", (e) => { $("#temp-out").value = e.target.value; });
  $("#settings-dialog").addEventListener("close", () => {
    if ($("#settings-dialog").returnValue === "ok") saveSettings();
  });
  $("#export-chats").addEventListener("click", exportChats);
  $("#import-chats").addEventListener("click", () => $("#import-file").click());
  $("#import-file").addEventListener("change", (e) => {
    if (e.target.files[0]) importChats(e.target.files[0]);
    e.target.value = "";
  });

  window.addEventListener("online", () => { $("#offline-banner").hidden = true; });
  window.addEventListener("offline", () => { $("#offline-banner").hidden = false; });
  $("#offline-banner").hidden = navigator.onLine;
}

init();
