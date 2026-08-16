/* Chat front end: keeps the conversation, streams replies over SSE. */

const el = (id) => document.getElementById(id);

const messagesEl = el("messages");
// Kept so "New chat" can restore the landing state without reloading the page.
const emptyState = el("empty-state");
const form = el("composer-form");
const input = el("input");
const sendButton = el("send");
const stopButton = el("stop");
const modelChip = el("model-chip");
const settingsPanel = el("settings");
const settingsToggle = el("settings-toggle");

const controls = {
  temperature: { input: el("temperature"), out: el("temperature-out"), digits: 2 },
  top_p: { input: el("top-p"), out: el("top-p-out"), digits: 2 },
  top_k: { input: el("top-k"), out: el("top-k-out"), digits: 0 },
  max_new_tokens: { input: el("max-tokens"), out: el("max-tokens-out"), digits: 0 },
};

/** The conversation as the model sees it. */
let conversation = [];
let inFlight = null;

/* ---------- rendering ---------- */

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Minimal markdown: fenced code blocks and inline code. Everything else is
 * escaped and left as-is, which is the right amount of formatting for a model
 * this size and avoids pulling in a markdown library.
 */
function renderContent(text) {
  return text
    .split(/```/)
    .map((part, index) => {
      if (index % 2 === 1) {
        // Odd chunks are inside a fence. Drop the opening language tag line.
        return `<pre><code>${escapeHtml(part.replace(/^[^\n]*\n/, ""))}</code></pre>`;
      }
      return escapeHtml(part).replace(/`([^`\n]+)`/g, "<code>$1</code>");
    })
    .join("");
}

function atBottom() {
  return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/**
 * Append a turn and return handles to it.
 *
 * User turns read as a compact right-aligned card; the assistant answers as
 * plain full-width prose, which keeps long replies comfortable to read and
 * visually separates "what I asked" from "what it said" without heavy chrome.
 */
function addTurn(role, text) {
  if (emptyState) emptyState.remove();

  const turn = document.createElement("div");
  turn.className = "mx-auto max-w-3xl px-5 first:pt-6 last:pb-8";

  const bubble = document.createElement("div");

  if (role === "user") {
    turn.classList.add("flex", "justify-end", "pt-6");
    bubble.className =
      "prose-body max-w-[85%] rounded-2xl rounded-br-md bg-zinc-100 px-4 py-2.5 text-[15px] dark:bg-white/[0.07]";
  } else if (role === "error") {
    turn.classList.add("pt-4");
    bubble.className =
      "prose-body rounded-xl border border-red-300 px-4 py-2.5 text-[14px] text-red-600 dark:border-red-500/40 dark:text-red-400";
  } else {
    turn.classList.add("pt-5");
    bubble.className = "prose-body text-[15px] text-zinc-800 dark:text-zinc-200";
  }

  bubble.innerHTML = renderContent(text);
  turn.append(bubble);
  messagesEl.append(turn);
  scrollToBottom();
  return { turn, bubble };
}

/* ---------- streaming ---------- */

function settings() {
  return {
    temperature: Number(controls.temperature.input.value),
    top_p: Number(controls.top_p.input.value),
    top_k: Number(controls.top_k.input.value),
    max_new_tokens: Number(controls.max_new_tokens.input.value),
  };
}

function setBusy(busy) {
  sendButton.hidden = busy;
  sendButton.disabled = busy;
  stopButton.hidden = !busy;
  input.disabled = busy;
  if (!busy) input.focus();
}

async function send(text) {
  if (!text.trim() || inFlight) return;

  addTurn("user", text);
  conversation.push({ role: "user", content: text });

  const { turn, bubble } = addTurn("assistant", "");
  turn.classList.add("streaming");
  setBusy(true);

  const controller = new AbortController();
  inFlight = controller;
  const started = performance.now();
  let reply = "";
  let tokens = 0;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation, ...settings() }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.error || `server returned ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (!frame.startsWith("data: ")) continue;

        const event = JSON.parse(frame.slice(6));
        if (event.error) throw new Error(event.error);
        if (event.delta) {
          const wasAtBottom = atBottom();
          reply += event.delta;
          tokens += 1;
          bubble.innerHTML = renderContent(reply);
          if (wasAtBottom) scrollToBottom();
        }
      }
    }

    conversation.push({ role: "assistant", content: reply });

    const seconds = (performance.now() - started) / 1000;
    const meta = document.createElement("div");
    meta.className = "mt-2 font-mono text-[11px] text-zinc-400 dark:text-zinc-600";
    meta.textContent = `${tokens} tok · ${seconds.toFixed(1)}s · ${(tokens / seconds).toFixed(1)} tok/s`;
    turn.append(meta);
  } catch (error) {
    if (error.name === "AbortError") {
      // Keep whatever streamed in before the stop, so the turn stays coherent.
      if (reply) {
        conversation.push({ role: "assistant", content: reply });
      } else {
        turn.remove();
        conversation.pop();
      }
    } else {
      turn.remove();
      conversation.pop();
      addTurn("error", error.message);
    }
  } finally {
    turn.classList.remove("streaming");
    inFlight = null;
    setBusy(false);
  }
}

/* ---------- events ---------- */

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value;
  input.value = "";
  resizeInput();
  send(text);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

input.addEventListener("input", resizeInput);

stopButton.addEventListener("click", () => inFlight?.abort());

el("new-chat").addEventListener("click", () => {
  inFlight?.abort();
  conversation = [];
  messagesEl.replaceChildren(emptyState);
  input.focus();
});

settingsToggle.addEventListener("click", () => {
  const open = settingsPanel.hidden;
  settingsPanel.hidden = !open;
  settingsToggle.setAttribute("aria-expanded", String(open));
  settingsToggle.classList.toggle("text-zinc-900", open);
  settingsToggle.classList.toggle("dark:text-white", open);
});

messagesEl.addEventListener("click", (event) => {
  const suggestion = event.target.closest(".suggestion");
  if (suggestion) send(suggestion.textContent);
});

/* ---------- boot ---------- */

function bindControl(key, value) {
  const { input: slider, out, digits } = controls[key];
  slider.value = value;
  const update = () => {
    out.textContent = Number(slider.value).toFixed(digits);
  };
  slider.addEventListener("input", update);
  update();
}

async function boot() {
  try {
    const info = await fetch("/api/info").then((response) => response.json());
    const model = info.model;
    const millions = (model.parameters / 1e6).toFixed(0);
    modelChip.textContent = `${millions}M · ${model.layers}L · ${model.context_length} ctx`;
    modelChip.title = JSON.stringify(model, null, 2);

    const defaults = info.defaults;
    controls.max_new_tokens.input.max = model.context_length;
    bindControl("temperature", defaults.temperature);
    bindControl("top_p", defaults.top_p);
    bindControl("top_k", defaults.top_k);
    bindControl("max_new_tokens", Math.min(defaults.max_new_tokens, model.context_length));
  } catch (error) {
    modelChip.textContent = "model info unavailable";
  }
  input.focus();
}

boot();
