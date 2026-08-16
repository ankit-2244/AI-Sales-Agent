const LIVE_API = "https://ai-sales-agent.fastapicloud.dev";
const threadId = "ui-" + Math.random().toString(36).slice(2, 10);
const $ = (id) => document.getElementById(id);

$("threadPill").textContent = "thread " + threadId;

function chatUrl() {
  if (window.API_BASE) return String(window.API_BASE).replace(/\/$/, "") + "/chat";
  if (location.port === "5500" || location.port === "8000" || location.pathname.startsWith("/ui")) return "/chat";
  if (location.hostname.includes("netlify")) return "/chat";
  return LIVE_API + "/chat";
}

function healthUrl() {
  if (window.API_BASE) return String(window.API_BASE).replace(/\/$/, "") + "/health";
  if (chatUrl().startsWith("/")) return "/health";
  return LIVE_API + "/health";
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = role;
  el.textContent = text;
  $("msgs").appendChild(el);
  $("msgs").scrollTop = $("msgs").scrollHeight;
}

async function send(text) {
  const message = String(text || $("q").value).trim();
  if (!message) return;
  $("q").value = "";
  addMsg("user", message);
  const pending = document.createElement("div");
  pending.className = "bot";
  pending.textContent = "Thinking…";
  $("msgs").appendChild(pending);

  try {
    const res = await fetch(chatUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || res.statusText);
    pending.textContent = data.answer || "No answer returned.";
  } catch (err) {
    pending.textContent = "Could not reach the API. " + (err.message || err);
  }
}

async function ping() {
  const pill = $("apiPill");
  try {
    const res = await fetch(healthUrl());
    const data = await res.json();
    if (res.ok && (data.status === "healthy" || data.message)) {
      pill.textContent = "API live";
      pill.className = "pill ok";
      return;
    }
    throw new Error("unhealthy");
  } catch {
    pill.textContent = "API unreachable";
    pill.className = "pill bad";
  }
}

$("sendBtn").onclick = () => send();
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
document.querySelectorAll(".chip").forEach((btn) => {
  btn.onclick = () => send(btn.dataset.q);
});

const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
let listening = false;

function micNote(on, msg) {
  listening = on;
  $("micBtn").classList.toggle("live", on);
  $("micBtn").textContent = on ? "Stop" : "Mic";
  $("listenBar").classList.toggle("on", Boolean(msg) || on);
  $("listenBar").textContent = msg || (on ? "Listening…" : "");
}

function micBlockedReason() {
  const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
  if (location.protocol === "file:") return "Open this UI via the server, not a file.";
  if (!window.isSecureContext || (isSafari && location.hostname === "127.0.0.1")) {
    return "Use http://localhost:8000/ui/ (not 127.0.0.1) so the mic is allowed.";
  }
  return "";
}

$("micBtn").onclick = () => {
  const blocked = micBlockedReason();
  if (blocked) {
    micNote(false, blocked);
    return;
  }
  if (!SpeechRec) {
    micNote(false, "This browser has no speech recognition. Use Chrome.");
    return;
  }
  if (listening) {
    try { rec && rec.stop(); } catch { /* ignore */ }
    micNote(false);
    return;
  }
  rec = new SpeechRec();
  rec.lang = "en-US";
  rec.interimResults = true;
  rec.continuous = false;
  rec.onresult = (ev) => {
    let finalText = "";
    let interim = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const t = ev.results[i][0].transcript;
      if (ev.results[i].isFinal) finalText += t;
      else interim += t;
    }
    $("q").value = (finalText || interim).trim();
    if (finalText.trim()) {
      try { rec.stop(); } catch { /* ignore */ }
      micNote(false, "Heard: “" + finalText.trim() + "”");
      send(finalText.trim());
    }
  };
  rec.onerror = (ev) => micNote(false, ev.error === "not-allowed" ? "Allow the microphone for this site." : ev.error);
  rec.onend = () => { if (listening) micNote(false); };
  rec.start();
  micNote(true);
};

{
  const reason = micBlockedReason();
  if (reason) micNote(false, reason);
}
ping();
