const threadId = "ui-" + Math.random().toString(36).slice(2, 10);
const $ = (id) => document.getElementById(id);
const FIELD_KEYS = [
  ["name", "Name"],
  ["email", "Email"],
  ["company", "Company"],
  ["use_case", "Use case"],
  ["budget", "Budget"],
  ["preferred_time", "Demo time"],
  ["site_url", "Website"],
];

function chatUrl() {
  if (window.API_BASE) return String(window.API_BASE).replace(/\/$/, "") + "/chat";
  return "/chat";
}

function renderLead(lead) {
  const data = lead || {};
  $("fields").innerHTML = FIELD_KEYS.map(([key, label]) => {
    const value = data[key];
    return `<li><span>${label}</span><span class="${value ? "" : "empty"}">${value || "—"}</span></li>`;
  }).join("");
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
    $("intentPill").textContent = "intent " + (data.intent || "—");
    renderLead(data.lead);
  } catch (err) {
    pending.textContent = "Could not reach the API. " + (err.message || err);
  }
}

async function ping() {
  try {
    const res = await fetch("/health");
    $("apiPill").textContent = res.ok ? "API live" : "API down";
    $("apiPill").className = "pill " + (res.ok ? "ok" : "bad");
  } catch {
    $("apiPill").textContent = "API down";
    $("apiPill").className = "pill bad";
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
renderLead({});
ping();
