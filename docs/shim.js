/* SPDX-License-Identifier: GPL-3.0-or-later
   Static (GitHub Pages) mode: answers the page's /api/* calls from a Pyodide worker instead of a server. */
(() => {
  const realFetch = window.fetch.bind(window);
  const worker = new Worker("worker.js?v=4e083f6593");
  const pending = new Map();
  let nextId = 1, isReady = false, fatal = null, onReady, onFail;
  const ready = new Promise((res, rej) => { onReady = res; onFail = rej; });
  ready.catch(() => {});

  const box = document.createElement("div");
  box.setAttribute("role", "status");
  box.style.cssText = "position:fixed;inset:0;z-index:50;display:grid;place-items:center;background:var(--surface-1);color:var(--text-1);text-align:center;padding:24px";
  const title = document.createElement("div"), sub = document.createElement("div");
  title.style.cssText = "font-size:18px;font-weight:650;margin-bottom:6px"; title.textContent = "Starting xrfit";
  sub.style.cssText = "color:var(--text-2);max-width:46ch"; sub.textContent = "Loading Python…";
  const note = document.createElement("div");
  note.style.cssText = "color:var(--text-3);font-size:12px;margin-top:14px;max-width:46ch";
  note.textContent = "The fitting code runs in your browser. Your spectra are not uploaded. The first start downloads about 30 MB; later visits use the browser cache.";
  const inner = document.createElement("div"); inner.append(title, sub, note); box.append(inner);
  document.addEventListener("DOMContentLoaded", () => document.body.append(box));

  worker.onmessage = ({ data }) => {
    if (data.type === "status") sub.textContent = data.text;
    else if (data.type === "ready") { isReady = true; box.remove(); onReady(); }
    else if (data.type === "fatal") {
      fatal = data.text; title.textContent = "Could not start xrfit";
      sub.textContent = data.text + " Check your connection (Python is loaded from cdn.jsdelivr.net) and reload the page.";
      note.hidden = true; onFail(new Error(data.text));
    } else if (data.type === "reply") {
      const p = pending.get(data.id); if (!p) return; pending.delete(data.id);
      p(new Response(data.status === 204 ? null : data.body, { status: data.status, headers: { "Content-Type": data.ctype } }));
    }
  };

  async function bodyBytes(b) {
    if (b == null) return null;
    if (typeof b === "string") return new TextEncoder().encode(b).buffer;
    if (b instanceof Blob) return await b.arrayBuffer();
    if (b instanceof ArrayBuffer) return b;
    return new Uint8Array(b).buffer;
  }

  window.fetch = async (input, opt = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (!url.startsWith("/api/")) return realFetch(input, opt);
    try { await ready; } catch { return new Response(JSON.stringify({ error: fatal || "xrfit could not start." }), { status: 500, headers: { "Content-Type": "application/json" } }); }
    const id = nextId++, headers = {};
    for (const [k, v] of Object.entries(opt.headers || {})) headers[k] = v;
    const body = await bodyBytes(opt.body);
    return new Promise(res => { pending.set(id, res); worker.postMessage({ id, method: opt.method || "GET", path: url, headers, body }, body ? [body] : []); });
  };
})();
