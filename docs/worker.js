/* SPDX-License-Identifier: GPL-3.0-or-later
   Runs the xrfit Python backend inside the browser with Pyodide (numpy + scipy compiled to WebAssembly).
   The page talks to it through shim.js, which answers the UI's /api/* calls. Nothing is uploaded anywhere. */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
importScripts(PYODIDE + "pyodide.js");

let call = null;
const ready = (async () => {
  postMessage({ type: "status", text: "Loading Python…" });
  const py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "status", text: "Loading numpy and scipy…" });
  await py.loadPackage(["numpy", "scipy"]);
  postMessage({ type: "status", text: "Loading xrfit…" });
  const zip = await (await fetch("xrfit-py.zip?v=f63fcbb3f5")).arrayBuffer();
  py.unpackArchive(zip, "zip");
  await py.runPythonAsync(`
import sys, json
if "" not in sys.path: sys.path.insert(0, "")
from pyodide.ffi import to_js
from js import Object
from xrfit import webapp

def _call(method, path, headers_json, raw):
    body = bytes(raw.to_py()) if raw is not None else b""
    status, ctype, data, extra = webapp.dispatch(method, path, json.loads(headers_json), body)
    return to_js({"status": status, "ctype": ctype, "body": data, "extra": extra}, dict_converter=Object.fromEntries)
`);
  call = py.globals.get("_call");
  postMessage({ type: "ready" });
})().catch(e => { postMessage({ type: "fatal", text: String(e && e.message || e) }); throw e; });

onmessage = async ({ data }) => {
  const { id, method, path, headers, body } = data;
  try {
    await ready;
    const r = call(method, path, JSON.stringify(headers || {}), body ? new Uint8Array(body) : null);
    postMessage({ type: "reply", id, status: r.status, ctype: r.ctype, body: r.body.buffer.slice(r.body.byteOffset, r.body.byteOffset + r.body.byteLength), extra: r.extra }, []);
  } catch (e) {
    postMessage({ type: "reply", id, status: 500, ctype: "application/json", body: new TextEncoder().encode(JSON.stringify({ error: String(e && e.message || e) })).buffer, extra: {} });
  }
};
