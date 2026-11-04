# SPDX-License-Identifier: GPL-3.0-or-later
"""Browser user interface for xrfit.

    xrfit-web                  # serves http://127.0.0.1:8765 and opens it in your browser
    xrfit-web --port 9000 --no-browser

Standard library only (plus numpy/scipy for the fits). One user, one session: the loaded spectrum and
the last fit live in memory on the server. The server listens on 127.0.0.1 by default; it has no
authentication, so only pass --host 0.0.0.0 on a network you trust.
"""
import argparse
import base64
import csv
import dataclasses
import io
import json
import math
import os
import re
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import core, demo, io_ketek, lines, xdata

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
DATA_DIR = os.path.join(HERE, "data")
MAX_BODY = 256 * 1024 * 1024
CFG_FIELDS = {f.name for f in dataclasses.fields(core.Config)}
NUM = {"zero", "gain", "noise", "fano", "tail_frac", "beta", "step_frac", "esc_frac", "emin", "emax", "e0"}
INT = {"snip_width", "snip_smooth", "poly_order"}
BOOL = {"sum_peaks", "free_ratios"}
PARAMS = ["zero", "gain", "noise", "fano", "tail_frac", "beta", "step_frac", "esc_frac"]
BACKGROUNDS = ("snip", "poly", "snip+poly", "none")
SOURCES = ("xraylib", "krause")


class ApiError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.spec = None
        self.result = None


STATE = State()


# ------------------------------------------------------------------ helpers
def msg(e):
    """Readable text of an exception (KeyError would otherwise add quotes)."""
    return str(e.args[0]) if e.args else str(e)


def clean(o):
    """Make an object JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def cfg_from_json(d):
    cfg = core.Config()
    for k, v in (d or {}).items():
        if k not in CFG_FIELDS:
            continue
        if k in NUM:
            if v is None or v == "":
                if k == "emax":
                    v = 1e9
                elif k == "emin":
                    v = 0.0
                else:
                    continue
            setattr(cfg, k, float(v))
        elif k in INT:
            setattr(cfg, k, int(v))
        elif k in BOOL:
            setattr(cfg, k, bool(v))
        elif k == "elements":
            cfg.elements = [str(x) for x in v]
        elif k == "free":
            cfg.free = [x for x in v if x in PARAMS]
        elif k == "background":
            if v not in BACKGROUNDS:
                raise ApiError(f"background must be one of {', '.join(BACKGROUNDS)}")
            cfg.background = v
        elif k == "xs_source":
            if v not in SOURCES:
                raise ApiError("cross-section source must be xraylib or krause")
            cfg.xs_source = v
    if cfg.gain <= 0:
        raise ApiError("Gain must be greater than 0 keV/channel.")
    if cfg.e0 <= 0:
        raise ApiError("Excitation energy e0 must be greater than 0 keV.")
    if cfg.emax <= cfg.emin:
        raise ApiError("The fit region must have an upper energy above the lower one.")
    return cfg


def spectrum_json(spec, suggest=None):
    return {
        "name": spec.name, "n": int(len(spec.counts)), "total": float(spec.counts.sum()),
        "live_time": float(spec.live_time),
        "calib": list(spec.calib) if spec.calib else None,
        "counts": [int(c) if float(c).is_integer() else float(c) for c in spec.counts],
        "format": spec.meta.get("_format") if spec.meta else None,
        "meta": io_ketek.describe_groups(spec.meta),
        "suggest": suggest or {},
    }


def set_spectrum(spec, suggest=None):
    with STATE.lock:
        STATE.spec, STATE.result = spec, None
    sug = dict(suggest or {})
    if spec.calib and "zero" not in sug:
        sug["zero"], sug["gain"] = spec.calib
    return spectrum_json(spec, sug)


def load_bytes(name, data):
    suffix = re.sub(r"[^A-Za-z0-9.]", "_", os.path.splitext(name)[1])[:12] or ".txt"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            spec = core.load_spectrum(path)
        except Exception as e:
            raise ValueError(str(e).replace(path, name)) from None
    finally:
        os.unlink(path)
    spec.name = name
    return spec


def groups_info(tokens, free_ratios, e0, source):
    out = []
    for tok in tokens:
        try:
            sym, fam, line = lines.parse_token(tok, e0)
            groups = lines.build_groups([tok], free_ratios, e0, source)
            out.append({"token": tok, "ok": True, "element": sym, "groups": [
                {"name": n, "lines": [{"E": E, "w": w} for E, w in ls]} for n, ls in groups]})
        except (KeyError, ValueError) as e:
            out.append({"token": tok, "ok": False, "error": msg(e), "groups": []})
    return out


def result_json(res):
    comps = {k: np.round(v, 3) for k, v in res.components.items() if not k.startswith("bg:")}
    q = None
    try:
        q = core.quantify_theory(res)
    except Exception:  # theory quantification is optional information
        q = None
    areas = []
    for name, (a, s) in res.areas.items():
        row = {"name": name, "counts": a, "sigma": s, "cps": a / res.live_time}
        if q and name in q:
            row["atoms"], row["atoms_note"] = q[name]
        areas.append(row)
    cfg = dataclasses.asdict(res.cfg)
    if cfg["emax"] >= 1e8:
        cfg["emax"] = None
    return clean({
        "chi2red": res.chi2red, "n_points": len(res.energy),
        "region": [float(res.energy[0]), float(res.energy[-1])],
        "params": res.params, "param_err": res.param_err, "areas": areas,
        "energy": np.round(res.energy, 5), "data": res.data, "background": np.round(res.background, 3),
        "fit": np.round(res.fit, 3), "components": comps, "calib": {"zero": res.params["zero"], "gain": res.params["gain"]},
        "e0": res.cfg.e0, "xs_source": res.cfg.xs_source, "live_time": res.live_time, "cfg": cfg,
    })


def fit_csv(res):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["channel", "energy_keV", "data", "background", "fit", "residual"])
    for row in zip(res.channels, res.energy, res.data, res.background, res.fit, res.data - res.fit):
        w.writerow([f"{v:.6g}" for v in row])
    return buf.getvalue()


# ---------------------------------------------------------------- API calls
def api_session(body, query):
    with STATE.lock:
        return {"spectrum": spectrum_json(STATE.spec) if STATE.spec else None}


def api_elements(body, query):
    out = []
    for s in xdata.symbols():
        x = xdata.xl().get(s)
        out.append({"symbol": s, "Z": (xdata.t12().get(s) or x)["Z"], "table12": s in xdata.t12(),
                    "xraylib": bool(x), "k": bool(x and "xs_K" in x), "l": bool(x and "xs_L" in x)})
    return {"elements": out}


def api_example(body, query):
    name = (query.get("name") or [""])[0]
    if name == "synthetic":
        return set_spectrum(demo.synthetic_spectrum(), demo.SUGGEST)
    if name == "ketek":
        spec = core.load_spectrum(os.path.join(DATA_DIR, "example_ketek.vspc"))
        spec.name = "example-ketek.vspc"
        return set_spectrum(spec, {"elements": [], "emin": 0.0, "emax": None})
    raise ApiError("Unknown example.", 404)


def api_groups(body, query):
    e0 = float(body.get("e0", 20.0))
    return {"entries": groups_info(body.get("tokens", []), bool(body.get("free_ratios")), e0,
                                   body.get("xs_source", "xraylib"))}


def api_lines(body, query):
    sym = str(body.get("symbol", "")).strip()
    try:
        return lines.table(sym, float(body.get("e0", 20.0)), body.get("xs_source", "xraylib"))
    except KeyError as e:
        raise ApiError(f"Unknown element '{sym}'. Use a symbol such as Fe or Pb.") from e


def api_fit(body, query):
    with STATE.lock:
        if STATE.spec is None:
            raise ApiError("Open a spectrum before fitting.")
        cfg = cfg_from_json(body.get("config"))
        if not cfg.elements:
            raise ApiError("Add at least one element to fit.")
        try:
            res = core.fit(STATE.spec, cfg)
        except KeyError as e:
            raise ApiError(msg(e)) from e
        except ValueError as e:
            raise ApiError(str(e)) from e
        STATE.result = res
        return result_json(res)


def api_batch(body, query):
    cfg = cfg_from_json(body.get("config"))
    if not cfg.elements:
        raise ApiError("Add at least one element to fit.")
    files = body.get("files") or []
    if not files:
        raise ApiError("Choose at least one spectrum file.")
    specs = []
    for f in files:
        try:
            specs.append(load_bytes(f["name"], base64.b64decode(f["b64"])))
        except Exception as e:
            raise ApiError(f"Could not read {f.get('name', 'file')}: {e}") from e
    try:
        results = core.fit_batch(specs, cfg)
    except (KeyError, ValueError) as e:
        raise ApiError(msg(e)) from e
    names = []
    for r in results:
        for n in r.areas:
            if n not in names and not n.startswith("sum:"):
                names.append(n)
    rows = [{"name": s.name, "live_time": s.live_time, "chi2red": r.chi2red,
             "areas": {n: list(r.areas[n]) for n in names if n in r.areas}} for s, r in zip(specs, results)]
    return clean({"groups": names, "rows": rows})


POST_ROUTES = {"/api/groups": api_groups, "/api/lines": api_lines, "/api/fit": api_fit, "/api/batch": api_batch}
GET_ROUTES = {"/api/session": api_session, "/api/elements": api_elements, "/api/example": api_example}


# ------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "xrfit-web"
    allowed_hosts = None  # set by serve()

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, status, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, obj):
        self._send(status, json.dumps(clean(obj), allow_nan=False))

    def _host_ok(self):
        if self.allowed_hosts is None:
            return True
        return (self.headers.get("Host") or "").lower() in self.allowed_hosts

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, "Forbidden host", "text/plain")
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if u.path == "/api/fit.csv":
                with STATE.lock:
                    res = STATE.result
                if res is None:
                    raise ApiError("Run a fit first.", 404)
                return self._send(200, fit_csv(res), "text/csv; charset=utf-8",
                                  {"Content-Disposition": 'attachment; filename="xrfit-fit.csv"'})
            fn = GET_ROUTES.get(u.path)
            if fn:
                return self._json(200, fn(None, parse_qs(u.query)))
            self._send(404, "Not found", "text/plain")
        except ApiError as e:
            self._json(e.status, {"error": str(e)})
        except Exception as e:  # surface unexpected problems in the UI instead of a blank page
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, "Forbidden host", "text/plain")
        if self.headers.get("X-Requested-With") != "xrfit":  # blocks cross-site form posts
            return self._send(403, "Missing X-Requested-With header", "text/plain")
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ApiError("File is too large (limit 256 MB).", 413)
            raw = self.rfile.read(n)
            if u.path == "/api/load":
                name = os.path.basename(requote(self.headers.get("X-Filename", "spectrum.txt"))) or "spectrum.txt"
                try:
                    spec = load_bytes(name, raw)
                except Exception as e:
                    raise ApiError(f"Could not read {name}: {e}") from e
                return self._json(200, set_spectrum(spec))
            fn = POST_ROUTES.get(u.path)
            if not fn:
                return self._send(404, "Not found", "text/plain")
            body = json.loads(raw.decode("utf-8") or "{}")
            return self._json(200, fn(body, parse_qs(u.query)))
        except ApiError as e:
            self._json(e.status, {"error": str(e)})
        except json.JSONDecodeError:
            self._json(400, {"error": "Request body is not valid JSON."})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def requote(s):
    from urllib.parse import unquote
    return unquote(s)


def serve(host="127.0.0.1", port=8765, open_browser=True):
    last = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), Handler)
            break
        except OSError as e:
            last = e
    else:
        raise SystemExit(f"Could not bind a port in {port}-{port + 19}: {last}")
    port = httpd.server_address[1]
    if host in ("127.0.0.1", "localhost", "::1"):
        Handler.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    print(f"xrfit web UI: {url}   (Ctrl+C to stop)")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Warning: no authentication. Anyone who can reach this port can use the app and read loaded files.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xrfit-web", description="Browser interface for xrfit")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    a = ap.parse_args(argv)
    serve(a.host, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
