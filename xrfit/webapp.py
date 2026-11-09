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

from . import attenuation, calibration, core, demo, io_ketek, lines, matrix, tube, xdata

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
DATA_DIR = os.path.join(HERE, "data")
MAX_BODY = 256 * 1024 * 1024
CFG_FIELDS = {f.name for f in dataclasses.fields(core.Config)}
NUM = {"zero", "gain", "noise", "fano", "tail_frac", "beta", "step_frac", "esc_frac", "esc_scale", "tail2_frac", "beta2", "emin", "emax", "e0",
       "scatter_angle", "compton_broad"}
INT = {"snip_width", "snip_smooth", "poly_order"}
BOOL = {"sum_peaks", "free_ratios"}
PARAMS = ["zero", "gain", "noise", "fano", "tail_frac", "beta", "step_frac", "esc_frac", "esc_scale", "tail2_frac", "beta2", "compton_broad"]
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


def beam_from_json(rows):
    """Validate the incident-spectrum table sent by the UI."""
    if rows in (None, ""):
        return []
    if not isinstance(rows, list) or len(rows) > 500:
        raise ApiError("The beam table must be a list of at most 500 energies.")
    out = []
    for i, r in enumerate(rows, 1):
        try:
            E, w = float(r["energy"]), float(r.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            raise ApiError(f"Beam row {i}: enter an energy and a weight as numbers.") from None
        if not (math.isfinite(E) and math.isfinite(w)):
            raise ApiError(f"Beam row {i}: energy and weight must be finite numbers.")
        if not 0.05 <= E <= 500:
            raise ApiError(f"Beam row {i}: the energy must be between 0.05 and 500 keV.")
        if w < 0:
            raise ApiError(f"Beam row {i}: the weight cannot be negative.")
        out.append({"energy": E, "weight": w, "use": bool(r.get("use", True)), "scatter": bool(r.get("scatter", False))})
    return out


def filters_from_json(v):
    if v in (None, ""):
        return []
    if not isinstance(v, list) or len(v) > 20:
        raise ApiError("The filters must be a list of at most 20 entries.")
    flt = [{"material": str(f.get("material", "")), "density": f.get("density", ""), "thickness_um": f.get("thickness_um", "")}
           for f in v if isinstance(f, dict)]
    try:
        matrix.parse_filters(flt)
    except matrix.MatrixError as e:
        raise ApiError(str(e)) from e
    return flt


def exc_from_body(body):
    """Excitation (float or tuple of (E, w)) for the line-table endpoints."""
    rows = beam_from_json(body.get("beam"))
    cfg = core.Config(beam=rows, beam_filters=filters_from_json(body.get("beam_filters")))
    if rows:
        try:
            return core.excitation(cfg)
        except ValueError as e:
            raise ApiError(str(e)) from e
    return float(body.get("e0", 20.0))


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
        elif k == "beam":
            cfg.beam = beam_from_json(v)
        elif k == "matrix":
            cfg.matrix = v if isinstance(v, dict) else {}
        elif k == "beam_filters":
            cfg.beam_filters = filters_from_json(v)
        elif k == "elements":
            cfg.elements = [str(x) for x in v]
        elif k == "instrument":
            cfg.instrument = [str(x) for x in v]
        elif k == "free":
            cfg.free = [x for x in v if x in PARAMS]
        elif k == "background":
            if v not in BACKGROUNDS:
                raise ApiError(f"background must be one of {', '.join(BACKGROUNDS)}")
            cfg.background = v
        elif k == "argon":
            if v not in ("auto", "on", "off"):
                raise ApiError("argon must be auto, on or off")
            cfg.argon = v
        elif k == "xs_source":
            if v not in SOURCES:
                raise ApiError("cross-section source must be xraylib or krause")
            cfg.xs_source = v
    if cfg.gain <= 0:
        raise ApiError("Gain must be greater than 0 keV/channel.")
    if cfg.e0 <= 0:
        raise ApiError("Excitation energy e0 must be greater than 0 keV.")
    if not 0 < cfg.scatter_angle < 180:
        raise ApiError("The scattering angle must be between 0 and 180 degrees.")
    if cfg.beam:
        try:
            cfg.e0 = lines.top_energy(core.excitation(cfg))
        except ValueError as e:
            raise ApiError(str(e)) from e
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
    inst = core.instrument_groups(res.cfg)
    for name, (a, s) in res.areas.items():
        row = {"name": name, "counts": a, "sigma": s, "cps": a / res.live_time, "instrument": name in inst}
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
    if name in ("srm1157", "ketek"):
        spec = core.load_spectrum(os.path.join(DATA_DIR, "example_srm1157.vspc"))
        spec.name = "srm1157-tool-steel.vspc"
        tube_cfg = {"anode": "Rh", "kv": 30.0, "bins": 24, "line_share": 0.03}   # Rh tube at 30 kV. 3 % anode K lines: calibrated on the certified composition ("Fit the anode line share")
        beam = tube.tube_beam(tube_cfg["anode"], tube_cfg["kv"], tube_cfg["bins"], 1.5, tube_cfg["line_share"])
        for r in beam:
            r["energy"] = round(r["energy"], 4)
        # Sample elements, then the lines that are in the spectrum for other reasons: Mo L (same Mo as the K line),
        # Rh L from the tube, Ar from the air path and Ca as a trace.
        sample = ["Cr", "Mn", "Fe", "Ni", "Cu", "Mo", "V", "Co", "W"]
        # The file says "uncalibrated" (5 eV per channel); the peaks put the energy scale at about 4.872 eV per channel.
        return set_spectrum(spec, {
            "elements": sample[:-1] + ["W-L", "Mo-L", "Rh-L", "Ar", "Ca"], "instrument": ["Rh-L", "Ar"], "zero": -0.018, "gain": 0.004872,
            "emin": 1.8, "emax": 22.0, "free": ["zero", "gain", "noise", "tail_frac", "esc_scale"],
            "tail_frac": 0.02, "esc_frac": 0.0, "sum_peaks": True, "beam": beam, "tube": tube_cfg,
            # MXR Lightbright 50 end-window tube: standard 50 um Be window, source facing the sample surface (normal incidence);
            # the detector is 47 deg from the sample normal, so 43 deg to the surface and a 133 deg scattering angle.
            "beam_filters": [{"material": "Be", "density": "", "thickness_um": "50"}], "scatter_angle": 133.0,
            "matrix": {"layers": [{"name": "SRM 1157 tool steel", "composition": "\n".join(f"{e} ?" for e in sample),
                                   "density": "7.8", "thickness_um": "", "fit_thickness": False,
                                   # NBS certificate of analysis, SRM 1157 (AISI M2), average of four laboratories, wt %
                                   "reference": "C 0.836\nMn 0.34\nP 0.011\nS 0.004\nSi 0.18\nCu 0.088\nNi 0.228\nCr 4.36\nV 1.82\nMo 4.86\nW 6.28\nCo 0.028\nFe balance"}],
                       "alpha_in": 90.0, "alpha_out": 43.0, "k": "", "fit_k": True, "secondary": True,
                       "detector": {"si_mm": 0.5, "be_um": 0, "air_mm": 0, "filters": []}}})
    raise ApiError("Unknown example.", 404)


def api_groups(body, query):
    e0 = exc_from_body(body)
    return {"entries": groups_info(body.get("tokens", []), bool(body.get("free_ratios")), e0,
                                   body.get("xs_source", "xraylib"))}


def api_lines(body, query):
    sym = str(body.get("symbol", "")).strip()
    try:
        return lines.table(sym, exc_from_body(body), body.get("xs_source", "xraylib"))
    except KeyError as e:
        raise ApiError(f"Unknown element '{sym}'. Use a symbol such as Fe or Pb.") from e


def api_filters(body, query):
    """Echo and transmission of a filter list at some energies (keV): {"filters": [...], "energies": [...]}."""
    try:
        flt = matrix.parse_filters(body.get("filters"))
        en = [float(e) for e in (body.get("energies") or [5, 10, 20, 30])][:12]
        t = matrix.filter_transmission(body.get("filters"), en)
    except (matrix.MatrixError, ValueError, TypeError) as e:
        return {"error": str(e)}
    return {"labels": [f[3] for f in flt], "energies": en, "transmission": [float(x) for x in t]}


def api_tube(body, query):
    try:
        rows = tube.tube_beam(str(body.get("anode", "Rh")), float(body.get("kv", 40)), int(body.get("bins", 24)),
                              float(body.get("emin", 1.5)), float(body.get("line_share", 0.25)),
                              None if body.get("l_share") in (None, "") else float(body.get("l_share")))
    except (KeyError, ValueError, TypeError) as e:
        raise ApiError(msg(e)) from e
    for r in rows:
        r["energy"] = round(r["energy"], 4)
    return {"beam": rows}


def api_composition(body, query):
    try:
        return {"entries": matrix.parse_composition(str(body.get("text", "")))}
    except matrix.MatrixError as e:
        return {"error": str(e)}


def api_matrix(body, query):
    mx = body.get("matrix")
    if not isinstance(mx, dict):
        raise ApiError("Describe the sample matrix first.")
    with STATE.lock:
        res = STATE.result
    if res is None:
        raise ApiError("Fit the spectrum first. The matrix calculation uses the fitted net areas.")
    try:
        return matrix.solve(mx, res)
    except (matrix.MatrixError, KeyError) as e:
        raise ApiError(msg(e)) from e


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


def api_spot(body, query):
    """Detection efficiency of soft lines versus the air distance, with the beam spot averaged in."""
    try:
        d1 = float(body.get("air_mm")); d2 = body.get("compare_mm")
        d2 = None if d2 in (None, "") else float(d2)
        spot = float(body.get("spot_mm") or 0)
        prof = "gaussian" if body.get("profile") == "gaussian" else "uniform"
    except (TypeError, ValueError):
        raise ApiError("Distances and spot size must be numbers.") from None
    if d1 <= 0 or (d2 is not None and d2 <= 0) or spot < 0:
        raise ApiError("Distances must be above zero and the spot size cannot be negative.")
    lines_ = body.get("lines") or [["Al Ka", 1.4867], ["Si Ka", 1.7400], ["S Ka", 2.3076], ["Rh La", 2.6967], ["Ar Ka", 2.9577], ["Fe Ka", 6.404]]
    names, E = [str(a) for a, _ in lines_], np.array([float(b) for _, b in lines_])
    mu = np.array([sum(w * attenuation.mu(s, e) for s, w in matrix.AIR.items()) * matrix.AIR_RHO for e in E])
    rho, w = matrix.spot_nodes(spot, prof)
    edge = float(rho[-1])
    rows = []
    for n, e, m_ in zip(names, E, mu):
        a_pt = float(matrix.spot_factor([m_], d1, 0)[0])            # on-axis air transmission
        a_sp = float(matrix.spot_factor([m_], d1, spot, prof)[0])   # spot average of solid angle x slanted-path transmission
        r = {"line": n, "E_keV": float(e), "transmission": a_pt, "spot_average": a_sp, "spot_vs_point": a_sp / a_pt}
        h = math.hypot(d1 / 10, edge)
        r["edge_vs_centre"] = float((d1 / 10 / h) ** 3 * math.exp(-m_ * (h - d1 / 10))) if spot > 0 else 1.0
        if d2 is not None:  # detected rate scales as (factor / distance^2) at fixed tube and sample
            q = (d1 / d2) ** 2
            r["gain"] = float(matrix.spot_factor([m_], d2, spot, prof)[0]) / a_sp * q
            r["gain_point"] = float(matrix.spot_factor([m_], d2, 0)[0]) / a_pt * q
        rows.append(r)
    return {"rows": rows, "air_mm": d1, "compare_mm": d2, "spot_mm": spot, "profile": prof}


def api_calsignals(body, query):
    """Signal (ROI sum or fitted net area) of each calibration spectrum."""
    files = body.get("files") or []
    if not files:
        raise ApiError("Choose at least one spectrum file.")
    specs = []
    for f in files:
        try:
            specs.append(load_bytes(f["name"], base64.b64decode(f["b64"])))
        except Exception as e:
            raise ApiError(f"Could not read {f.get('name', 'file')}: {e}") from e
    sig = body.get("signal") or {}
    norm = "live_time" if sig.get("normalize") == "live_time" else "none"
    try:
        if sig.get("kind") == "area":
            cfg = cfg_from_json(body.get("config"))
            if not cfg.elements:
                raise ApiError("Add the element to fit under Elements first.")
            if not sig.get("group"):
                raise ApiError("Choose the line group to use as the signal.")
            vals = calibration.area_signals(specs, cfg, sig["group"], norm)
        else:
            cfg = cfg_from_json(body.get("config"))
            zero = cfg.zero if sig.get("zero") in (None, "") else float(sig["zero"])
            gain = cfg.gain if sig.get("gain") in (None, "") else float(sig["gain"])
            vals = calibration.roi_signals(specs, zero, gain, float(sig.get("emin")), float(sig.get("emax")),
                                           "edges" if sig.get("background") == "edges" else "none", normalize=norm)
    except (KeyError, ValueError, TypeError) as e:
        raise ApiError(msg(e)) from e
    return {"rows": [{"name": s.name, "live_time": s.live_time, "signal": v[0], "err": v[1]} for s, v in zip(specs, vals)],
            "zero": zero if sig.get("kind") != "area" else cfg.zero, "gain": gain if sig.get("kind") != "area" else cfg.gain}


def api_calfit(body, query):
    rows = []
    for r in body.get("rows") or []:
        amt = r.get("amount")
        try:
            amt = None if amt in (None, "") else float(amt)
        except (TypeError, ValueError):
            raise ApiError(f"The amount for {r.get('name', 'a row')} is not a number.") from None
        rows.append({"name": r.get("name", ""), "signal": float(r["signal"]), "err": float(r["err"]), "amount": amt, "use": bool(r.get("use", True))})
    try:
        return calibration.calibrate(rows, bool(body.get("through_zero")), bool(body.get("weighted")))
    except (ValueError, np.linalg.LinAlgError) as e:
        raise ApiError(msg(e)) from e


POST_ROUTES = {"/api/spot": api_spot, "/api/calsignals": api_calsignals, "/api/calfit": api_calfit, "/api/filters": api_filters, "/api/matrix": api_matrix, "/api/composition": api_composition, "/api/tube": api_tube, "/api/groups": api_groups, "/api/lines": api_lines, "/api/fit": api_fit, "/api/batch": api_batch}
GET_ROUTES = {"/api/session": api_session, "/api/elements": api_elements, "/api/example": api_example}


# ----------------------------------------------------------------- dispatch
def dispatch(method, path, headers, raw=b""):
    """Transport-independent request handler: returns (status, content_type, body_bytes, extra_headers).

    Used by the HTTP server below and by the in-browser (Pyodide) build on GitHub Pages."""
    u = urlparse(path)
    js = "application/json; charset=utf-8"

    def j(status, obj):
        return status, js, json.dumps(clean(obj), allow_nan=False).encode("utf-8"), {}

    try:
        if method == "GET":
            if u.path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    return 200, "text/html; charset=utf-8", f.read(), {}
            if u.path == "/favicon.ico":
                return 204, "image/x-icon", b"", {}
            if u.path == "/api/fit.csv":
                with STATE.lock:
                    res = STATE.result
                if res is None:
                    raise ApiError("Run a fit first.", 404)
                return 200, "text/csv; charset=utf-8", fit_csv(res).encode("utf-8"), \
                    {"Content-Disposition": 'attachment; filename="xrfit-fit.csv"'}
            fn = GET_ROUTES.get(u.path)
            if fn:
                return j(200, fn(None, parse_qs(u.query)))
            return 404, "text/plain", b"Not found", {}
        if headers.get("X-Requested-With") != "xrfit":  # blocks cross-site form posts
            return 403, "text/plain", b"Missing X-Requested-With header", {}
        if len(raw) > MAX_BODY:
            raise ApiError("File is too large (limit 256 MB).", 413)
        if u.path == "/api/load":
            name = os.path.basename(requote(headers.get("X-Filename", "spectrum.txt"))) or "spectrum.txt"
            try:
                spec = load_bytes(name, raw)
            except Exception as e:
                raise ApiError(f"Could not read {name}: {e}") from e
            return j(200, set_spectrum(spec))
        fn = POST_ROUTES.get(u.path)
        if not fn:
            return 404, "text/plain", b"Not found", {}
        body = json.loads(raw.decode("utf-8") or "{}")
        return j(200, fn(body, parse_qs(u.query)))
    except ApiError as e:
        return j(e.status, {"error": str(e)})
    except json.JSONDecodeError:
        return j(400, {"error": "Request body is not valid JSON."})
    except Exception as e:  # surface unexpected problems in the UI instead of a blank page
        return j(500, {"error": f"{type(e).__name__}: {e}"})


# ------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "xrfit-web"
    allowed_hosts = None  # set by serve()

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _host_ok(self):
        if self.allowed_hosts is None:
            return True
        return (self.headers.get("Host") or "").lower() in self.allowed_hosts

    def _handle(self, method):
        if not self._host_ok():
            status, ctype, data, extra = 403, "text/plain", b"Forbidden host", {}
        else:
            raw = b""
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    status, ctype, data, extra = 413, "application/json", b'{"error": "File is too large (limit 256 MB)."}', {}
                    return self._reply(status, ctype, data, extra)
                raw = self.rfile.read(n)
            status, ctype, data, extra = dispatch(method, self.path, self.headers, raw)
        self._reply(status, ctype, data, extra)

    def _reply(self, status, ctype, data, extra):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


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
