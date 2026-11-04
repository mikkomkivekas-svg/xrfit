"""KETEK .vspc reader.

Verified against a real VICO-Scope JSON export (Version 1) with the sections
  Calibration{BinSize_keV/Bin, Gain, Offset_keV, Type}
  Configuration.Parameters[{ParameterID, ParameterName, ParameterValue}]
  MeasurementInfo{Firmware, Hardware, ROI[], RunStatistics, Software, Status, Temperature, Time}
  RawData[counts]
Also accepted: the plain 'Channel,RawData' CSV export (no metadata), a RawData given as
{channel: counts} / [{Channel, RawData}], and a text file with a "key: value" header.
"""
import json
import re

import numpy as np

KEY_LIVE = re.compile(r"live\s*[_ ]?time|\blive\b", re.I)
KEY_REAL = re.compile(r"real\s*[_ ]?time|\breal\b|elapsed", re.I)
KEY_EVBIN = re.compile(r"ev\s*/\s*bin|bin\s*(width|size)|ev\s*per\s*(bin|channel)", re.I)
KEY_OFFSET = re.compile(r"offset|zero", re.I)
_SPLIT = re.compile(r"\s*[:=\t]\s*")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# (title, regex on full dotted key, shown without 'info all').  First match wins.
GROUPS = [
    ("Slow filter", re.compile(r"slow", re.I), True),
    ("Detector / calibration", re.compile(r"SDDTemperature|^Calibration", re.I), True),
    ("Board temperatures / ready flags", re.compile(r"Temperature", re.I), False),
    ("Fast/medium filter, reset, baseline, gain",
     re.compile(r"fastfilter|mediumfilter|baseline|digital energy|dynamic reset|reset (inhibit|detection)|trigger|threshold", re.I), False),
    ("Measurement time", re.compile(r"^MeasurementInfo\.Time\.|RunStatistics|Stop Condition|^[^.]*(live|real|start|stop|duration)", re.I), True),
    ("Identification", re.compile(r"^MeasurementInfo\.Hardware|name|serial|sample|title|operator|comment", re.I), True),
    ("Firmware / software / status", re.compile(r"Firmware|Software|Status|version", re.I), False),
]
_STRIP = ("MeasurementInfo.", "Configuration.Parameters.", "Configuration.")


def _num(s):
    m = _NUM.search(str(s))
    return float(m.group()) if m else None


def _flatten(o, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(o, list) and o and all(isinstance(x, dict) and "ParameterName" in x for x in o):
        for x in o:  # KETEK parameter list -> {name: value}
            out[f"{prefix}.{x['ParameterName']}"] = x.get("ParameterValue")
    elif isinstance(o, list) and len(o) <= 32 and all(not isinstance(x, (dict, list)) for x in o):
        out[prefix] = ", ".join(map(str, o))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = o
    return out


def _find_rawdata(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if re.fullmatch(r"raw[_ ]?data", str(k), re.I):
                return v, {kk: vv for kk, vv in o.items() if kk != k}
        for k, v in o.items():
            r = _find_rawdata(v)
            if r[0] is not None:
                rest = dict(o); rest[k] = r[1]
                return r[0], rest
    elif isinstance(o, list):
        for i, v in enumerate(o):
            r = _find_rawdata(v)
            if r[0] is not None:
                rest = list(o); rest[i] = r[1]
                return r[0], rest
    return None, o


def _to_counts(data):
    if isinstance(data, dict):
        return np.array([v for _, v in sorted((float(k), float(v)) for k, v in data.items())])
    if isinstance(data, list) and data and isinstance(data[0], dict):
        recs = []
        for r in data:
            low = {k.lower(): v for k, v in r.items()}
            recs.append((float(low.get("channel", len(recs))), float(low.get("rawdata", low.get("counts", 0)))))
        return np.array([v for _, v in sorted(recs)])
    return np.asarray(data, float).ravel()


def parse_vspc(path):
    """Return (meta dict, counts ndarray, fmt)."""
    raw = open(path, "rb").read()
    if b"\x00" in raw[:4096]:
        raise ValueError("binary .vspc detected; only JSON/CSV/text VSPC export is supported")
    text = raw.decode("utf-8-sig", "replace")
    if text.lstrip()[:1] in "{[":
        doc = json.loads(text)
        data, rest = _find_rawdata(doc)
        if data is None:
            raise ValueError("JSON .vspc has no 'RawData' section")
        return _flatten(rest), _to_counts(data), "json"
    first = text.splitlines()[0].strip().lower() if text.strip() else ""
    if re.match(r"channel\s*,\s*raw\s*data", first):
        rows = sorted((l.split(",") for l in text.splitlines()[1:] if l.strip()), key=lambda r: float(r[0]))
        return {"format": "KETEK CSV export (no metadata)", "channels": len(rows)}, \
            np.array([float(r[1]) for r in rows]), "csv"
    meta, cur, blocks = {}, [], []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            cur.append([float(t) for t in re.split(r"[,;\s]+", s) if t]); continue
        except ValueError:
            pass
        if cur: blocks.append(cur); cur = []
        kv = _SPLIT.split(s, maxsplit=1)
        if len(kv) == 2: meta[kv[0].strip()] = kv[1].strip()
    if cur: blocks.append(cur)
    blocks = [b for b in blocks if len({len(r) for r in b}) == 1]
    if not blocks:
        raise ValueError(f"no spectrum found in {path}")
    return meta, np.array(max(blocks, key=len), float)[:, -1], "text"


def load_vspc(path):
    from .core import Spectrum
    meta, counts, fmt = parse_vspc(path)
    g = lambda k: _num(meta[k]) if k in meta and not isinstance(meta[k], bool) else None
    live, real = g("MeasurementInfo.RunStatistics.LiveTime_s"), g("MeasurementInfo.RunStatistics.RealTime_s")
    binsize, offset = g("Calibration.BinSize_keV/Bin"), g("Calibration.Offset_keV") or 0.0
    ev_bin = None
    if (live is None and real is None) or binsize is None:  # generic key scan (other layouts)
        for k, v in meta.items():
            if k.startswith("Configuration.") or isinstance(v, bool):
                continue
            x = _num(v)
            if x is None:
                continue
            leaf = k.split(".")[-1]
            if KEY_LIVE.search(leaf) and live is None: live = x
            elif KEY_REAL.search(leaf) and real is None: real = x
            elif KEY_EVBIN.search(leaf) and ev_bin is None and binsize is None: ev_bin = x
            elif KEY_OFFSET.search(leaf) and "ev" in (k + str(v)).lower(): offset = x / 1000.0
    if binsize is None and ev_bin:
        binsize = ev_bin / 1000.0
    calib = (offset, binsize) if binsize else None
    return Spectrum(counts, path.split("/")[-1], live or real or 1.0, calib, {"_format": fmt, **meta})


def _label(k):
    for p in _STRIP:
        if k.startswith(p):
            return k[len(p):]
    return k


def describe(meta, show_all=False, name=None, width=44):
    """Grouped, human-readable metadata block."""
    if not meta:
        return "  (no metadata)"
    groups = {t: [] for t, _, _ in GROUPS}; other = []
    for k, v in meta.items():
        if k.startswith("_"):
            continue
        for t, rx, _ in GROUPS:
            if rx.search(k):
                groups[t].append((k, v)); break
        else:
            other.append((k, v))
    out = [f"  file: {name or '?'}   (format: {meta.get('_format', 'n/a')})"]
    ordered = [(t, groups[t], brief) for t, _, brief in GROUPS] + [("Other", other, False)]
    hidden = 0
    for t, kv, brief in ordered:
        if not kv:
            continue
        if brief or show_all:
            out.append(f" {t}:")
            out += [f"   {_label(k):{width}s} {v if not isinstance(v, float) else f'{v:.6g}'}" for k, v in kv]
        else:
            hidden += len(kv)
    if hidden:
        out.append(f" ({hidden} more fields; 'info all' to show)")
    return "\n".join(out)


def summary(meta, name=None):
    """Short multi-line text for the plot annotation."""
    def g(k, f):
        return f.format(meta[k]) if k in meta else None
    rows = [
        f"file {name}" if name else None,
        g("MeasurementInfo.Hardware.ProductSerialNumber", "serial {}"),
        g("MeasurementInfo.Time.Start", "start {}"),
        g("MeasurementInfo.RunStatistics.RealTime_s", "real {} s"),
        g("MeasurementInfo.RunStatistics.LiveTime_s", "live {:.4g} s"),
        g("MeasurementInfo.RunStatistics.DeadTime_%", "dead {:.3g} %"),
        g("MeasurementInfo.RunStatistics.InputCountRate_cps", "ICR {} cps"),
        g("Configuration.Parameters.Slowfilter Peaking Time/ns", "slow filter peaking {} ns"),
        g("Configuration.Parameters.Slowfilter Gap Time/ns", "slow filter gap {} ns"),
    ]
    return "\n".join(r for r in rows if r)
