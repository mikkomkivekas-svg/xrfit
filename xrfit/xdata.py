# SPDX-License-Identifier: GPL-3.0-or-later
"""Line energies and production cross sections from the two bundled source tables.

  * X-Ray Data Booklet Table 1-2 (eV): Ka1 Ka2 Kb1 La1 La2 Lb1 Lb2 Lg1 Ma1, Z=3-95
  * XRAYLIB tables v2.3 (2003): edges, fluorescence yields, jump factors, line energies,
    fractional radiative rates and K/L-shell XRF cross sections (barns/atom) for Z=5-90.
    Every cross section is tabulated twice: the xraylib value and Krause's value.
Regenerate data/*.json from the PDFs with tools/build_data.py.
"""
import json
import math
import os

_DIR = os.path.join(os.path.dirname(__file__), "data")
_cache = {}

# xrfit line name -> (xraylib key, shell of the initial vacancy)
XL_KEY = {"Ka1": "KL3", "Ka2": "KL2", "Kb1": "KM3", "Kb3": "KM2",
          "Lb3": "L1M3", "Lb1": "L2M4", "Lg1": "L2N4", "La1": "L3M5", "Ll": "L3M1"}
VACANCY = {"Ka1": "K", "Ka2": "K", "Kb1": "K", "Kb3": "K", "Lb3": "L1", "Lb1": "L2",
           "Lg1": "L2", "La1": "L3", "Ll": "L3", "La2": "L3", "Lb2": "L3", "Ma1": "M"}
T12_ONLY = ("La2", "Lb2", "Ma1")
ALL_LINES = ["Ka1", "Ka2", "Kb1", "Kb3", "La1", "La2", "Lb1", "Lb2", "Lb3", "Lg1", "Ll", "Ma1"]
_ALIASES = {n.lower(): n for n in ALL_LINES}


def _load(name):
    if name not in _cache:
        with open(os.path.join(_DIR, name)) as f:
            _cache[name] = json.load(f)
    return _cache[name]


def t12():
    return _load("xdb_table_1_2.json")


def xl():
    return _load("xraylib_v23.json")


def symbols():
    return sorted(set(t12()) | set(xl()), key=lambda s: (t12().get(s) or xl()[s])["Z"])


def canon_line(name):
    try:
        return _ALIASES[name.lower().replace("α", "a").replace("β", "b").replace("γ", "g")]
    except KeyError:
        raise KeyError(f"unknown line '{name}' (known: {', '.join(ALL_LINES)})")


def check_element(sym):
    if sym not in t12() and sym not in xl():
        raise KeyError(f"unknown element '{sym}'")


def line_energy(sym, line):
    """keV, or None. Table 1-2 where listed; xraylib otherwise (Kb3 is placed relative to Kb1
    using the xraylib Kb3-Kb1 spacing so the two sources' small offsets do not mix)."""
    line = canon_line(line)
    t = t12().get(sym, {})
    if t.get(line) is not None:
        return t[line] / 1000.0
    x = xl().get(sym)
    if not x or line in T12_ONLY:
        return None
    e = x["line_keV"].get(XL_KEY[line])
    if e is None or e <= 0:
        return None
    if line == "Kb3" and t.get("Kb1") is not None and x["line_keV"].get("KM3"):
        return t["Kb1"] / 1000.0 + (e - x["line_keV"]["KM3"])
    return e


def line_rate(sym, line):
    """xraylib fractional radiative rate (relative to its own subshell vacancy), or None."""
    line = canon_line(line)
    x = xl().get(sym)
    if not x or line in T12_ONLY:
        return None
    r = x["rate"].get(XL_KEY[line])
    return r if r else None


def edge(sym, shell):
    x = xl().get(sym)
    return x["edge_keV"].get(shell) if x else None


def _series(sym, line, source):
    """(energies, sigma) valid grid points of the production cross section of one line."""
    x = xl().get(sym)
    line = canon_line(line)
    if line in T12_ONLY or not x:
        raise KeyError(f"no cross-section data for {sym}-{line}")
    key = XL_KEY[line]
    if key[0] == "K":
        tab = x.get("xs_K")
        if not tab:
            raise KeyError(f"{sym}: no K-shell cross sections (tables cover Z=5-90)")
        if key in ("KL3", "KM3"):
            vals = tab[key][source]
        else:  # Ka2 = KL23 - KL3, Kb3 = KM23 - KM3
            base, tot_k = ("KL3", "KL23") if key == "KL2" else ("KM3", "KM23")
            vals = [None if (a is None or b is None) else a - b
                    for a, b in zip(tab[tot_k][source], tab[base][source])]
        e_edge = x["edge_keV"]["K"] or 0.0
        pts = [(E, v) for E, v in zip(tab["E_keV"], vals) if v and v > 0 and E > e_edge * (1 + 1e-9)]
    else:
        tab = x.get("xs_L")
        if not tab:
            raise KeyError(f"{sym}: no L-shell cross sections (tables cover Z=12-90 where L lines exist)")
        vals = tab[key][source]
        l_edges = [x["edge_keV"][k] for k in ("L1", "L2", "L3") if x["edge_keV"].get(k)]
        # The PDF inserts extra rows exactly at the L1/L2/L3 edge energies (before the regular grid,
        # so rows are not energy-sorted). Their values are edge-adjacent and not smooth, so only
        # the regular grid (plus the closing row at the K edge) is used for interpolation.
        pts = [(E, v) for E, v in zip(tab["E_keV"], vals)
               if v and v > 0 and not any(abs(E - e) < 0.0055 for e in l_edges)]
    pts.sort()
    if len(pts) < 2:
        raise KeyError(f"{sym}-{line}: not enough cross-section points")
    return [p[0] for p in pts], [p[1] for p in pts]


def xs_line(sym, line, E, source="xraylib"):
    """Production (XRF) cross section, barns/atom, of one line at excitation energy E (keV).
    Log-log interpolation of the tabulated grid. Returns (value, note); value is 0 below the
    absorption edge; note says when the value is extrapolated outside the tabulated grid
    (notably just above an edge, where cross sections change fastest)."""
    line = canon_line(line)
    shell = VACANCY[line]
    ed = edge(sym, shell)
    if ed is not None and E < ed:
        return 0.0, "below edge"
    xs, ys = _series(sym, line, source)
    lx, ly, E_l = [math.log(v) for v in xs], [math.log(v) for v in ys], math.log(E)
    note = ""
    if E_l <= lx[0]:
        i = 0
        if E < xs[0] * 0.999: note = f"extrapolated below {xs[0]:g} keV (grid start)"
    elif E_l >= lx[-1]:
        i = len(lx) - 2
        if E > xs[-1] * 1.001: note = f"extrapolated above {xs[-1]:g} keV (grid end)"
    else:
        i = max(j for j in range(len(lx) - 1) if lx[j] <= E_l)
    s = (ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i])
    return math.exp(ly[i] + s * (E_l - lx[i])), note
