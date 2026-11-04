# SPDX-License-Identifier: GPL-3.0-or-later
"""Build xrfit/data/*.json from the two source PDFs.

    python tools/build_data.py Table_1-2.pdf xraylib_tables_v2.3.pdf

Sources
  * X-Ray Data Booklet Table 1-2 (LBNL): emission line energies, eV, Z=3-95
  * XRAYLIB tables v2.3 (Sanchez del Rio et al., ESRF, 2003): edges, fluorescence yields, jump
    factors, line energies, fractional radiative rates, K/L-shell XRF cross sections (barns/atom),
    each given as (xraylib value, Krause value), Z=5-90 (Z=1-4 pages are empty).
"""
import json
import math
import re
import sys

import pdfplumber

T12_COLS = ["Ka1", "Ka2", "Kb1", "La1", "La2", "Lb1", "Lb2", "Lg1", "Ma1"]
EDGES = ["L3", "L2", "L1", "K"]
K_LINES = ["KL3", "KL2", "KM3", "KM2"]                    # Ka1 Ka2 Kb1 Kb3
L_LINES = ["L1M3", "L2M4", "L2N4", "L3M5", "L3M1"]        # Lb3 Lb1 Lg1 La1 Ll
XS_K = ["KL3", "KL23", "KM3", "KM23"]                     # Ka1, Ka(2,1)=total Ka, Kb1, Kb(3,1)=total Kb
XS_L = L_LINES
NAN = None


def flt(tok):
    t = tok.replace(",", "")
    if t in ("-", "—", "–"):
        return NAN
    return float(t)


def parse_table12(path):
    out = {}
    with pdfplumber.open(path) as p:
        for pg in p.pages:
            for line in (pg.extract_text() or "").splitlines():
                m = re.match(r"^(\d+)\s+([A-Z][a-z]?)\s+(.+)$", line.strip())
                if not m:
                    continue
                z, sym, rest = int(m.group(1)), m.group(2), m.group(3).split()
                vals = [flt(t) for t in rest]
                if len(vals) > len(T12_COLS):
                    raise ValueError(f"Table 1-2 row too long: {line}")
                vals += [NAN] * (len(T12_COLS) - len(vals))
                out[sym] = {"Z": z, **{c: v for c, v in zip(T12_COLS, vals)}}  # eV
    return out


def _row(text, label, n, occurrence=0):
    hits = [l for l in text.splitlines() if l.startswith(label)]
    toks = hits[occurrence][len(label):].split()
    if len(toks) != n:
        raise ValueError(f"{label!r}[{occurrence}]: expected {n} values, got {toks}")
    return [flt(t) for t in toks]


def parse_xraylib(path):
    els = {}
    with pdfplumber.open(path) as p:
        texts = [(pg.extract_text() or "") for pg in p.pages]
    for t in texts:
        m = re.match(r"Element:\s*(\w+)\s*,\s*Z=\s*(\d+)", t)
        if not m:
            continue
        sym, z = m.group(1), int(m.group(2))
        d = {"Z": z}
        d["edge_keV"] = dict(zip(EDGES, _row(t, "Edge [keV]", 4)))
        d["fluor_yield_xraylib"] = dict(zip(EDGES, _row(t, "Fluor Yield (xraylib)", 4)))
        d["fluor_yield_krause"] = dict(zip(EDGES, _row(t, "Fluor Yield (Krause)", 4)))
        d["jump"] = dict(zip(EDGES, _row(t, "Jump Factor", 4)))
        d["line_keV"] = {**dict(zip(K_LINES, _row(t, "Line energy [keV]", 4, 0))),
                         **dict(zip(L_LINES, _row(t, "Line energy [keV]", 5, 1)))}
        d["rate"] = {**dict(zip(K_LINES, _row(t, "Radiative rate", 4, 0))),
                     **dict(zip(L_LINES, _row(t, "Radiative rate", 5, 1)))}
        els[sym] = d
    for t in texts:
        if not re.search(r"cross\s*sections", t[:200]):
            continue
        m = re.search(r"for\s+(\w+)\s+\(Z=\s*(\d+)\)", t)
        if not m:
            continue
        sym = m.group(1)
        sect, tables = None, {}
        for line in t.splitlines():
            if re.search(r"K-shell\s*XRF", line): sect = "K"; tables[sect] = []; continue
            if re.search(r"L-shell\s*XRF", line): sect = "L"; tables[sect] = []; continue
            if sect and re.match(r"^\d+\.\d+\s", line):
                tables[sect].append(line.split())
        for sect, rows in tables.items():
            names = XS_K if sect == "K" else XS_L
            tab = {"E_keV": []}
            for n in names:
                tab[n] = {"xraylib": [], "krause": []}
            for toks in rows:
                e, rest, k = float(toks[0]), toks[1:], 0
                vals = []
                for _ in names:
                    if k >= len(rest):
                        raise ValueError(f"{sym} {sect} row too short: {toks}")
                    if rest[k] == "-":
                        vals.append((NAN, NAN)); k += 1
                    else:
                        vals.append((flt(rest[k]), flt(rest[k + 1]))); k += 2
                if k != len(rest):
                    raise ValueError(f"{sym} {sect} row has leftover tokens: {toks}")
                tab["E_keV"].append(e)
                for n, (a, b) in zip(names, vals):
                    tab[n]["xraylib"].append(a); tab[n]["krause"].append(b)
            els[sym]["xs_" + sect] = tab
    return els


def validate(t12, xl):
    rep, bad = [], 0
    # 1. line energies, Table 1-2 (eV) vs xraylib (keV, 3 decimals)
    pairs = {"Ka1": "KL3", "Ka2": "KL2", "Kb1": "KM3", "Lb1": "L2M4", "Lg1": "L2N4", "La1": "L3M5"}
    worst = []
    for sym, e in xl.items():
        if sym not in t12: continue
        for a, b in pairs.items():
            va, vb = t12[sym][a], xl[sym]["line_keV"].get(b)
            if va is None or vb is None: continue
            worst.append((abs(va - vb * 1000), sym, a, va, vb * 1000))
    worst.sort(reverse=True)
    rep.append(f"line energy cross-check: {len(worst)} pairs; median |dE| = {worst[len(worst)//2][0]:.2f} eV")
    rep.append("  largest differences (eV): " + "; ".join(f"{s} {a} {d:.1f} (T12 {x:.1f} vs xl {y:.1f})" for d, s, a, x, y in worst[:8]))
    # 2. K-shell cross-section columns against radiative rates
    errs = []
    for sym, e in xl.items():
        if "xs_K" not in e: continue
        r = e["rate"]; tab = e["xs_K"]
        for i, E in enumerate(tab["E_keV"]):
            for src in ("xraylib", "krause"):
                a1, tot = tab["KL3"][src][i], tab["KL23"][src][i]
                if a1 and tot and r["KL3"] and r["KL2"] is not None:
                    errs.append((abs(tot / a1 / (1 + r["KL2"] / r["KL3"]) - 1), sym, E, src, "Ka"))
                b1, tb = tab["KM3"][src][i], tab["KM23"][src][i]
                if b1 and tb and r["KM3"] and r["KM2"] is not None:
                    errs.append((abs(tb / b1 / (1 + r["KM2"] / r["KM3"]) - 1), sym, E, src, "Kb"))
    errs.sort(reverse=True)
    rep.append(f"K cross-section vs radiative-rate consistency: {len(errs)} checks; median rel.err {errs[len(errs)//2][0]:.3%}; "
               f"worst: " + "; ".join(f"{s} {E}keV {src} {k} {d:.1%}" for d, s, E, src, k in errs[:5]))
    # 3. monotonic decrease of K cross sections above the edge (typo detector)
    nonmono = []
    for sym, e in xl.items():
        for sect in ("xs_K", "xs_L"):
            if sect not in e: continue
            tab = e[sect]
            for n, v in tab.items():
                if n == "E_keV": continue
                for src in ("xraylib", "krause"):
                    ys = [(E, y) for E, y in zip(tab["E_keV"], v[src]) if y]
                    if sect == "xs_K" and any(y2 > y1 * 1.0001 for (_, y1), (_, y2) in zip(ys, ys[1:])):
                        nonmono.append(f"{sym}:{n}:{src}")
    rep.append(f"K-shell series not monotonically decreasing: {len(nonmono)} {nonmono[:10]}")
    # 4. rate sums
    over = [(s, sum(x for k, x in e['rate'].items() if k in K_LINES and x)) for s, e in xl.items()]
    over = [(s, v) for s, v in over if v > 1.0005]
    rep.append(f"K radiative-rate sums > 1: {over}")
    # 5. coverage
    rep.append(f"Table 1-2: {len(t12)} elements (Z={min(v['Z'] for v in t12.values())}-{max(v['Z'] for v in t12.values())}); "
               f"xraylib pages: {len(xl)} elements, with K xs: {sum('xs_K' in e for e in xl.values())}, with L xs: {sum('xs_L' in e for e in xl.values())}")
    return rep


if __name__ == "__main__":
    t12 = parse_table12(sys.argv[1])
    xl = parse_xraylib(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "xrfit/data"
    json.dump(t12, open(f"{out}/xdb_table_1_2.json", "w"), indent=0, sort_keys=False)
    json.dump(xl, open(f"{out}/xraylib_v23.json", "w"), indent=0)
    print("\n".join(validate(t12, xl)))
