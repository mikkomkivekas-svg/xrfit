# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit line groups built from the bundled Table 1-2 energies and xraylib rates/cross sections.

Tokens understood by build_groups():
    Fe, Fe-K, Pb-L, Fe-M     element / shell family (bare symbol: K when the K edge <= e0, otherwise L)
    Fe-Ka1, Pb-La2, ...      one single line (any line of Table 1-2, plus Kb3, Lb3, Ll from xraylib)
Group intensities (weights sum to 1, so the fitted coefficient is the group's net counts):
    bare symbol: K if the K edge <= e0, else L (set e0 with 'set e0 <keV>')
    K family   xraylib fractional radiative rates of Ka1, Ka2, Kb1, Kb3
    L family   production cross sections of Lb3, Lb1, Lg1, La1, Ll at the excitation energy e0
    M family   Ma1 only
La2 and Lb2 are in Table 1-2 but neither source gives an intensity for them, so they are only
available as single-line tokens, not inside the L family group.
"""
import functools

from . import xdata

SI_ESCAPE = 1.7400  # Si K-alpha, keV (Si-detector escape)

_K = ["Ka1", "Ka2", "Kb1", "Kb3"]
_L = ["Lb3", "Lb1", "Lg1", "La1", "Ll"]
_SPLIT = {"K": [("Ka", ["Ka1", "Ka2"]), ("Kb", ["Kb1", "Kb3"])],
          "L": [("La", ["La1", "Ll"]), ("Lb", ["Lb1", "Lb3"]), ("Lg", ["Lg1"])]}


def _norm(items):
    tot = sum(w for _, _, w in items)
    if tot <= 0:
        raise KeyError("zero total intensity")
    return [(n, E, w / tot) for n, E, w in items]


@functools.lru_cache(maxsize=1024)
def family_lines(sym, fam, e0=20.0, source="xraylib"):
    """[(line, E_keV, weight)] normalised to 1 for family K / L / M."""
    xdata.check_element(sym)
    if fam == "M":
        E = xdata.line_energy(sym, "Ma1")
        if E is None:
            raise KeyError(f"{sym}: no M line in Table 1-2")
        return [("Ma1", E, 1.0)]
    names = _K if fam == "K" else _L
    items = []
    if fam == "K":
        for n in names:
            E, r = xdata.line_energy(sym, n), xdata.line_rate(sym, n)
            if E is not None and r:
                items.append((n, E, r))
        if not items:
            E = xdata.line_energy(sym, "Ka1")  # Li-F style: only Ka1 listed
            if E is None or xdata.line_energy(sym, "Kb1") is not None:
                raise KeyError(f"{sym}-K: no intensity data in the xraylib tables (Z=5-90); use single lines, e.g. {sym}-Ka1")
            items = [("Ka1", E, 1.0)]
    else:
        for n in names:
            E = xdata.line_energy(sym, n)
            if E is None:
                continue
            try:
                xs, _ = xdata.xs_line(sym, n, e0, source)
            except KeyError as e:
                raise KeyError(f"{sym}-L: the xraylib tables have no cross section for {n} ({e}); "
                               f"use single lines, e.g. {sym}-La1") from None
            if xs > 0:
                items.append((n, E, xs))
        if not items:
            raise KeyError(f"{sym}-L: no L cross sections at e0={e0} keV (below the L edges, or element "
                           f"outside the xraylib tables Z=5-90); use single lines, e.g. {sym}-La1")
    return tuple(_norm(items))


def default_family(sym, e0=20.0):
    """K if the K edge is at or below the excitation energy e0, otherwise L if the element has
    L lines in Table 1-2, otherwise K."""
    x = xdata.xl().get(sym)
    k_edge = x["edge_keV"].get("K") if x else None
    if k_edge and k_edge <= e0 * (1 + 1e-9):
        return "K"
    if xdata.t12().get(sym, {}).get("La1") is not None:
        return "L"
    return "K"


def parse_token(tok, e0=20.0):
    sym, _, rest = tok.partition("-")
    xdata.check_element(sym)
    if not rest:
        rest = default_family(sym, e0)
    if rest in ("K", "L", "M"):
        return sym, rest, None
    return sym, None, xdata.canon_line(rest)


def build_groups(tokens, free_ratios=False, e0=20.0, source="xraylib"):
    """-> [(group name, [(E_keV, weight), ...]), ...]"""
    out = []
    for tok in tokens:
        sym, fam, line = parse_token(tok, e0)
        if line:
            E = xdata.line_energy(sym, line)
            if E is None:
                raise KeyError(f"{sym}-{line}: no energy in the tables")
            out.append((f"{sym}-{line}", [(E, 1.0)]))
            continue
        lines = family_lines(sym, fam, e0, source)
        if free_ratios and fam in _SPLIT:
            for suf, members in _SPLIT[fam]:
                sub = [(n, E, w) for n, E, w in lines if n in members]
                if sub:
                    out.append((f"{sym}-{suf}", [(E, w) for _, E, w in _norm(sub)]))
        else:
            out.append((f"{sym}-{fam}", [(E, w) for _, E, w in lines]))
    return out


def group_members(name, e0=20.0, source="xraylib"):
    """Lines inside a fitted group name like 'Fe-K', 'Fe-Ka', 'Pb-L', 'Pb-La2' -> [line names]."""
    sym, _, rest = name.partition("-")
    for fam, subs in _SPLIT.items():
        for suf, members in subs:
            if rest == suf:
                return sym, [n for n, _, _ in family_lines(sym, fam, e0, source) if n in members]
    if rest in ("K", "L", "M"):
        return sym, [n for n, _, _ in family_lines(sym, rest, e0, source)]
    return sym, [xdata.canon_line(rest)]


def group_sigma(name, e0=20.0, source="xraylib"):
    """Total production cross section (barns/atom) of the lines in a fitted group at e0."""
    sym, members = group_members(name, e0, source)
    notes, tot = set(), 0.0
    for n in members:
        s, note = xdata.xs_line(sym, n, e0, source)
        tot += s
        if note: notes.add(note)
    return tot, "; ".join(sorted(notes))


def table(sym, e0=20.0, source="xraylib"):
    """Every line known for an element, structured (used by the web UI and describe())."""
    xdata.check_element(sym)
    fam_w = {}
    for fam in ("K", "L"):
        try:
            fam_w.update({n: w for n, _, w in family_lines(sym, fam, e0, source)})
        except KeyError:
            pass
    rows = []
    for n in xdata.ALL_LINES:
        E = xdata.line_energy(sym, n)
        if E is None:
            continue
        try:
            sig, note = xdata.xs_line(sym, n, e0, source)
        except KeyError:
            sig, note = None, ""
        rows.append({"line": n, "E_keV": E, "rate": xdata.line_rate(sym, n), "sigma_b": sig,
                     "note": note, "weight": fam_w.get(n)})
    edges = {sh: xdata.edge(sym, sh) for sh in ("K", "L1", "L2", "L3") if xdata.edge(sym, sh)}
    z = (xdata.t12().get(sym) or xdata.xl()[sym])["Z"]
    return {"symbol": sym, "Z": z, "e0": e0, "source": source, "rows": rows, "edges": edges}


def describe(sym, e0=20.0, source="xraylib"):
    """Text table of every line known for an element."""
    t = table(sym, e0, source)
    rows = [f"{sym}: lines (keV) | rate = xraylib radiative rate | sigma = production cross section at e0={e0:g} keV [{source}]"]
    for r in t["rows"]:
        sg = f"{r['sigma_b']:10.4g} b" + (f"  ({r['note']})" if r["note"] else "") if r["sigma_b"] is not None else "         - "
        w = f"{r['weight']:.3f}" if r["weight"] is not None else "  -  "
        rt = f"{r['rate']:.3f}" if r["rate"] else "  -  "
        rows.append(f"  {r['line']:4s} {r['E_keV']:10.4f}   rate {rt}   sigma {sg}   weight in group {w}")
    for sh, e in t["edges"].items():
        rows.append(f"  edge {sh}: {e:g} keV")
    return "\n".join(rows)
