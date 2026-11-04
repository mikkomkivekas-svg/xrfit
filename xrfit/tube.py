# SPDX-License-Identifier: GPL-3.0-or-later
"""Approximate X-ray tube spectrum, discretised into a beam table (energy, weight).

PyMca builds an incident spectrum from an energy table (energy, weight, scatter flag); its tube tool fills
that table from an Ebel tube model. The Ebel model needs mass attenuation coefficients for the anode,
window and filters, which xrfit does not carry, so this module is a simplified model:

  continuum        Kramers' law, photons per keV  N(E) ~ Z (E0 - E) / E   for E < E0 (tube voltage)
  characteristic   anode K lines (energies and relative rates from the data tables), present when E0 > K edge;
                   their total share of all photons is set by `line_share`
No anode self-absorption, window or filter is applied. For quantitative work compute the tube spectrum
elsewhere (PyMca, xraylib) and load it as a table.
"""
import math

from . import xdata

K_LINES = ("Ka1", "Ka2", "Kb1", "Kb3")


def tube_beam(anode="Rh", kv=40.0, bins=24, emin=1.5, line_share=0.25):
    """-> [{"energy", "weight", "use", "scatter"}] with weights summing to 1."""
    xdata.check_element(anode)
    if not 2 <= kv <= 450:
        raise ValueError("tube voltage must be between 2 and 450 kV")
    if not 3 <= bins <= 200:
        raise ValueError("use between 3 and 200 continuum bins")
    if not 0 <= line_share < 1:
        raise ValueError("characteristic share must be from 0 up to (not including) 1")
    emin = min(max(emin, 0.2), kv * 0.5)
    z = (xdata.t12().get(anode) or xdata.xl()[anode])["Z"]
    # continuum: integrate Z (E0/E - 1) over equal-width bins; analytic integral E0 ln E - E
    edges = [emin + (kv - emin) * i / bins for i in range(bins + 1)]
    F = lambda E: kv * math.log(E) - E
    cont = [z * (F(edges[i + 1]) - F(edges[i])) for i in range(bins)]
    ctot = sum(cont)
    rows = [{"energy": round(0.5 * (edges[i] + edges[i + 1]), 4), "weight": (1 - line_share) * c / ctot,
             "use": True, "scatter": False} for i, c in enumerate(cont)]
    k_edge = xdata.edge(anode, "K")
    lines = []
    if k_edge and kv > k_edge:
        for n in K_LINES:
            E, r = xdata.line_energy(anode, n), xdata.line_rate(anode, n)
            if E is not None and r and E < kv:
                lines.append((E, r))
    if lines and line_share > 0:
        tot = sum(r for _, r in lines)
        rows += [{"energy": E, "weight": line_share * r / tot, "use": True, "scatter": False} for E, r in lines]
        rows[len(rows) - len(lines)]["scatter"] = True      # main line scatters (Rayleigh + Compton)
    else:
        for r in rows: r["weight"] /= (1 - line_share) or 1
    return sorted(rows, key=lambda r: r["energy"])


def parse_beam_text(text):
    """Read a beam table from text: one energy per line, "energy [weight [scatter]]" separated by spaces, commas or
    semicolons. '#' starts a comment; lines whose first field is not a number (headers) are skipped. Energies in keV."""
    import re
    rows = []
    for line in text.splitlines():
        line = re.sub(r"[#;].*$", "", line).strip()
        if not line:
            continue
        f = [t for t in re.split(r"[\s,;]+", line) if t]
        try:
            E = float(f[0]); w = float(f[1]) if len(f) > 1 else 1.0
        except ValueError:
            continue
        rows.append({"energy": E, "weight": w, "use": True,
                     "scatter": len(f) > 2 and f[2].lower() in ("1", "true", "yes", "y", "x", "on")})
    if not rows:
        raise ValueError('no energies found; use lines like "17.44 1.0 1" (keV, weight, optional scatter flag)')
    return rows
