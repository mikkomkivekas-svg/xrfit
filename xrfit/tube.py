# SPDX-License-Identifier: GPL-3.0-or-later
"""Approximate X-ray tube spectrum, discretised into a beam table (energy, weight).

PyMca builds an incident spectrum from an energy table (energy, weight, scatter flag); its tube tool fills
that table from an Ebel tube model. The Ebel model needs mass attenuation coefficients for the anode,
window and filters, which xrfit does not carry, so this module is a simplified model:

  continuum        Kramers' law, photons per keV  N(E) ~ Z (E0 - E) / E   for E < E0 (tube voltage)
  characteristic   anode K lines (energies and relative rates from the data tables) when E0 > K edge, otherwise the
                   L lines (weighted by ln(U)/U of their shell); their total share of all photons is `line_share`
No anode self-absorption, window or filter is applied. For quantitative work compute the tube spectrum
elsewhere (PyMca, xraylib) and load it as a table.
"""
import math

from . import xdata

K_LINES = ("Ka1", "Ka2", "Kb1", "Kb3")
L_LINES = ("La1", "Ll", "Lb1", "Lb3", "Lg1")        # the L lines xrfit fits, so the beam and the fit use the same set


def _lines(anode, kv, names, shell_weight=False):
    """[(energy, relative rate)] of the anode lines that the tube voltage can excite."""
    out = []
    for n in names:
        E, r = xdata.line_energy(anode, n), xdata.line_rate(anode, n)
        edge = xdata.edge(anode, xdata.VACANCY[n])
        if E is None or not r or not edge or kv <= edge or E >= kv:
            continue
        if shell_weight:                  # electron impact: ionisation of a shell falls off like ln(U)/U with overvoltage U
            u = kv / edge
            r = r * math.log(u) / u
        out.append((E, r))
    return out


def tube_beam(anode="Rh", kv=40.0, bins=24, emin=1.5, line_share=0.25, l_share=None):
    """-> [{"energy", "weight", "use", "scatter"}] with weights summing to 1.

    `line_share` is the share of all photons in the anode's characteristic lines: K lines when the voltage is above
    the K edge, otherwise the L lines (a 5 kV Rh tube has only L lines). `l_share` adds an L-line share on top of K
    lines (None: none), for a high-voltage tube whose L lines also reach the sample."""
    xdata.check_element(anode)
    if not 2 <= kv <= 450:
        raise ValueError("tube voltage must be between 2 and 450 kV")
    if not 3 <= bins <= 200:
        raise ValueError("use between 3 and 200 continuum bins")
    if not 0 <= line_share < 1 or not 0 <= (l_share or 0) < 1:
        raise ValueError("characteristic share must be from 0 up to (not including) 1")
    emin = min(max(emin, 0.2), kv * 0.5)
    z = (xdata.t12().get(anode) or xdata.xl()[anode])["Z"]
    # continuum: integrate Z (E0/E - 1) over equal-width bins; analytic integral E0 ln E - E
    edges = [emin + (kv - emin) * i / bins for i in range(bins + 1)]
    F = lambda E: kv * math.log(E) - E
    cont = [z * (F(edges[i + 1]) - F(edges[i])) for i in range(bins)]
    ctot = sum(cont)
    k_lines, l_lines = _lines(anode, kv, K_LINES), _lines(anode, kv, L_LINES, True)
    groups = []                           # (lines, share)
    if k_lines and line_share > 0:
        groups.append((k_lines, line_share))
    if l_lines:
        sh = l_share if k_lines else (line_share if l_share is None else l_share)
        if sh and sh > 0:
            groups.append((l_lines, sh))
    total = sum(sh for _, sh in groups)
    if total >= 1:
        raise ValueError("the anode line shares add up to 100 % or more")
    rows = [{"energy": round(0.5 * (edges[i] + edges[i + 1]), 4), "weight": (1 - total) * c / ctot,
             "use": True, "scatter": False} for i, c in enumerate(cont)]
    for lines, sh in groups:
        tot = sum(r for _, r in lines)
        for E, r in lines:
            # anode lines scatter (Rayleigh + Compton); they share one amplitude in the fit
            rows.append({"energy": E, "weight": sh * r / tot, "use": True, "scatter": True})
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
