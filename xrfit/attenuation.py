# SPDX-License-Identifier: GPL-3.0-or-later
"""Mass attenuation coefficients (cm^2/g) from the Elam, Ravel and Sieber tables, via the public-domain XrayDB
compilation (see tools/build_attenuation.py). Interpolation follows the published method: cubic spline in
ln(sigma) versus ln(E) using the tabulated second derivatives."""
import functools
import json
import os

import numpy as np

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "attenuation.json")


@functools.lru_cache(maxsize=1)
def _db():
    with open(_FILE) as f:
        return json.load(f)["elements"]


def symbols():
    return list(_db())


def has(sym):
    return sym in _db()


def atomic_weight(sym):
    return _db()[_check(sym)]["A"]


def density(sym):
    return _db()[_check(sym)]["rho"]


def _check(sym):
    if sym not in _db():
        raise KeyError(f"no attenuation data for '{sym}' (elements H to Cf, Z = 1-98)")
    return sym


@functools.lru_cache(maxsize=None)
def _tab(sym, kind):
    x, y, d = _db()[_check(sym)][kind]
    return np.array(x), np.array(y), np.array(d)


def _spline(xin, yin, ysp, x):
    lo = np.clip(np.searchsorted(xin, x, "left") - 1, 0, len(xin) - 1)
    hi = np.clip(np.searchsorted(xin, x, "right"), 0, len(xin) - 1)
    diff = xin[hi] - xin[lo]
    diff = np.where(diff <= 0, 1.0, diff)
    a, b = (xin[hi] - x) / diff, (x - xin[lo]) / diff
    return a * yin[lo] + b * yin[hi] + (diff * diff / 6) * ((a * a - 1) * a * ysp[lo] + (b * b - 1) * b * ysp[hi])


def mu(sym, energy_keV, kind="total"):
    """Mass attenuation coefficient in cm^2/g of an element. kind: total | photo | coh | incoh.
    Valid from 0.1 keV to 800 keV (clamped outside)."""
    e = np.atleast_1d(np.asarray(energy_keV, float))
    ln_e = np.log(np.clip(e * 1000.0, 100.0, 8.0e5))
    kinds = ("photo", "coh", "incoh") if kind == "total" else (kind,)
    out = np.zeros_like(ln_e)
    for k in kinds:
        out += np.exp(_spline(*_tab(sym, k), ln_e))
    return out if np.ndim(energy_keV) else float(out[0])
