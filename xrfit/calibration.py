# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear calibration of a measured signal against known amounts (film thickness, mass per area, ...).

signal = a + b * amount, fitted to the standards; unknowns are converted back with
amount = (signal - a) / b and an uncertainty from the fit and the counting error.
"""
from __future__ import annotations

import math

import numpy as np

from . import core


def roi_signals(spectra, zero, gain, emin, emax, background="none", edge_kev=0.06, normalize="none"):
    """Sum counts between emin and emax (keV, using the given zero/gain), optionally minus a linear
    baseline drawn between two windows just outside the region."""
    if not gain > 0 or emax <= emin:
        raise ValueError("Energy limits must satisfy low < high, and the gain must be positive.")
    out = []
    for s in spectra:
        y = np.asarray(s.counts, float)
        n = len(y)
        i0 = max(0, int(round((emin - zero) / gain)))
        i1 = min(n, int(round((emax - zero) / gain)) + 1)
        if i1 - i0 < 2:
            raise ValueError("The energy window holds fewer than two channels. Check the gain and the limits.")
        tot, var = y[i0:i1].sum(), y[i0:i1].sum()
        if background == "edges":
            w = max(2, int(round(edge_kev / gain)))
            lo, hi = y[max(0, i0 - w):i0], y[i1:i1 + w]
            if len(lo) and len(hi):
                m = i1 - i0
                bg = 0.5 * m * (lo.mean() + hi.mean())
                tot -= bg
                var += (0.5 * m) ** 2 * (lo.sum() / len(lo) ** 2 + hi.sum() / len(hi) ** 2)
        k = 1.0 / s.live_time if normalize == "live_time" and s.live_time > 0 else 1.0
        out.append((tot * k, math.sqrt(max(var, 1.0)) * k))
    return out


def area_signals(spectra, cfg, group, normalize="none"):
    """Net peak area of one line group from a full fit of each spectrum."""
    results = core.fit_batch(spectra, cfg)
    out = []
    for s, r in zip(spectra, results):
        if group not in r.areas:
            have = ", ".join(n for n in r.areas if not n.startswith("sum:"))
            raise ValueError(f"No group named {group}. Available: {have}.")
        a, e = r.areas[group]
        k = 1.0 / s.live_time if normalize == "live_time" and s.live_time > 0 else 1.0
        out.append((a * k, e * k))
    return out


def linfit(x, y, sig=None, through_zero=False):
    """Least-squares line y = a + b x (or y = b x). Weighted by 1/sig^2 when sig is given."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    w = np.ones(n) if sig is None else 1.0 / np.maximum(np.asarray(sig, float), 1e-12) ** 2
    A = x[:, None] if through_zero else np.column_stack([np.ones(n), x])
    npar = A.shape[1]
    if n < npar:
        raise ValueError(f"Need at least {npar} standards with known amounts.")
    if np.ptp(x) == 0 and not through_zero:
        raise ValueError("All standards have the same amount. Add standards at different amounts.")
    W = A * w[:, None]
    cov0 = np.linalg.inv(A.T @ W)
    p = cov0 @ (W.T @ y)
    res = y - A @ p
    dof = n - npar
    chi2 = float(np.sum(w * res ** 2))
    if dof > 0:
        scale = chi2 / dof if sig is None else max(chi2 / dof, 1.0)  # with counting errors, excess scatter inflates them
    else:
        scale = 1.0
    cov = cov0 * scale
    a, b = (0.0, p[0]) if through_zero else (p[0], p[1])
    sb = math.sqrt(cov[-1, -1])
    sa = 0.0 if through_zero else math.sqrt(cov[0, 0])
    xbar = float(np.average(x, weights=w))
    if through_zero:
        r2 = 1 - float(np.sum(w * res ** 2)) / float(np.sum(w * y ** 2))  # uncentred
    else:
        ybar = np.average(y, weights=w)
        ss_tot = float(np.sum(w * (y - ybar) ** 2))
        r2 = 1 - chi2 / ss_tot if ss_tot > 0 else None
    s_y = math.sqrt(float(np.sum(res ** 2)) / dof) if dof > 0 else None  # unweighted scatter of signal
    return {"a": float(a), "b": float(b), "sa": sa, "sb": sb, "r2": r2, "dof": dof, "n": n,
            "scatter": s_y, "xbar": xbar, "through_zero": through_zero}


def calibrate(rows, through_zero=False, weighted=False):
    """rows: [{name, signal, err, amount (None for unknowns), use (bool)}] -> fit, per-row amounts."""
    std = [r for r in rows if r.get("amount") is not None and r.get("use", True)]
    if len(std) < (1 if through_zero else 2):
        raise ValueError("Enter known amounts for at least " + ("one standard." if through_zero else "two standards."))
    x = [r["amount"] for r in std]
    y = [r["signal"] for r in std]
    sig = [r["err"] for r in std] if weighted else None
    f = linfit(x, y, sig, through_zero)
    if f["b"] <= 0:
        raise ValueError("The fitted slope is zero or negative. Check that the amounts and signals belong together.")
    a, b = f["a"], f["b"]
    s_y = f["scatter"]
    out = []
    for r in rows:
        sg, er = r["signal"], r["err"]
        est = (sg - a) / b
        u = max(er, s_y) if s_y is not None else er  # larger of counting error and scatter about the line
        if through_zero:
            v = (u / b) ** 2 + (est * f["sb"] / b) ** 2
        else:
            v = (u / b) ** 2 * (1.0 + 1.0 / f["n"]) + (est - f["xbar"]) ** 2 * (f["sb"] / b) ** 2
        has = r.get("amount") is not None
        row = {"name": r["name"], "signal": sg, "err": er, "amount": r.get("amount"), "use": bool(r.get("use", True)),
               "estimate": est, "estimate_err": math.sqrt(v),
               "fitted_signal": a + b * (r["amount"] if has else est)}
        if has:
            row["resid"] = sg - (a + b * r["amount"])
            row["resid_pct"] = 100.0 * row["resid"] / sg if sg else None
            row["resid_amount"] = est - r["amount"]
        out.append(row)
    low = min(std, key=lambda r: r["signal"])
    lod = 3.0 * max(low["err"], 1e-12) / b
    fit = {k: f[k] for k in ("a", "b", "sa", "sb", "r2", "dof", "n", "scatter", "through_zero")}
    fit["lod"] = lod
    fit["rmse_amount"] = (s_y / b) if s_y is not None else None
    return {"fit": fit, "rows": out}
