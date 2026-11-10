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


# ---------------------------------------------------------------- thick reference
def _mass_fractions(text, label):
    """'Al 100' or 'Al 52.9, O balance' -> {element: mass fraction} summing to 1."""
    from . import matrix
    ent = matrix.parse_composition(text)
    fixed = sum(e["value"] for e in ent if e["mode"] == "fixed")
    out = {}
    for e in ent:
        if e["mode"] == "unknown":
            raise ValueError(f"{label}: the composition must be known (no ?).")
        out[e["element"]] = e["value"] if e["mode"] == "fixed" else None
    if any(v is None for v in out.values()):
        rest = 1.0 - fixed
        if rest < 0:
            raise ValueError(f"{label}: the amounts add up to more than 100 %.")
        for k, v in out.items():
            if v is None:
                out[k] = rest
    tot = sum(out.values())
    if not tot > 0:
        raise ValueError(f"{label}: enter the composition.")
    return {k: v / tot for k, v in out.items()}


def _density(text, frac, rho, label):
    from . import attenuation
    if rho in (None, ""):
        if len(frac) == 1:
            return float(attenuation.density(next(iter(frac))))
        raise ValueError(f"{label}: enter the density (g/cm³).")
    rho = float(rho)
    if not rho > 0:
        raise ValueError(f"{label}: the density must be above 0.")
    return rho


def film_mass(rows, ref_index, cfg, line, film, ref, substrate, scale_to_known=False):
    """Convert the signals of thin films into film mass per area, with a thick sample as the reference.

    The signal of a film of mass m (g/cm2) on a substrate, divided by the signal of the infinitely thick
    reference, is R(m) from the matrix model (geometry, tube spectrum, attenuation in the film). Geometry,
    solid angle, air and detector cancel in the ratio. R(m) is inverted numerically for each film.
    rows: [{name, signal, err, amount}], signal per second of live time (or counts, if every spectrum has the same live time).
    film/ref/substrate: {composition, density}.  Returns dict with rows and the model scale."""
    from . import matrix
    if not 0 <= ref_index < len(rows):
        raise ValueError("Choose which spectrum is the thick reference.")
    r0 = rows[ref_index]
    if not r0["signal"] > 0:
        raise ValueError("The reference has no signal in this window.")
    ffr = _mass_fractions(film.get("composition", ""), "Film")
    rfr = _mass_fractions(ref.get("composition", ""), "Reference")
    sfr = _mass_fractions(substrate.get("composition", ""), "Substrate")
    rho_f = _density(film.get("composition"), ffr, film.get("density"), "Film")
    rho_r = _density(ref.get("composition"), rfr, ref.get("density"), "Reference")
    rho_s = _density(substrate.get("composition"), sfr, substrate.get("density"), "Substrate")
    sym = str(line).split("-")[0]
    if sym not in ffr or sym not in rfr:
        raise ValueError(f"The film and the reference must both contain {sym}.")
    mx = dict(cfg.matrix or {})
    mx.update({"secondary": False, "layers": [
        {"composition": film["composition"], "density": rho_f, "thickness_um": 1.0},
        {"composition": substrate["composition"], "density": rho_s}]})
    exc = core.excitation(cfg)
    model = matrix.Model(mx, [line], exc, cfg.xs_source)
    name = next(iter(model.groups))
    i_ref = model.predict([rfr], [float("inf")])[name]
    if not i_ref > 0:
        raise ValueError("The model predicts no signal from the reference. Check the line and the composition.")
    grid = np.logspace(-8, -2.0, 90)                                 # film mass, g/cm2 (1e-4 g/m2 ... 100 g/m2)
    rr = np.array([model.predict([ffr, sfr], [m, float("inf")])[name] for m in grid]) / i_ref
    if np.any(np.diff(rr) <= 0):
        keep = np.concatenate([[True], np.diff(rr) > 0])
        grid, rr = grid[keep], rr[keep]
    r_max = float(rr[-1])
    a = rr[0] / grid[0]
    def invert(R):
        if R <= 0:
            return 0.0
        if R < rr[0]:
            return R / a
        return float(np.exp(np.interp(np.log(R), np.log(rr), np.log(grid))))
    sc = float(grid[np.searchsorted(rr, 0.632 * r_max)]) if r_max > 0.632 else float("nan")
    wsym = ffr[sym]
    out = []
    for i, r in enumerate(rows):
        if i == ref_index:
            out.append({"name": r["name"], "is_reference": True, "ratio": 1.0})
            continue
        R = r["signal"] / r0["signal"]
        eR = R * math.hypot(r["err"] / r["signal"] if r["signal"] else 0.0, r0["err"] / r0["signal"])
        m = invert(R)
        d = (invert(R * 1.001) - invert(R * 0.999)) / (0.002 * R) if R > 0 else 1.0 / a
        out.append({"name": r["name"], "is_reference": False, "ratio": R, "ratio_err": eR, "model_mass_g_m2": m * 1e4,
                    "model_mass_err_g_m2": abs(d) * eR * 1e4, "above_reference": bool(R >= r_max),
                    "known": r.get("amount")})
    factor, n_known = 1.0, 0
    if scale_to_known:
        pairs = [(o["known"], o["model_mass_g_m2"]) for o in out if not o["is_reference"] and o.get("known") is not None and o["model_mass_g_m2"] > 0]
        n_known = len(pairs)
        if not pairs:
            raise ValueError("Enter known film masses (g/m²) for at least one film to scale the model.")
        kn = np.array([p[0] for p in pairs], float); md = np.array([p[1] for p in pairs], float)
        factor = float(np.sum(kn * md) / np.sum(md * md))   # least squares through the origin, known = factor * model
    for o in out:
        if o["is_reference"]:
            continue
        o["mass_g_m2"] = o["model_mass_g_m2"] * factor
        o["mass_err_g_m2"] = o["model_mass_err_g_m2"] * factor
        o["element_g_m2"] = o["mass_g_m2"] * wsym
        o["thickness_nm"] = o["mass_g_m2"] / rho_f * 1000.0        # g/m2 divided by g/cm3 is 1e-6 m3/m2 = 1000 nm per unit
    return {"rows": out, "line": name, "element": sym, "element_fraction": wsym, "film_density": rho_f,
            "ratio_at_saturation": r_max, "saturation_mass_g_m2": sc * 1e4 if sc == sc else None,
            "slope_ratio_per_g_m2": a * 1e-4, "scale_factor": factor, "n_known": n_known,
            "monochromatic": not bool(cfg.beam)}
