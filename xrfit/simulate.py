# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectrum simulator: what the detector sees for a tube, a sample and a set of filter options (no fitting).

Simple on purpose: Gaussian peaks (plus the Si escape), primary fluorescence from the matrix model, Rayleigh and
Compton scatter of the whole tube spectrum, and the argon of the air path. The numbers are relative: scenario 1 is
scaled to the chosen input count rate and the others use the same instrument factor, so the differences between
filter options are meaningful even when the absolute level is only approximate."""
import copy
import math

import numpy as np

from . import attenuation, calibration, core, lines, matrix, tube, xdata

ER_MC2 = core.ELECTRON_MC2
KLINES = ("Ka1", "Ka2", "Kb1", "Kb3")
LLINES = ("La1", "Lb1", "Lb3", "Lg1", "Ll")


def _eff(det, filters, E, air_mm=None):
    """Detection efficiency at energies E: Be window, Si absorption, filters, air (averaged over the beam spot)."""
    E = np.atleast_1d(np.asarray(E, float))
    t = np.ones_like(E)
    be = float(det.get("be_um", 0)) * 1e-4
    si = float(det.get("si_mm", 0.5)) / 10
    if be > 0:
        t *= np.exp(-attenuation.mu("Be", E) * matrix.BE_RHO * be)
    if si > 0:
        t *= 1 - np.exp(-attenuation.mu("Si", E) * matrix.SI_RHO * si)
    t *= matrix.filter_transmission(filters, E)
    air = float(det.get("air_mm", 0) if air_mm is None else air_mm)
    if air > 0:
        mu = sum(w * attenuation.mu(s, E) for s, w in matrix.AIR.items()) * matrix.AIR_RHO
        t *= matrix.spot_factor(mu, air, float(det.get("spot_mm") or 0), det.get("spot_profile", "uniform"))
    return t


def _sample(mx):
    fr, mass = [], []
    for i, L in enumerate(mx["layers"]):
        f = calibration._mass_fractions(L.get("composition", ""), f"Layer {i + 1}")
        rho = calibration._density(L.get("composition"), f, L.get("density"), f"Layer {i + 1}")
        fr.append(f)
        last = i == len(mx["layers"]) - 1
        mass.append(math.inf if last else rho * float(L.get("thickness_um") or 0) * 1e-4)
    return fr, mass


def _line_names(sym, kv):
    out = []
    for n in KLINES + LLINES + ("Ma1",):
        E = xdata.line_energy(sym, n)
        ed = xdata.edge(sym, xdata.VACANCY[n])
        if E is None or ed is None or kv <= ed or E < 0.3:
            continue
        try:
            xdata.xs_line(sym, n, kv, "xraylib")
        except KeyError:
            continue
        out.append(n)
    return out


def simulate(req):
    tb = req.get("tube") or {}
    kv = float(tb.get("kv", 5))
    rows = tube.tube_beam(tb.get("anode", "Rh"), kv, int(tb.get("bins", 80)), float(tb.get("emin", 0.3)),
                          float(tb.get("line_share", 0.1)))
    E0 = np.array([r["energy"] for r in rows]); W0 = np.array([r["weight"] for r in rows])
    mx = copy.deepcopy(req["matrix"]); mx["secondary"] = False
    det = mx.get("detector") or {}
    fr, mass = _sample(mx)
    s_in, s_out = math.sin(math.radians(float(mx.get("alpha_in", 90)))), math.sin(math.radians(float(mx.get("alpha_out", 45))))
    theta = float(req.get("scatter_angle", 133))
    d = req.get("detector") or {}
    gain, zero = float(d.get("gain", 0.00487)), float(d.get("zero", 0))
    p = {"zero": zero, "gain": gain, "noise": float(d.get("noise", 0.047)), "fano": float(d.get("fano", 0.122)),
         "tail_frac": 0, "beta": 0.1, "step_frac": 0, "esc_frac": 0, "esc_scale": float(d.get("esc_scale", 1.0)),
         "tail2_frac": 0, "beta2": 0.6}
    cb = float(d.get("compton_broad", 0.2))
    emax = float(req.get("emax") or kv + 0.5)
    nch = int((emax - zero) / gain) + 1
    e = zero + gain * np.arange(nch)
    ang = 0.75 * (1 + math.cos(math.radians(theta)) ** 2)
    ar_scale = float(req.get("ar_scale", 1.0))
    elems = sorted({s for f in fr for s in f})
    names = [f"{s}-{n}" for s in elems for n in _line_names(s, kv)]
    scen = req.get("scenarios") or [{}]
    out, K = [], None
    for sc in scen:
        bf, df = sc.get("beam_filters") or [], sc.get("det_filters") or []
        t = matrix.filter_transmission(bf, E0) if bf else np.ones_like(E0)
        wt = W0 * t
        keep = wt > 1e-9 * wt.max()
        Ek, wk = E0[keep], wt[keep]
        comps = {"Fluorescence": np.zeros(nch), "Rayleigh": np.zeros(nch), "Compton": np.zeros(nch), "Air argon": np.zeros(nch)}
        table = {}
        # fluorescence
        m2 = copy.deepcopy(mx); m2.setdefault("detector", {})["filters"] = df
        if names:
            exc = tuple(zip(Ek.tolist(), (wk / wk.sum()).tolist()))
            model = matrix.Model(m2, names, exc, "xraylib")
            tot = wk.sum()
            rates = model.predict(fr, mass)
            for gname, (sym, ls) in model.groups.items():
                R = rates[gname] * tot
                if R <= 0:
                    continue
                El = ls[0][0]
                comps["Fluorescence"] += R * core.peak(e, El, p, gain)
                fam = "K" if ln_is_k(gname) else ("M" if gname.endswith("Ma1") else "L")
                key = f"{sym} {fam}"
                table[key] = table.get(key, 0.0) + R
        # scatter of every tube photon
        c = 1 - math.cos(math.radians(theta))
        Ec = Ek / (1 + Ek * c / ER_MC2)
        for kind, Eo in (("Rayleigh", Ek), ("Compton", Ec)):
            ek = "coh" if kind == "Rayleigh" else "incoh"
            eff = _eff(det, df, Eo)
            rate = np.zeros(len(Ek)); above = np.ones(len(Ek))
            for f, M in zip(fr, mass):
                sg = sum(w * attenuation.mu(s, Ek, ek) for s, w in f.items())
                chi = sum(w * attenuation.mu(s, Ek) for s, w in f.items()) / s_in + \
                    sum(w * attenuation.mu(s, Eo) for s, w in f.items()) / s_out
                layer = 1 / (chi * s_in) if math.isinf(M) else -np.expm1(-chi * M) / (chi * s_in)
                rate += sg * above * layer
                above = above * (0.0 if math.isinf(M) else np.exp(-chi * M))
            rate = wk * ang * rate * eff
            for Ei, Eoi, ri in zip(Ek, Eo, rate):
                if ri > 0:
                    pp = p if kind == "Rayleigh" else dict(p, noise=math.hypot(p["noise"], cb * Eoi / 20))
                    if e[0] - 1 < Eoi < e[-1] + 1:
                        comps[kind] += ri * core.peak(e, Eoi, pp, gain)
            table[kind] = float(rate.sum())
        # argon in the air path (primary excitation of the air column, emission through half of it)
        L = float(det.get("air_mm", 0))
        if L > 0:
            ar = 0.0
            for n in KLINES:
                Ea = xdata.line_energy("Ar", n)
                sig = np.array([xdata.xs_line("Ar", n, E, "xraylib")[0] for E in Ek])
                eff = float(_eff(det, df, Ea, air_mm=L / 2)[0])
                R = float(np.sum(wk * sig)) * 6.022e23 * 1e-24 / attenuation.atomic_weight("Ar") * matrix.AIR["Ar"] \
                    * matrix.AIR_RHO * (L / 10) * eff * ar_scale
                if R > 0:
                    comps["Air argon"] += R * core.peak(e, Ea, p, gain)
                    ar += R
            table["Ar K (air)"] = ar
        out.append({"name": sc.get("name") or f"Option {len(out) + 1}", "comps": comps, "table": table})
    total0 = sum(float(c.sum()) for c in out[0]["comps"].values())
    if not total0 > 0:
        raise ValueError("the first option gives no signal; check the sample and the filters")
    K = float(req.get("input_kcps", 50)) * 1000 / total0
    res = []
    for o in out:
        spec = sum(o["comps"].values()) * K
        res.append({"name": o["name"], "spectrum": np.round(spec, 5).tolist(),
                    "components": {k: np.round(v * K, 5).tolist() for k, v in o["comps"].items()},
                    "lines": {k: v * K for k, v in o["table"].items() if v > 0},
                    "total_cps": float(spec.sum())})
    return {"energy": np.round(e, 5).tolist(), "scenarios": res, "gain": gain}


def ln_is_k(gname):
    return gname.split("-")[1][0] == "K"
