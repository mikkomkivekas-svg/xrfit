# SPDX-License-Identifier: GPL-3.0-or-later
"""Sample matrix: a stack of layers (the last one infinitely thick) and the fundamental-parameter calculation that
links fitted net areas to composition.

Model (primary excitation only, see README for what is left out). For element i with mass fraction w in layer j and
line l (energy E_l), excited by beam energy E0 (relative weight b), the detected rate per unit instrument factor is

    b * sigma_l(E0) * N_A * 1e-24 / A_i * w * T_above * (1 - exp(-chi M)) / (chi sin(a_in)) * eff(E_l)

    chi = mu_j(E0)/sin(a_in) + mu_j(E_l)/sin(a_out)       M = density * thickness (g/cm^2; infinite for the last layer)
    T_above = exp(-sum over the layers above of chi_k M_k)  eff = detector path (air, Be window) x Si absorption

sigma_l is the production cross section (barn/atom) used everywhere else in xrfit; mu is the Elam mass attenuation.
The instrument factor K lumps source flux, solid angle and time: rate = K * (the expression above).
"""
import math
import re

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expn

from . import attenuation, core, lines, xdata

N_A = 6.02214076e23
AIR = {"N": 0.7552, "O": 0.2314, "Ar": 0.0128}   # mass fractions of dry air
AIR_RHO, BE_RHO, SI_RHO = 1.205e-3, 1.848, 2.33


class MatrixError(ValueError):
    pass


# ------------------------------------------------------------- composition text
_FORMULA = re.compile(r"^[A-Za-z0-9().]+$")
_UNKNOWN = {"?", "*", "unknown", "x"}
_BALANCE = {"balance", "rest", "remainder", "bal"}


def _elem(tok):
    t = tok if tok[:1].isupper() and (len(tok) == 1 or tok[1:].islower()) else tok.capitalize()
    if not attenuation.has(t):
        raise MatrixError(f"'{tok}' is not an element symbol (H to Cf)")
    return t


def parse_formula(f):
    """'Fe2O3' or '(NH4)2SO4' -> {element: mass fraction}."""
    pos = 0

    def group():
        nonlocal pos
        counts = {}
        while pos < len(f):
            c = f[pos]
            if c == "(":
                pos += 1
                inner = group()
                if pos >= len(f) or f[pos] != ")":
                    raise MatrixError(f"unbalanced parentheses in '{f}'")
                pos += 1
                m = re.match(r"\d+(\.\d+)?", f[pos:])
                n = float(m.group()) if m else 1.0
                pos += len(m.group()) if m else 0
                for k, v in inner.items():
                    counts[k] = counts.get(k, 0) + v * n
            elif c == ")":
                return counts
            elif c.isupper():
                m = re.match(r"[A-Z][a-z]?", f[pos:])
                sym = m.group()
                if not attenuation.has(sym):
                    if len(sym) == 2 and attenuation.has(sym[0]):
                        sym = sym[0]
                    else:
                        raise MatrixError(f"'{sym}' in formula '{f}' is not an element")
                pos += len(sym)
                m = re.match(r"\d+(\.\d+)?", f[pos:])
                n = float(m.group()) if m else 1.0
                pos += len(m.group()) if m else 0
                counts[sym] = counts.get(sym, 0) + n
            else:
                raise MatrixError(f"cannot read '{f}' as a chemical formula")
        return counts

    counts = group()
    if pos != len(f) or not counts:
        raise MatrixError(f"cannot read '{f}' as a chemical formula")
    mass = {s: n * attenuation.atomic_weight(s) for s, n in counts.items()}
    tot = sum(mass.values())
    return {s: m / tot for s, m in mass.items()}


def _compound(tok):
    """Element symbol or formula token -> {element: mass fraction}. 'fe' is iron; 'CO' is carbon monoxide, 'Co' cobalt."""
    if tok.islower():
        return {_elem(tok): 1.0}
    try:
        return parse_formula(tok)
    except MatrixError:
        if tok.isalpha():
            return {_elem(tok): 1.0}
        raise


def parse_composition(text):
    """Composition text -> list of {"element", "mode": fixed|unknown|balance, "value": mass fraction or None}.

    One entry per line (or separated by , or ;):
        Fe 70          70 wt % iron          Cr ?      amount to be determined from the fitted lines
        SiO2 30        a compound by weight  O balance takes whatever is left to 100 %
        Fe2O3          a lone formula or element is the whole layer
    """
    raw = [p.strip() for p in re.split(r"[\n;,]+", text or "") if p.strip()]
    if not raw:
        raise MatrixError("enter the composition of this layer, for example: Fe 70, Cr ?, O balance")
    toks = []
    for p in raw:
        mt = re.match(r"^([A-Za-z0-9().]+)\s*(?:=|\s)\s*(.*)$", p)
        toks.append([mt.group(1), mt.group(2) or ("balance" if "=" in p else "")] if mt else [p])
        if len(toks[-1]) == 2 and not toks[-1][1]:
            toks[-1] = toks[-1][:1]
    out = {}   # element -> [mode, value]

    def put(sym, mode, val):
        if sym in out:
            if out[sym][0] != mode or mode != "fixed":
                raise MatrixError(f"{sym} is listed more than once")
            out[sym][1] += val
        else:
            out[sym] = [mode, val]

    if len(toks) == 1 and len(toks[0]) == 1:       # a lone formula or element
        for s, w in _compound(toks[0][0]).items():
            put(s, "fixed", w)
        return [{"element": s, "mode": m, "value": v} for s, (m, v) in out.items()]
    for t in toks:
        if len(t) == 1:
            raise MatrixError(f"'{t[0]}': add an amount (wt %), ? or balance")
        name, val = t[0], t[1].strip().rstrip("%").strip()
        if val.lower() in _UNKNOWN:
            put(_elem(name), "unknown", None)
        elif val.lower() in _BALANCE:
            put(_elem(name), "balance", None)
        else:
            try:
                v = float(val)
            except ValueError:
                raise MatrixError(f"'{t[1]}' after {name} is not a number, ? or balance") from None
            if not 0 <= v <= 100:
                raise MatrixError(f"{name}: {v} wt % is outside 0 to 100")
            for s, w in _compound(name).items():
                put(s, "fixed", v / 100 * w)
    if sum(1 for m, _ in out.values() if m == "balance") > 1:
        raise MatrixError("only one element per layer can be the balance")
    return [{"element": s, "mode": m, "value": v} for s, (m, v) in out.items()]


# ------------------------------------------------------------------ filters
FILTER_PRESETS = {      # name -> (composition text, density g/cm3)
    "Al": ("Al", 2.70), "Ti": ("Ti", 4.51), "Cu": ("Cu", 8.96), "Zr": ("Zr", 6.51), "Mo": ("Mo", 10.22),
    "Rh": ("Rh", 12.41), "Ag": ("Ag", 10.49), "Be": ("Be", 1.848),
    "Kapton": ("C22H10N2O5", 1.42), "Mylar": ("C10H8O4", 1.38), "Polypropylene": ("C3H6", 0.90),
}


def parse_filters(filters):
    """[{"material", "density" (blank = element default), "thickness_um"}] -> [(composition, density, thickness cm, label)]."""
    out = []
    for i, f in enumerate(filters or [], 1):
        label = f"Filter {i}"
        mat = str(f.get("material", "")).strip()
        if mat in FILTER_PRESETS and not str(f.get("density", "")).strip():
            mat_text, rho = FILTER_PRESETS[mat]
        else:
            mat_text, rho = mat, f.get("density")
        if not mat_text:
            raise MatrixError(f"{label}: enter a material, for example Al or Kapton")
        try:
            ent = parse_composition(mat_text)
        except MatrixError as e:
            raise MatrixError(f"{label}: {e}") from None
        if any(e["mode"] != "fixed" for e in ent):
            raise MatrixError(f"{label}: give the material as an element or formula (no ? or balance)")
        tot = sum(e["value"] for e in ent)
        comp = {e["element"]: e["value"] / tot for e in ent}
        if rho in (None, ""):
            if len(comp) != 1:
                raise MatrixError(f"{label}: enter the density (g/cm³) of {mat_text}")
            rho = attenuation.density(next(iter(comp)))
        try:
            rho = float(rho); t = float(f.get("thickness_um"))
        except (TypeError, ValueError):
            raise MatrixError(f"{label}: enter the density and the thickness in µm as numbers") from None
        if not rho > 0 or t < 0:
            raise MatrixError(f"{label}: the density must be above 0 and the thickness cannot be negative")
        out.append((comp, rho, t * 1e-4, f"{mat} {f.get('thickness_um')} µm"))
    return out


def filter_transmission(filters, energy_keV):
    """Fraction of photons passing all filters at each energy (keV)."""
    e = np.atleast_1d(np.asarray(energy_keV, float))
    t = np.ones_like(e)
    for comp, rho, cm, _ in parse_filters(filters):
        t *= np.exp(-sum(w * attenuation.mu(s, e) for s, w in comp.items()) * rho * cm)
    return t


# ------------------------------------------------------------------- model
def _layers(mx):
    ls = mx.get("layers") or []
    if not ls:
        raise MatrixError("describe at least one layer")
    out = []
    for i, L in enumerate(ls):
        last = i == len(ls) - 1
        label = f"Layer {i + 1}"
        try:
            ent = parse_composition(L.get("composition", ""))
        except MatrixError as e:
            raise MatrixError(f"{label}: {e}") from None
        rho = L.get("density")
        if rho in (None, ""):
            if len(ent) == 1:
                rho = attenuation.density(ent[0]["element"])
            else:
                raise MatrixError(f"{label}: enter the density (g/cm³)")
        rho = float(rho)
        if not rho > 0:
            raise MatrixError(f"{label}: the density must be above 0")
        t = None
        if not last:
            try:
                t = float(L.get("thickness_um"))
            except (TypeError, ValueError):
                raise MatrixError(f"{label}: enter the thickness in µm (only the last layer is infinitely thick)") from None
            if t < 0:
                raise MatrixError(f"{label}: the thickness cannot be negative")
        out.append({"name": (L.get("name") or "").strip() or label, "entries": ent, "rho": rho, "t_um": t,
                    "fit_t": bool(L.get("fit_thickness")) and not last, "reference": (L.get("reference") or "").strip()})
    return out


class Model:
    """Builds the forward calculation for the measured groups."""

    def __init__(self, mx, group_names, exc, source):
        self.mx, self.source, self.exc = mx, source, exc
        self.layers = _layers(mx)
        try:
            self.a_in, self.a_out = (math.radians(float(mx.get("alpha_in", 45))), math.radians(float(mx.get("alpha_out", 45))))
        except (TypeError, ValueError):
            raise MatrixError("the incidence and take-off angles must be numbers") from None
        if not (0 < self.a_in <= math.pi / 2 and 0 < self.a_out <= math.pi / 2):
            raise MatrixError("the incidence and take-off angles must be between 0 and 90 degrees")
        self.s_in, self.s_out = math.sin(self.a_in), math.sin(self.a_out)
        det = mx.get("detector") or {}
        self.si_cm, self.be_cm, self.air_cm = (float(det.get("si_mm", 0.5)) / 10, float(det.get("be_um", 0)) * 1e-4,
                                               float(det.get("air_mm", 0)) / 10)
        self.det_filters = det.get("filters") or []
        parse_filters(self.det_filters)       # fail early with a readable message
        self.beam = lines.beam_energies(exc)
        self.E0 = np.array([E for E, _ in self.beam]); self.wb = np.array([w for _, w in self.beam])
        self.elems = sorted({e["element"] for L in self.layers for e in L["entries"]})
        self.groups = {}
        for name in group_names:
            sym, members = lines.group_members(name, exc, source)
            if sym not in self.elems:
                continue
            ls = []
            for n in members:
                El = xdata.line_energy(sym, n)
                if El is None:
                    continue
                try:
                    sig = np.array([xdata.xs_line(sym, n, E, source)[0] for E in self.E0])
                except KeyError as e:
                    raise MatrixError(f"{name}: {core_msg(e)}") from None
                ls.append((El, sig, n))
            self.groups[name] = (sym, ls)
        # lines of every matrix element that can excite other lines (secondary excitation)
        self.secondary = bool(mx.get("secondary", True))
        self.src = []
        if self.secondary:
            for sym in self.elems:
                for n in lines._K + lines._L:
                    El = xdata.line_energy(sym, n)
                    if El is None:
                        continue
                    try:
                        sig = np.array([xdata.xs_line(sym, n, E, source)[0] for E in self.E0])
                    except KeyError:
                        continue
                    if sig.max() > 0:
                        self.src.append((sym, n, El, sig))
        self._xs_cache = {}
        # attenuation of every matrix element at every energy needed
        need = set(self.E0.tolist())
        for _, ls in self.groups.values():
            need.update(El for El, _, _ in ls)
        need.update(El for _, _, El, _ in self.src)
        self.E_all = np.array(sorted(need))
        self.idx = {round(E, 9): i for i, E in enumerate(self.E_all)}
        self.mu = {s: attenuation.mu(s, self.E_all) for s in self.elems}
        self.eff = {}
        for El in self.E_all:
            e = 1.0
            if self.be_cm > 0:
                e *= math.exp(-attenuation.mu("Be", El) * BE_RHO * self.be_cm)
            if self.air_cm > 0:
                mu_air = sum(w * attenuation.mu(s, El) for s, w in AIR.items())
                e *= math.exp(-mu_air * AIR_RHO * self.air_cm)
            if self.si_cm > 0:
                e *= 1 - math.exp(-attenuation.mu("Si", El) * SI_RHO * self.si_cm)
            self.eff[round(El, 9)] = e * float(filter_transmission(self.det_filters, El)[0])

    def _mu_layer(self, frac, E):
        i = [self.idx[round(x, 9)] for x in np.atleast_1d(E)]
        return sum(w * self.mu[s][i] for s, w in frac.items())

    def _xs_at(self, sym, line, E):
        k = (sym, line, round(E, 9))
        if k not in self._xs_cache:
            try:
                self._xs_cache[k] = xdata.xs_line(sym, line, E, self.source)[0]
            except KeyError:
                self._xs_cache[k] = 0.0
        return self._xs_cache[k]

    def _cells(self, fracs, mass, targets, sources):
        """Split the sample into thin cells for the secondary-excitation integrals -> (layer index, mass) arrays."""
        lay, M = [], []
        for j, m in enumerate(mass):
            if math.isinf(m):
                fr = fracs[j]
                mus = lambda E: float(self._mu_layer(fr, [E])[0])
                dt = max((12 * self.s_out / mus(E) for E in targets), default=1.0)
                db = 14 * self.s_in / min(mus(E) for E in self.E0)
                reach = max((6.0 / mus(E) for E in sources), default=0.0)
                top = min(dt, db) + reach
                n = 36
                edges = top * (np.arange(n + 1) / n) ** 3
                lay += [j] * n; M += list(np.diff(edges))
            elif m > 0:
                n = 14
                lay += [j] * n; M += [m / n] * n
        return np.array(lay), np.array(M)

    def _secondary(self, fracs, mass):
        """Rates per unit K from lines of the sample that excite other lines of the sample (isotropic emission,
        exponential-integral kernel, thin-cell discretisation; exact cell-to-cell double integrals)."""
        out = {n: 0.0 for n in self.groups}
        pairs = []     # (group, element, line, E_line, source element, source line, E', sig0, xs of target line at E')
        for name, (sym, ls) in self.groups.items():
            for El, _, ln in ls:
                for (ks, kn, Ep, sig0) in self.src:
                    xs = self._xs_at(sym, ln, Ep)
                    if xs > 0:
                        pairs.append((name, sym, ln, El, ks, kn, Ep, sig0, xs))
        if not pairs:
            return out
        lay, M = self._cells(fracs, mass, {p[3] for p in pairs}, {p[6] for p in pairs})
        N = len(M)
        if N == 0:
            return out
        w_cell = lambda sym: np.array([fracs[j].get(sym, 0.0) for j in lay])
        def mu_cell(E):
            per = np.array([float(self._mu_layer(fr, [E])[0]) for fr in fracs])
            return per[lay]
        a0 = np.array([mu_cell(E) / self.s_in for E in self.E0])               # (nb, N)
        Bs = np.cumsum(a0 * M, axis=1) - a0 * M                               # beam attenuation above each cell
        a0M = a0 * M
        prim_layer = -np.expm1(-a0M) / (a0 * self.s_in)
        by_src = {}
        for pr in pairs:
            by_src.setdefault((pr[4], pr[5], pr[6]), []).append(pr)
        for (ks, kn, Ep), grp in by_src.items():
            sig0 = grp[0][7]
            Ak = attenuation.atomic_weight(ks)
            P = np.sum((self.wb * sig0 * N_A * 1e-24 / Ak)[:, None] * w_cell(ks)[None, :] * np.exp(-Bs) * prim_layer, axis=0)   # photons emitted per cell
            if not np.any(P > 0):
                continue
            p = mu_cell(Ep); tau = p * M
            C = np.cumsum(tau) - tau                                          # optical depth above each cell
            D = np.zeros((N, N))
            e3 = lambda x: expn(3, x)
            top, bot = C[:, None] + tau[:, None], C[None, :]                  # bottom of source s, top of target t
            cgap = np.maximum(bot - top, 0.0)                                 # valid for t below s
            num = e3(cgap) - e3(cgap + tau[:, None]) - e3(cgap + tau[None, :]) + e3(cgap + tau[:, None] + tau[None, :])
            Dt = num / (p[:, None] * p[None, :])
            iu = np.triu_indices(N, 1)
            D[iu] = Dt[iu]; D = D + D.T
            diag = 2 * (M / p + (e3(tau) - 0.5) / p ** 2)
            D[np.diag_indices(N)] = diag
            src_w = (P / M)[:, None] * 0.5 * D                                # (s, t)
            for (name, sym, ln, El, _, _, _, _, xs) in grp:
                Ai = attenuation.atomic_weight(sym)
                mo = mu_cell(El) / self.s_out
                A_t = np.cumsum(mo * M) - mo * M
                x = mo * M
                esc = np.exp(-A_t) * np.where(x > 1e-12, -np.expm1(-x) / np.where(x > 1e-12, x, 1), 1.0)
                St = xs * N_A * 1e-24 / Ai * w_cell(sym) * esc * self.eff[round(El, 9)]
                out[name] += float(np.sum(src_w.sum(axis=0) * St))
        return out

    def predict(self, fracs, mass, secondary=False):
        """Rates per unit K for every group. fracs: [{element: mass fraction}] per layer; mass: g/cm^2 (inf last)."""
        out = {}
        for name, (sym, ls) in self.groups.items():
            tot = 0.0
            for El, sig, _ in ls:
                eff = self.eff[round(El, 9)]
                above = np.ones(len(self.E0))
                for j, fr in enumerate(fracs):
                    chi = self._mu_layer(fr, self.E0) / self.s_in + float(self._mu_layer(fr, [El])[0]) / self.s_out
                    M = mass[j]
                    layer = 1.0 / (chi * self.s_in) if math.isinf(M) else -np.expm1(-chi * M) / (chi * self.s_in)
                    w = fr.get(sym, 0.0)
                    if w > 0:
                        tot += float(np.sum(self.wb * sig * (N_A * 1e-24 / attenuation.atomic_weight(sym)) * w * above * layer)) * eff
                    above = above * (np.exp(-chi * M) if not math.isinf(M) else 0.0)
            out[name] = tot
        if secondary and self.secondary:
            for n, v in self._secondary(fracs, mass).items():
                out[n] += v
        return out


def _reference(text, elements):
    """Certified or expected composition text -> ({element: wt % renormalised to `elements`}, wt % outside them)."""
    ent = parse_composition(text)
    if any(e["mode"] == "unknown" for e in ent):
        raise MatrixError("the reference values cannot contain ? (use amounts, and at most one balance element)")
    fx = {e["element"]: e["value"] for e in ent if e["mode"] == "fixed"}
    for e in ent:
        if e["mode"] == "balance":
            fx[e["element"]] = max(1 - sum(fx.values()), 0.0)
    tot = sum(fx.values())
    if tot <= 0:
        raise MatrixError("the reference values add up to zero")
    fx = {k: v / tot for k, v in fx.items()}
    inside = sum(fx.get(e, 0.0) for e in elements)
    if inside <= 0:
        raise MatrixError("none of the reference elements is in this layer")
    return {e: 100 * fx.get(e, 0.0) / inside for e in elements}, 100 * (1 - inside)


def _ratio_sets(L):
    """[(reference element, [other elements])] for a layer: ratios are reported when it has at least two elements,
    relative to the first element whose amount is unknown (or the first element)."""
    els = [e["element"] for e in L["entries"]]
    if len(els) < 2:
        return []
    unk = [e["element"] for e in L["entries"] if e["mode"] == "unknown"]
    ref = unk[0] if unk else els[0]
    return [(ref, [e for e in els if e != ref])]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def core_msg(e):
    return str(e.args[0]) if e.args else str(e)


# ------------------------------------------------------------------- solving
def solve(mx, result, source=None):
    """Quantify with the matrix. result: core.FitResult. Returns a JSON-ready dict."""
    cfg = result.cfg
    exc = core.excitation(cfg)
    source = source or cfg.xs_source
    meas, notes = {}, []
    inst = core.instrument_groups(cfg)
    for name, (a, s) in result.areas.items():
        if name.startswith(("sum:", "Rayleigh", "Compton")) or name in inst:
            continue
        meas[name] = (a / result.live_time, s / result.live_time)
    model = Model(mx, list(meas), exc, source)
    fit_ls = bool(mx.get("fit_line_share"))
    model.wb0 = model.wb.copy()
    sc_E = {round(E, 6) for E, _, sc in core._beam_rows(cfg) if sc}
    sc_mask = np.array([round(float(E), 6) in sc_E for E in model.E0])
    # share of anode lines among the photons leaving the tube (before the beam filters), the number the tube panel uses
    raw = [(float(b["weight"]), bool(b.get("scatter"))) for b in cfg.beam if b.get("use", True) and float(b["weight"]) > 0]
    share0 = sum(w for w, sc in raw if sc) / sum(w for w, _ in raw) if raw else 0.0
    if fit_ls and not (sc_mask.any() and (~sc_mask).any()):
        raise MatrixError("the anode line share can only be fitted when the incident spectrum has both scatter-flagged "
                          "anode lines and a continuum (generate it with 'Generate from X-ray tube')")
    used = {n: meas[n] for n in model.groups}
    for n in meas:
        if n not in used:
            notes.append(f"{n} is not in the matrix and is ignored")
    if not used:
        raise MatrixError("none of the fitted element groups is an element of the matrix")

    # ---- unknowns
    layers, spec = model.layers, []     # spec entries: (kind, layer index, element or None)
    fixed_sum, balance, unknown = [], [], []
    for j, L in enumerate(layers):
        fx = sum(e["value"] for e in L["entries"] if e["mode"] == "fixed")
        un = [e["element"] for e in L["entries"] if e["mode"] == "unknown"]
        bal = [e["element"] for e in L["entries"] if e["mode"] == "balance"]
        if fx > 1 + 1e-9 and (un or bal):
            raise MatrixError(f"{L['name']}: the fixed amounts add up to {fx * 100:.1f} %, over 100 %")
        fixed_sum.append(fx); unknown.append(un); balance.append(bal[0] if bal else None)
        if not un and not bal and abs(fx - 1) > 0.01:
            notes.append(f"{L['name']}: the composition adds up to {fx * 100:.1f} % and was scaled to 100 %")
    where = {}
    for j, un in enumerate(unknown):
        for s in un:
            where.setdefault(s, []).append(j)
    for s, js in where.items():
        n_groups = sum(1 for g in used if model.groups[g][0] == s)
        if n_groups == 0:
            raise MatrixError(f"{s} is unknown in {layers[js[0]]['name']} but no {s} lines were fitted. Add it under Elements, "
                              "or fix its amount, or make it the balance")
        if n_groups < len(js):
            where_txt = " and ".join(layers[j]["name"] for j in js)
            raise MatrixError(f"{s} is unknown in {where_txt}, which needs {len(js)} independent {s} measurements but only "
                              f"{n_groups} line group{' was' if n_groups == 1 else 's were'} fitted. Tick 'Fit α, β and γ lines separately' "
                              "(Kα and Kβ are absorbed differently, which tells the depths apart), or fix its amount in one layer")
    k_fit = bool(mx.get("fit_k", True))
    try:
        k0 = float(mx.get("k") or 0)
    except (TypeError, ValueError):
        raise MatrixError("the instrument factor must be a number") from None
    if not k_fit and not k0 > 0:
        raise MatrixError("enter the instrument factor K, or tick 'fit K'")
    has_bal_unknown = any(balance[j] and unknown[j] for j in range(len(layers)))
    if k_fit and has_bal_unknown:
        raise MatrixError("a balance element together with a fitted K cannot be separated: absolute amounts need a known K. "
                          "Calibrate K on a standard (all amounts fixed, 'fit K' on), then enter it here and untick 'fit K'")

    # parameter layout: ("q", j, i) log-ratio, ("a", j, i) absolute fraction, ("k",), ("t", j)
    P = []
    for j in range(len(layers)):
        un = unknown[j]
        if balance[j]:
            P += [("a", j, s) for s in un]
        elif len(un) >= 2:
            P += [("q", j, s) for s in un[1:]]
    if k_fit:
        P.append(("k",))
    if fit_ls:
        P.append(("s",))
    P += [("t", j) for j, L in enumerate(layers) if L["fit_t"]]
    if len(P) > len(used):
        raise MatrixError(f"{len(P)} quantities are to be determined but only {len(used)} line groups were fitted; "
                          "fix some amounts or add lines")

    def unpack(x):
        pen = 0.0
        it = {p: v for p, v in zip(P, x)}
        fr = []
        for j, L in enumerate(layers):
            f = {e["element"]: e["value"] for e in L["entries"] if e["mode"] == "fixed"}
            rem = max(1 - fixed_sum[j], 0.0)
            un = unknown[j]
            if balance[j]:
                vals = [float(it[("a", j, s)]) for s in un]
                tot = sum(vals)
                if tot > rem:
                    pen += tot - rem
                    vals = [v * rem / tot for v in vals]
                    tot = rem
                f.update(dict(zip(un, vals)))
                f[balance[j]] = rem - tot
            elif un:
                q = [1.0] + [math.exp(float(it[("q", j, s)])) for s in un[1:]]
                tq = sum(q)
                f.update({s: rem * qi / tq for s, qi in zip(un, q)})
            else:
                tot = sum(f.values())
                f = {s: v / tot for s, v in f.items()}
            fr.append(f)
        mass = []
        for j, L in enumerate(layers):
            if L["t_um"] is None:
                mass.append(math.inf)
            else:
                t = math.exp(float(it[("t", j)])) if ("t", j) in it else L["t_um"]
                mass.append(L["rho"] * t * 1e-4)
        K = math.exp(float(it[("k",)])) if ("k",) in it else k0
        f_sc = math.exp(float(it[("s",)])) if ("s",) in it else 1.0
        # anode lines relative to the continuum; the tube beam stays normalised to 1 before the filters, so K keeps its meaning
        model.wb = model.wb0 * np.where(sc_mask, f_sc, 1.0) / ((1 - share0) + f_sc * share0)
        return fr, mass, K, pen

    names = list(used)
    m = np.array([used[n][0] for n in names]); sg = np.array([used[n][1] for n in names])
    wgt = 1 / np.sqrt(sg ** 2 + (0.01 * m) ** 2 + 1e-12)

    use_sec = [False]

    def resid(x):
        fr, mass, K, pen = unpack(x)
        f = model.predict(fr, mass, use_sec[0])
        r = np.array([(K * f[n] - used[n][0]) for n in names]) * wgt
        return np.append(r, pen * 1e3) if pen else r

    # ---- start values: fixed-point (Criss-Birks style) on the unknown amounts, then bounded least squares
    x = np.zeros(len(P)); lo = np.full(len(P), -np.inf); hi = np.full(len(P), np.inf)
    for i, p in enumerate(P):
        if p[0] == "a":
            lo[i], hi[i] = 0.0, max(1 - fixed_sum[p[1]], 1e-6); x[i] = hi[i] / (len(unknown[p[1]]) + 1)
        elif p[0] == "q":
            lo[i], hi[i] = -14, 14
        elif p[0] == "k":
            lo[i], hi[i] = -60, 60; x[i] = math.log(k0) if k0 > 0 else 0.0
        elif p[0] == "s":
            lo[i], hi[i] = math.log(1e-3), math.log(1e3)
        elif p[0] == "t":
            lo[i], hi[i] = math.log(1e-4), math.log(1e5); x[i] = math.log(max(layers[p[1]]["t_um"], 1e-3))
    idx = {p: i for i, p in enumerate(P)}
    for _ in range(12):
        fr, mass, K, _pen = unpack(x)
        f = model.predict(fr, mass)
        if ("k",) in idx:
            s_meas, s_pred = sum(used[n][0] for n in names), sum(f[n] for n in names)
            if s_pred > 0:
                x[idx[("k",)]] = min(max(math.log(s_meas / s_pred), -60), 60)
                K = math.exp(x[idx[("k",)]])
        new = {}
        for j in range(len(layers)):
            for s in unknown[j]:
                gs = [n for n in names if model.groups[n][0] == s]
                pm, pf = sum(used[n][0] for n in gs), sum(K * f[n] for n in gs)
                new[(j, s)] = max(fr[j].get(s, 1e-6), 1e-6) * (pm / pf if pf > 0 else 1.0)
        for j in range(len(layers)):
            un = unknown[j]
            if balance[j]:
                for s in un:
                    i = idx[("a", j, s)]; x[i] = min(max(new[(j, s)], lo[i]), hi[i])
            elif len(un) >= 2:
                for s in un[1:]:
                    x[idx[("q", j, s)]] = min(max(math.log(max(new[(j, s)], 1e-12) / max(new[(j, un[0])], 1e-12)), -14), 14)
    # secondary excitation changes the numbers slowly: solve with the primary model first, then refine with it on
    n_sec = 0
    if model.secondary and model.src:
        try:
            model.predict(*unpack(x)[:2], True)
            n_sec = 1
        except Exception as e:               # keep the primary result rather than fail the whole calculation
            notes.append(f"secondary excitation was skipped ({e})")
    if len(P):
        sol = least_squares(resid, x, bounds=(lo, hi), x_scale="jac")
        x = sol.x
        if n_sec:
            use_sec[0] = True
            sol = least_squares(resid, x, bounds=(lo, hi), x_scale="jac")
            x = sol.x
        J = sol.jac
    else:
        use_sec[0] = bool(n_sec)
        J = np.zeros((len(names), 0))
    r = resid(x)[:len(names)]
    dof = len(names) - len(P)
    chi2red = float(r @ r / dof) if dof > 0 else None
    if n_sec:
        base = model.predict(*unpack(x)[:2], False); full = model.predict(*unpack(x)[:2], True)
        worst = max(full, key=lambda g: full[g] / base[g] - 1 if base[g] > 0 else 0)
        if base[worst] > 0 and full[worst] / base[worst] - 1 > 0.005:
            notes.append(f"secondary excitation raises {worst} by {100 * (full[worst] / base[worst] - 1):.1f} %")

    # ---- uncertainties: parameter covariance mapped to the reported quantities
    def outputs(xx):
        fr, mass, K, _ = unpack(xx)
        o = [fr[j].get(e["element"], 0.0) * 100 for j, L in enumerate(layers) for e in L["entries"]]
        o += [(mass[j] / L["rho"] * 1e4) for j, L in enumerate(layers) if L["t_um"] is not None]
        for j, L in enumerate(layers):
            for ref, others in _ratio_sets(L):
                for e in others:
                    a, b = fr[j].get(e, 0.0), fr[j].get(ref, 0.0)
                    o.append(a / b if b > 0 else float("nan"))
                    o.append((a / attenuation.atomic_weight(e)) / (b / attenuation.atomic_weight(ref)) if b > 0 else float("nan"))
        o.append(K)
        if fit_ls:
            f_sc = math.exp(float(xx[idx[("s",)]]))
            o.append(f_sc * share0 / (f_sc * share0 + 1 - share0))
        return np.array(o)
    sig_out = None
    if len(P):
        try:
            cov = np.linalg.pinv(J.T @ J) * (max(chi2red, 1.0) if dof > 0 else 1.0)
            sv = np.linalg.svd(J, compute_uv=False)
            if sv.min() < 1e-6 * sv.max():
                notes.append("some of the quantities cannot be told apart by the fitted lines (they are almost fully correlated), "
                             "so their uncertainties are very large. Fit α, β and γ lines separately, add lines, or fix some amounts")
            Jo = np.zeros((len(outputs(x)), len(P)))
            for i in range(len(P)):
                d = np.zeros(len(P)); h = 1e-6 * max(1.0, abs(x[i])); d[i] = h
                Jo[:, i] = (outputs(np.clip(x + d, lo, hi)) - outputs(np.clip(x - d, lo, hi))) / (2 * h)
            sig_out = np.sqrt(np.maximum(np.diag(Jo @ cov @ Jo.T), 0))
        except np.linalg.LinAlgError:
            sig_out = None
    fr, mass, K, _ = unpack(x)
    pred = model.predict(fr, mass, use_sec[0])
    o, k = outputs(x), 0
    out_layers = []
    for j, L in enumerate(layers):
        comp = []
        for e in L["entries"]:
            comp.append({"element": e["element"], "mode": e["mode"], "wt_pct": float(o[k]),
                         "sigma": float(sig_out[k]) if sig_out is not None and e["mode"] != "fixed" else None})
            k += 1
        reference = None
        if L["reference"]:
            try:
                rv, outside = _reference(L["reference"], [c["element"] for c in comp])
            except MatrixError as e:
                raise MatrixError(f"{L['name']}, reference values: {e}") from None
            for c in comp:
                c["reference"] = rv[c["element"]]
                c["difference"] = c["wt_pct"] - rv[c["element"]]
            reference = {"outside_pct": outside}
        at = {c["element"]: c["wt_pct"] / attenuation.atomic_weight(c["element"]) for c in comp}
        tot = sum(at.values())
        for c in comp:
            c["at_pct"] = float(100 * at[c["element"]] / tot) if tot > 0 else None
        out_layers.append({"name": L["name"], "density": L["rho"], "composition": comp, "infinite": L["t_um"] is None,
                           "fit_thickness": L["fit_t"], "ratios": [], "reference": reference})
    for j, L in enumerate(layers):
        if L["t_um"] is not None:
            out_layers[j]["thickness_um"] = float(o[k])
            out_layers[j]["thickness_sigma"] = float(sig_out[k]) if sig_out is not None and L["fit_t"] else None
            k += 1
    # ratios of the elements in a layer to its first element that has an unknown amount (or its first element)
    for j, L in enumerate(layers):
        for ref, others in _ratio_sets(L):
            for e in others:
                out_layers[j]["ratios"].append({"element": e, "ref": ref, "mass": float(o[k]), "mass_sigma": float(sig_out[k]) if sig_out is not None else None,
                                                "atom": float(o[k + 1]), "atom_sigma": float(sig_out[k + 1]) if sig_out is not None else None})
                k += 2
    for i, p in enumerate(P):
        if p[0] == "a" and (x[i] <= lo[i] + 1e-9 or x[i] >= hi[i] - 1e-9):
            notes.append(f"{p[2]} in {layers[p[1]]['name']} reached its limit (0 or the whole remainder)")
        if p[0] == "t" and (x[i] <= lo[i] + 1e-6 or x[i] >= hi[i] - 1e-6):
            notes.append(f"the thickness of {layers[p[1]]['name']} reached its limit")
    groups = [{"name": n, "measured": used[n][0], "sigma": used[n][1], "predicted": K * pred[n],
               "ratio": (K * pred[n] / used[n][0]) if used[n][0] > 0 else None} for n in names]
    return _clean({"groups": groups, "layers": out_layers, "k": float(K), "k_fitted": k_fit,
                            "k_sigma": float(sig_out[-2 if fit_ls else -1]) if sig_out is not None and k_fit else None,
                            "line_share": ({"start": share0, "fitted": float(o[-1]), "sigma": float(sig_out[-1]) if sig_out is not None else None}
                                           if fit_ls else None),
                            "chi2red": chi2red, "n_free": len(P), "n_data": len(names), "notes": notes,
                            "alpha_in": mx.get("alpha_in", 45), "alpha_out": mx.get("alpha_out", 45)})
