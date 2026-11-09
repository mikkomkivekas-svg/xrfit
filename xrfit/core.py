# SPDX-License-Identifier: GPL-3.0-or-later
"""xrfit core: spectrum I/O, SNIP background, peak model, fitting, quantification.

Fit strategy (as in PyMca): nonlinear parameters (energy calibration, resolution,
tail, step, escape) are optimised by Levenberg-Marquardt/TRF; line-group areas and
polynomial background terms are solved linearly (variable projection, areas >= 0).
"""
from __future__ import annotations

import copy
import functools
import math
import json
import re
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.special import erfc, erfcx

from . import lines as _lines
from .lines import build_groups, SI_ESCAPE

FWHM_K = 2 * np.sqrt(2 * np.log(2))
SQ2PI = np.sqrt(2 * np.pi)
EPS_SI = 0.00385  # keV per e-h pair in Si


# ----------------------------------------------------------------- spectrum
@dataclass
class Spectrum:
    counts: np.ndarray
    name: str = "spectrum"
    live_time: float = 1.0
    calib: tuple | None = None  # (zero keV, gain keV/ch) if file provides it
    meta: dict = field(default_factory=dict)  # file metadata (name, times, filter settings ...)


def load_spectrum(path):
    """Text (1 col counts, or 2 col channel/energy + counts), Amptek .mca, or KETEK .vspc."""
    if path.lower().endswith(".vspc") or open(path, "rb").read(64).lstrip()[:1] == b"{" \
            or re.match(rb"\s*channel\s*,\s*raw\s*data", open(path, "rb").read(64), re.I):
        from .io_ketek import load_vspc
        return load_vspc(path)
    lines = open(path, errors="replace").read().splitlines()
    live, in_data, has_block = 1.0, False, any(l.startswith("<<DATA>>") for l in lines)
    rows = []
    for l in lines:
        s = l.strip()
        if s.startswith("<<DATA>>"):
            in_data = True
            continue
        if s.startswith("<<END>>"):
            in_data = False
            continue
        m = re.match(r"LIVE_TIME\s*-\s*([\d.eE+-]+)", s)
        if m:
            live = float(m.group(1))
            continue
        if has_block and not in_data:
            continue
        try:
            rows.append([float(t) for t in re.split(r"[,;\s]+", s) if t])
        except ValueError:
            continue
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError(f"no numeric data in {path}")
    calib = None
    if len(rows[0]) == 1:
        counts = np.array([r[0] for r in rows])
    else:
        x = np.array([r[0] for r in rows])
        counts = np.array([r[-1] for r in rows])
        if not np.allclose(x, np.round(x)) or x.max() < 100 and x.max() != len(x):
            g, z = np.polyfit(np.arange(len(x)), x, 1)  # x is energy in keV
            calib = (z, g)
    return Spectrum(counts, path.split("/")[-1], live, calib)


# --------------------------------------------------------------- background
def snip(y, width=24, smooth=5):
    """SNIP (LLS-transformed, decreasing window) background."""
    y = np.asarray(y, float)
    ys = np.convolve(y, np.ones(smooth) / smooth, "same") if smooth > 1 else y
    v = np.log(np.log(np.sqrt(np.maximum(ys, 0) + 1) + 1) + 1)
    n = len(v)
    for w in range(int(width), 0, -1):
        if 2 * w >= n:
            continue
        nv = v.copy()
        nv[w:n - w] = np.minimum(v[w:n - w], 0.5 * (v[:n - 2 * w] + v[2 * w:]))
        v = nv
    b = (np.exp(np.exp(v) - 1) - 1) ** 2 - 1
    return np.clip(b, 0, y)


# -------------------------------------------------------------------- model
def sigma_of(E, p):
    return np.sqrt((p["noise"] / FWHM_K) ** 2 + EPS_SI * p["fano"] * max(E, 0.0))


def _exp_tail(x, s, beta, ex, de):
    """Gaussian convolved with a one-sided exponential (unit area), stable on both sides of the peak."""
    b = x / (np.sqrt(2) * s) + s / (np.sqrt(2) * beta)
    pos = b >= 0  # erfcx form is stable for b>=0; plain erfc form for b<0
    return np.where(pos, ex * erfcx(np.where(pos, b, 0.0)),
                    np.exp(np.minimum(x / beta + s * s / (2 * beta * beta), 700))
                    * erfc(np.where(pos, 0.0, b))) / (2 * beta) * de


def profile(e, E0, s, de, tail_frac, beta, step_frac, tail2_frac=0.0, beta2=0.6):
    """Gaussian + short exponential tail (hypermet) + long exponential tail + step; area per unit = 1."""
    x = e - E0
    ex = np.exp(-0.5 * (x / s) ** 2)
    g = ex / (s * SQ2PI) * de
    y = (1 - tail_frac - tail2_frac) * g
    if tail_frac > 0:
        y = y + tail_frac * _exp_tail(x, s, beta, ex, de)
    if tail2_frac > 0:
        y = y + tail2_frac * _exp_tail(x, s, max(beta2, beta), ex, de)
    if step_frac > 0:
        y = y + step_frac * 0.5 * erfc(x / (np.sqrt(2) * s)) * de / (s * SQ2PI)
    return y


@functools.lru_cache(maxsize=None)
def si_escape_ratio(E):
    """Intensity of the Si K-alpha escape peak relative to the full-energy peak, for a thick Si detector.
    Reed & Ware: P = 1/2 w_K f_Ka (r-1)/r [1 - (mu_Ka/mu_E) ln(1 + mu_E/mu_Ka)] is the share of absorbed photons whose
    Si K-alpha photon leaves the crystal through the front face; the peak ratio is P / (1 - P). Zero below the Si K edge."""
    from . import attenuation, xdata
    x = xdata.xl()["Si"]
    if E <= x["edge_keV"]["K"] + 1e-3:
        return 0.0
    w, r = x["fluor_yield_xraylib"]["K"], x["jump"]["K"]
    f_ka = (x["rate"]["KL3"] + x["rate"]["KL2"]) / (x["rate"]["KL3"] + x["rate"]["KL2"] + x["rate"]["KM3"] + x["rate"]["KM2"])
    mu_k, mu_e = attenuation.mu("Si", SI_ESCAPE), attenuation.mu("Si", E)
    pr = 0.5 * w * f_ka * (r - 1) / r * (1 - (mu_k / mu_e) * math.log(1 + mu_e / mu_k))
    return pr / (1 - pr)


def peak(e, E, p, de):
    y = profile(e, E, sigma_of(E, p), de, p["tail_frac"], p["beta"], p["step_frac"], p.get("tail2_frac", 0.0), p.get("beta2", 0.6))
    r = p.get("esc_frac", 0.0) + p.get("esc_scale", 0.0) * si_escape_ratio(round(float(E), 4))
    if r > 0 and E > SI_ESCAPE + 0.1:
        Ee = E - SI_ESCAPE
        y = y + r * profile(e, Ee, sigma_of(Ee, p), de, 0, 0.1, 0)
    return y


PARAM_BOUNDS = {  # relative to start for zero/gain, absolute otherwise
    "noise": (0.01, 0.6), "fano": (0.05, 0.3), "tail_frac": (0.0, 0.5),
    "beta": (0.02, 0.6), "step_frac": (0.0, 0.2), "esc_frac": (0.0, 0.1), "esc_scale": (0.0, 5.0),
    "tail2_frac": (0.0, 0.3), "beta2": (0.15, 3.0), "compton_broad": (0.0, 1.5),
}
ELECTRON_MC2 = 510.999  # keV


@dataclass
class Config:
    elements: list = field(default_factory=list)
    instrument: list = field(default_factory=list)  # fit tokens for lines that do not come from the sample (tube anode L lines, air Ar)
    zero: float = 0.0
    gain: float = 0.01          # keV / channel
    noise: float = 0.10         # electronic noise FWHM, keV
    fano: float = 0.114
    tail_frac: float = 0.0
    beta: float = 0.10
    step_frac: float = 0.0
    esc_frac: float = 0.0       # extra constant escape ratio on top of the calculated one
    esc_scale: float = 1.0      # multiplier of the calculated Si escape ratio (si_escape_ratio)
    tail2_frac: float = 0.0     # long low-energy tail (shelf under strong soft lines) share and length
    beta2: float = 0.6
    free: list = field(default_factory=lambda: ["zero", "gain", "noise"])
    emin: float = 0.0
    emax: float = 1e9
    background: str = "snip"    # snip | poly | snip+poly | none
    snip_width: int = 24
    snip_smooth: int = 5
    poly_order: int = 2
    sum_peaks: bool = False
    free_ratios: bool = False
    e0: float = 20.0            # excitation energy (keV): sets L-line weights and theory quantification
    xs_source: str = "xraylib"  # xraylib | krause  (which tabulated cross-section column to use)
    # Incident spectrum, PyMca style: rows {"energy" keV, "weight" relative photons, "use", "scatter"}.
    # Empty = monochromatic beam at e0. Weights set the line intensities (cross sections averaged over the
    # beam); a row with "scatter" adds a Rayleigh peak at E and a Compton peak at E / (1 + E (1 - cos th) / mc2).
    beam: list = field(default_factory=list)
    matrix: dict = field(default_factory=dict)   # sample layers and geometry, see matrix.py (not used by fit())
    beam_filters: list = field(default_factory=list)  # filters between source and sample: {"material","density","thickness_um"}
    scatter_angle: float = 90.0  # scattering angle th (degrees)
    argon: str = "auto"         # Ar-K from the air between sample and detector: auto (when matrix detector.air_mm > 0) | on | off
    compton_broad: float = 0.20  # extra Compton peak width, keV FWHM at 20 keV (scales with energy; added in quadrature to the detector width)

    def to_json(self, path):
        json.dump(asdict(self), open(path, "w"), indent=2)

    @staticmethod
    def from_json(path):
        return Config(**json.load(open(path)))

    def params(self):
        return {k: getattr(self, k) for k in
                ("zero", "gain", "noise", "fano", "tail_frac", "beta", "step_frac", "esc_frac", "esc_scale", "tail2_frac", "beta2", "compton_broad")}


def _beam_rows(cfg):
    """Enabled beam rows [(energy, weight after the beam filters, scatter flag)]."""
    rows = [(float(b["energy"]), float(b["weight"]), bool(b.get("scatter"))) for b in cfg.beam
            if b.get("use", True) and float(b["weight"]) > 0]
    if rows and cfg.beam_filters:
        from . import matrix as _mx
        t = _mx.filter_transmission(cfg.beam_filters, [r[0] for r in rows])
        rows = [(E, w * float(ti), sc) for (E, w, sc), ti in zip(rows, t)]
        if max(float(ti) for ti in t) < 1e-12:
            raise ValueError("the beam filters block the whole incident beam")
    return rows


def excitation(cfg):
    """What the line tables see: a float (monochromatic at e0) or a tuple of (E, weight) pairs, after the beam filters."""
    if not cfg.beam:
        return float(cfg.e0)
    used = [(E, w) for E, w, _ in _beam_rows(cfg)]
    if not used:
        raise ValueError("no beam energy is enabled with a weight above zero")
    return tuple(sorted(used))


def scatter_energies(cfg):
    """[(E_rayleigh, E_compton, weight)] for every enabled beam row that has the scatter flag; weights sum to 1."""
    out, c = [], 1 - math.cos(math.radians(cfg.scatter_angle))
    for E, w, sc in _beam_rows(cfg):
        if sc and w > 0:
            out.append((E, E / (1 + E * c / ELECTRON_MC2), w))
    tot = sum(w for *_, w in out)
    return [(Er, Ec, w / tot) for Er, Ec, w in out]


def air_argon(cfg):
    """True when an Ar-K group for the argon in air is fitted automatically (the user has not added one)."""
    mode = getattr(cfg, "argon", "auto")
    if mode == "off" or any(str(t).split("-")[0] == "Ar" for t in cfg.elements):
        return False
    if mode == "on":
        return True
    det = (cfg.matrix or {}).get("detector") or {}
    try:
        return float(det.get("air_mm") or 0) > 0
    except (TypeError, ValueError):
        return False


def fit_elements(cfg):
    """Element tokens that are fitted: the user's list plus Ar-K when there is air in the path."""
    return list(cfg.elements) + (["Ar-K"] if air_argon(cfg) else [])


def design(cfg, p, ch):
    e = p["zero"] + p["gain"] * ch
    de = p["gain"]
    groups = build_groups(fit_elements(cfg), cfg.free_ratios, excitation(cfg), cfg.xs_source)
    cols, names, nonneg = [], [], []
    for name, lines in groups:
        cols.append(sum(w * peak(e, E, p, de) for E, w in lines))
        names.append(name); nonneg.append(True)
    cb = p.get("compton_broad", 0.0)             # Doppler broadening of the Compton peak, FWHM at 20 keV, grows in proportion to E
    pc = lambda E: dict(p, noise=math.hypot(p["noise"], cb * E / 20.0))
    # All scatter-flagged beam lines share ONE Rayleigh and ONE Compton amplitude, their relative sizes tied to the
    # beam weights (like the fluorescence lines of an element). One free amplitude per group keeps close lines such as
    # Rh Ka1/Ka2/Kb from competing with each other or with element peaks.
    sc = [t for t in scatter_energies(cfg) if e.min() - 1 < t[0] < e.max() + 1 or e.min() - 1 < t[1] < e.max() + 1]
    if sc:
        single = len(sc) == 1
        for label, idx in (("Rayleigh", 0), ("Compton", 1)):
            cols.append(sum(t[2] * peak(e, t[idx], p if idx == 0 else pc(t[1]), de) for t in sc))
            names.append(f"{label} {sc[0][0]:g}" if single else label); nonneg.append(True)
    if cfg.sum_peaks:
        main = [(n, max(l, key=lambda t: t[1])[0]) for n, l in groups]
        for i in range(len(main)):
            for j in range(i, len(main)):
                Es = main[i][1] + main[j][1]
                if e.min() < Es < e.max():
                    cols.append(peak(e, Es, p, de))
                    names.append(f"sum:{main[i][0]}+{main[j][0]}"); nonneg.append(True)
    if "poly" in cfg.background and cfg.poly_order >= 0:
        x = (e - e.min()) / max(e.max() - e.min(), 1e-9)
        for k in range(cfg.poly_order + 1):
            cols.append(x ** k); names.append(f"bg:poly{k}"); nonneg.append(False)
    return np.array(cols).T, names, np.array(nonneg)


@dataclass
class FitResult:
    params: dict
    param_err: dict
    areas: dict            # name -> (counts, sigma)
    chi2red: float
    channels: np.ndarray
    energy: np.ndarray
    data: np.ndarray
    background: np.ndarray
    fit: np.ndarray
    components: dict
    cfg: Config
    live_time: float = 1.0

    def report(self):
        o = [f"chi2_red = {self.chi2red:.3f}   region = {self.energy[0]:.2f}-{self.energy[-1]:.2f} keV",
             "Parameters:"]
        for k, v in self.params.items():
            err = self.param_err.get(k)
            o.append(f"  {k:10s} {v:12.6g}" + (f" +- {err:.2g}" if err is not None else "  (fixed)"))
        o.append("Net areas (counts):")
        for k, (a, s) in self.areas.items():
            o.append(f"  {k:16s} {a:12.1f} +- {s:8.1f}   ({a / self.live_time:.2f} cps)")
        return "\n".join(o)

    def save_csv(self, path):
        res = self.data - self.fit
        arr = np.column_stack([self.channels, self.energy, self.data, self.background, self.fit, res])
        np.savetxt(path, arr, delimiter=",", header="channel,energy_keV,data,background,fit,residual",
                   comments="", fmt="%.6g")


def _region(cfg, n):
    i0 = max(0, int((cfg.emin - cfg.zero) / cfg.gain))
    i1 = n if cfg.emax >= 1e8 else min(n, int((cfg.emax - cfg.zero) / cfg.gain) + 1)
    return i0, i1


def fit(spec, cfg0, outer=2):
    cfg = copy.deepcopy(cfg0)
    if not cfg.elements:
        raise ValueError("no elements defined")
    y_all = np.asarray(spec.counts, float)
    i0, i1 = _region(cfg, len(y_all))
    ch, y = np.arange(i0, i1, dtype=float), y_all[i0:i1]
    bg = snip(y_all, cfg.snip_width, cfg.snip_smooth)[i0:i1] if "snip" in cfg.background else np.zeros_like(y)
    yn = y - bg
    base, free = cfg.params(), [k for k in cfg.free if k in cfg.params()]
    lo, hi = [], []
    for k in free:
        if k == "zero": lo.append(base[k] - 0.15); hi.append(base[k] + 0.15)
        elif k == "gain": lo.append(base[k] * 0.95); hi.append(base[k] * 1.05)
        else: lo.append(PARAM_BOUNDS[k][0]); hi.append(PARAM_BOUNDS[k][1])
    x0 = np.clip([base[k] for k in free], lo, hi) if free else []
    wts = 1 / np.sqrt(np.maximum(y, 1.0))

    def solve(x):
        p = dict(base); p.update(zip(free, x))
        A, names, nn = design(cfg, p, ch)
        Aw, yw = A * wts[:, None], yn * wts
        sol = lsq_linear(Aw, yw, bounds=(np.where(nn, 0, -np.inf), np.inf), method="bvls")
        return p, A, Aw, yw, names, nn, sol.x

    x = np.array(x0, float)
    res = None
    for _ in range(outer):
        if free:
            res = least_squares(lambda xx: (lambda r: r[3] - r[2] @ r[6])(solve(xx)),
                                x, bounds=(lo, hi), x_scale="jac")
            x = res.x
        p, A, Aw, yw, names, nn, c = solve(x)
        model = A @ c + bg
        wts = 1 / np.sqrt(np.maximum(model, 1.0))  # Poisson reweighting
    p, A, Aw, yw, names, nn, c = solve(x)
    r = yw - Aw @ c
    act = (c > 1e-9 * max(c.max(), 1)) | ~nn
    dof = max(len(y) - act.sum() - len(free), 1)
    chi2red = float(r @ r / dof)
    cov = np.linalg.pinv(Aw[:, act].T @ Aw[:, act]) * chi2red
    sig = np.zeros(len(c)); sig[act] = np.sqrt(np.maximum(np.diag(cov), 0))
    perr = {}
    if free and res is not None:
        pc = np.linalg.pinv(res.jac.T @ res.jac) * chi2red
        perr = dict(zip(free, np.sqrt(np.maximum(np.diag(pc), 0))))
    comps = {n: A[:, i] * c[i] for i, n in enumerate(names)}
    fitted = A @ c + bg
    areas = {n: (float(c[i]), float(sig[i])) for i, n in enumerate(names) if not n.startswith("bg:")}
    bg_tot = bg + sum((v for k, v in comps.items() if k.startswith("bg:")), np.zeros_like(y))
    return FitResult(p, perr, areas, chi2red, ch, p["zero"] + p["gain"] * ch, y, bg_tot, fitted,
                     comps, cfg, spec.live_time)


def fit_batch(spectra, cfg):
    """Sequential batch fit; refined calibration/resolution seeds the next spectrum."""
    out, cur = [], copy.deepcopy(cfg)
    for s in spectra:
        r = fit(s, cur)
        out.append(r)
        for k in cur.free:
            setattr(cur, k, r.params[k])
    return out


# ------------------------------------------------------------ quantification
def sensitivities(result, standard):
    """standard: {'Fe-K': mass fraction or mass thickness}. Returns counts per unit conc per second."""
    return {k: result.areas[k][0] / result.live_time / v for k, v in standard.items() if k in result.areas}


def quantify(result, sens, normalize=False):
    conc = {k: result.areas[k][0] / result.live_time / s for k, s in sens.items() if k in result.areas}
    if normalize and conc:
        t = sum(conc.values()); conc = {k: v / t for k, v in conc.items()}
    return conc


def instrument_groups(cfg):
    """Names of the fitted line groups that come from the instrument (tube, air), not from the sample."""
    out = set()
    for tok in list(cfg.instrument) + (["Ar-K"] if air_argon(cfg) else []):
        try:
            out.update(n for n, _ in _lines.build_groups([tok], cfg.free_ratios, excitation(cfg), cfg.xs_source))
        except (KeyError, ValueError):
            pass
    return out


def quantify_theory(result, e0=None, source=None, normalize=True):
    """Relative atomic amounts from fundamental production cross sections (thin sample, mono-
    chromatic excitation at e0): n_i ~ (net counts / live time) / sigma_i(e0).
    Ignores detector efficiency, filters, matrix absorption and secondary excitation, so use it
    for relative amounts of similar-energy lines or thin samples. Returns {group: (value, note)}."""
    e0 = excitation(result.cfg) if e0 is None else e0
    source = result.cfg.xs_source if source is None else source
    out = {}
    inst = instrument_groups(result.cfg)
    for name, (area, _) in result.areas.items():
        if name.startswith(("sum:", "Rayleigh", "Compton")) or name in inst:
            continue
        sig, note = _lines.group_sigma(name, e0, source)
        if sig > 0:
            out[name] = [area / result.live_time / sig, note]
        else:
            out[name] = [float("nan"), f"not excited by the {_lines.beam_label(e0)}"]
    if normalize:
        tot = sum(v[0] for v in out.values() if v[0] == v[0])
        for v in out.values():
            if v[0] == v[0] and tot > 0: v[0] /= tot
    return {k: tuple(v) for k, v in out.items()}
