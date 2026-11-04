# SPDX-License-Identifier: GPL-3.0-or-later
"""JaBS-style command interpreter for xrfit.

    xrfit                 interactive prompt
    xrfit script.xrf      run a script (one command per line, '#' comments)
    xrfit -c "load a.mca; add Fe Cu; fit; show"
"""
import cmd
import glob
import shlex
import sys

from . import core, io_ketek, lines, matrix, tube, xdata

CFG_KEYS = ("zero gain noise fano tail_frac beta step_frac esc_frac emin emax snip_width "
            "snip_smooth poly_order e0 scatter_angle compton_broad").split()


class Shell(cmd.Cmd):
    prompt = "xrfit> "
    intro = "xrfit: PyMca-style XRF fitting. 'help' lists commands."

    def __init__(self):
        super().__init__()
        self.cfg, self.spec, self.result, self.sens = core.Config(), None, None, {}

    # -- helpers
    def _need(self, what):
        if what == "spec" and self.spec is None: print("load a spectrum first"); return False
        if what == "res" and self.result is None: print("run 'fit' first"); return False
        return True

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except (ValueError, KeyError, OSError, IndexError) as e:
            print(f"error: {e}")

    def emptyline(self):
        pass

    # -- commands
    def do_load(self, a):
        "load <file>  - read spectrum (text, csv, Amptek .mca); uses embedded calibration if present"
        self.spec = core.load_spectrum(shlex.split(a)[0])
        if self.spec.calib:
            self.cfg.zero, self.cfg.gain = self.spec.calib
        print(f"{self.spec.name}: {len(self.spec.counts)} channels, {self.spec.counts.sum():.0f} counts")
        if self.spec.meta:
            print(io_ketek.describe(self.spec.meta, name=self.spec.name))

    def do_info(self, a):
        "info [all]  - metadata of loaded spectrum: name, measurement time, slow filter parameters, ..."
        if self._need("spec"):
            print(f"{self.spec.name}: live time {self.spec.live_time:g} s, {len(self.spec.counts)} channels")
            print(io_ketek.describe(self.spec.meta, show_all="all" in a, name=self.spec.name))

    def do_set(self, a):
        "set <name> <value>  - zero gain noise fano tail_frac beta step_frac esc_frac emin emax\n"\
        "                      snip_width snip_smooth poly_order e0 (single excitation keV; clears 'beam') |\n"\
        "                      scatter_angle (deg) compton_broad (keV) |\n"\
        "                      xs_source xraylib|krause | background snip|poly|snip+poly|none\n"\
        "                      sum_peaks on|off | free_ratios on|off"
        k, v = shlex.split(a)[:2]
        if k in CFG_KEYS:
            if k == "e0":
                self.cfg.beam = []  # a single e0 means a monochromatic beam
            setattr(self.cfg, k, int(v) if k in ("snip_width", "snip_smooth", "poly_order") else float(v))
        elif k == "background":
            self.cfg.background = v
        elif k == "xs_source":
            if v not in ("xraylib", "krause"): raise ValueError("xs_source is xraylib or krause")
            self.cfg.xs_source = v
        elif k in ("sum_peaks", "free_ratios"):
            setattr(self.cfg, k, v.lower() in ("1", "on", "true", "yes"))
        else:
            raise KeyError(f"unknown setting '{k}'")

    def do_beam(self, a):
        "beam                      show the incident spectrum (energy, weight, scatter)\n"\
        "beam add E [W] [scatter]  add an energy (keV), relative weight (default 1), optional scatter flag\n"\
        "beam load <file>          read energy [weight [scatter]] lines from a text file (replaces the table)\n"\
        "beam tube <anode> <kV> [steps] [line%]  simplified tube spectrum (Kramers + anode K lines)\n"\
        "beam clear                monochromatic beam at e0 again\n"\
        "A row with 'scatter' adds Rayleigh and Compton peaks (see 'set scatter_angle', 'set compton_broad')."
        t = shlex.split(a)
        try:
            if not t:
                pass
            elif t[0] == "clear":
                self.cfg.beam = []
            elif t[0] == "add":
                self.cfg.beam.append({"energy": float(t[1]), "weight": float(t[2]) if len(t) > 2 and t[2] != "scatter" else 1.0,
                                      "use": True, "scatter": "scatter" in t[3:] or (len(t) > 2 and t[2] == "scatter")})
            elif t[0] == "load":
                self.cfg.beam = tube.parse_beam_text(open(t[1]).read())
            elif t[0] == "tube":
                self.cfg.beam = tube.tube_beam(t[1], float(t[2]), int(t[3]) if len(t) > 3 else 24, 1.5,
                                               float(t[4]) / 100 if len(t) > 4 else 0.25)
            else:
                return print("usage: beam [add E [W] [scatter] | load FILE | tube ANODE KV [steps] [line%] | clear]")
            exc = core.excitation(self.cfg)
        except (IndexError, ValueError, KeyError, OSError) as e:
            return print(f"beam: {e.args[0] if e.args else e}")
        if not self.cfg.beam:
            return print(f"monochromatic beam at e0 = {self.cfg.e0:g} keV")
        tot = sum(w for _, w in exc)
        for b in self.cfg.beam:
            sh = b["weight"] / tot * 100 if b.get("use", True) and b["weight"] > 0 else 0
            print(f"  {b['energy']:9.4f} keV  weight {b['weight']:<10.4g} {sh:5.1f} %" + ("  scatter" if b.get("scatter") else "")
                  + ("" if b.get("use", True) else "  (off)"))
        print(f"  scattering angle {self.cfg.scatter_angle:g} deg")

    def do_filter(self, a):
        "filter                          list the beam and detector filters\n"\
        "filter beam|detector add MATERIAL THICKNESS_UM [DENSITY]   element, formula or preset (Kapton, Mylar...)\n"\
        "filter beam|detector clear      remove all filters of that kind\n"\
        "filter T E [E ...]              transmission of the beam filters at these energies (keV)"
        t = shlex.split(a)
        mx = self.cfg.matrix
        det = mx.setdefault("detector", {"si_mm": 0.5, "be_um": 0, "air_mm": 0}) if mx else None
        try:
            if t[:1] == ["T"]:
                es = [float(x) for x in t[1:]]
                tr = matrix.filter_transmission(self.cfg.beam_filters, es)
                return print("  ".join(f"{e:g} keV: {x * 100:.3g} %" for e, x in zip(es, tr)))
            if len(t) >= 2 and t[0] in ("beam", "detector"):
                if t[0] == "detector" and det is None:
                    return print("filter: describe the matrix first (matrix layers 1)")
                lst = self.cfg.beam_filters if t[0] == "beam" else det.setdefault("filters", [])
                if t[1] == "clear":
                    lst.clear()
                elif t[1] == "add" and len(t) >= 4:
                    f = {"material": t[2], "thickness_um": float(t[3]), "density": float(t[4]) if len(t) > 4 else ""}
                    matrix.parse_filters(lst + [f])
                    lst.append(f)
                else:
                    return print("usage: filter beam|detector add MATERIAL THICKNESS_UM [DENSITY] | clear")
            elif t:
                return print("usage: filter [beam|detector add ... | clear | T E...]")
        except (matrix.MatrixError, ValueError) as e:
            return print(f"filter: {e.args[0] if e.args else e}")
        for kind, lst in (("beam", self.cfg.beam_filters), ("detector", (det or {}).get("filters", []))):
            print(f"{kind} filters: " + ("none" if not lst else "; ".join(
                f"{f['material']} {f['thickness_um']} um" + (f" @ {f['density']} g/cm3" if f.get("density") not in ("", None) else "") for f in lst)))

    def do_matrix(self, a):
        "matrix                          show the sample layers\n"\
        "matrix layers <n>               number of layers (layer 1 is the top, the last is infinitely thick)\n"\
        "matrix layer <i> comp \"Fe 70; Cr ?; O balance\"   composition: wt %, ? = find from lines, balance = rest; or a formula\n"\
        "matrix layer <i> density <g/cm3> | thickness <um> [fit] | name <text>\n"\
        "matrix angles <beam deg> <detector deg>   angles to the surface (default 45 45)\n"\
        "matrix k <value|fit>            instrument factor (flux x solid angle), or fit it\n"\
        "matrix secondary on|off         secondary excitation (default on)\n"\
        "matrix detector si=0.5 be=0 air=0   Si thickness (mm), Be window (um), air path (mm)\n"\
        "then: fit, quantify matrix"
        t = shlex.split(a)
        mx = self.cfg.matrix
        if not mx:
            mx.update(layers=[{"name": "", "composition": "", "density": "", "thickness_um": "", "fit_thickness": False}],
                      alpha_in=45, alpha_out=45, k="", fit_k=True, detector={"si_mm": 0.5, "be_um": 0, "air_mm": 0})
        try:
            if t[:1] == ["layers"]:
                n = int(t[1])
                if not 1 <= n <= 12: raise ValueError("use 1 to 12 layers")
                while len(mx["layers"]) < n:
                    mx["layers"].insert(0, {"name": "", "composition": "", "density": "", "thickness_um": "", "fit_thickness": False})
                del mx["layers"][:max(len(mx["layers"]) - n, 0)]
            elif t[:1] == ["layer"]:
                L = mx["layers"][int(t[1]) - 1]
                key, val = t[2], " ".join(t[3:])
                if key == "comp":
                    matrix.parse_composition(val); L["composition"] = val
                elif key == "density": L["density"] = float(t[3])
                elif key == "thickness":
                    L["thickness_um"] = float(t[3]); L["fit_thickness"] = "fit" in t[4:]
                elif key == "name": L["name"] = val
                else: raise ValueError("layer <i> comp|density|thickness|name ...")
            elif t[:1] == ["angles"]:
                mx["alpha_in"], mx["alpha_out"] = float(t[1]), float(t[2])
            elif t[:1] == ["k"]:
                if t[1] == "fit": mx["fit_k"] = True
                else: mx["k"], mx["fit_k"] = float(t[1]), False
            elif t[:1] == ["secondary"]:
                mx["secondary"] = t[1].lower() in ("1", "on", "true", "yes")
            elif t[:1] == ["detector"]:
                for kv in t[1:]:
                    k, v = kv.split("=")
                    mx["detector"][{"si": "si_mm", "be": "be_um", "air": "air_mm"}[k]] = float(v)
            elif t:
                return print("usage: matrix [layers N | layer I comp|density|thickness|name ... | angles A B | k V|fit | detector si= be= air=]")
        except (IndexError, ValueError, KeyError, matrix.MatrixError) as e:
            return print(f"matrix: {e.args[0] if e.args else e}")
        n = len(mx["layers"])
        for i, L in enumerate(mx["layers"], 1):
            th = "infinitely thick" if i == n else f"{L['thickness_um'] or '?'} um" + (" (fit)" if L["fit_thickness"] else "")
            print(f"  layer {i}{' (top)' if i == 1 and n > 1 else ''}: {L['name'] or '-'}  {th}  density {L['density'] or 'element default'}")
            print(f"      {L['composition'].replace(chr(10), '; ') or '(no composition yet)'}")
        print(f"  angles {mx['alpha_in']} / {mx['alpha_out']} deg, K {'fit' if mx['fit_k'] else mx['k']}, detector {mx['detector']}")

    def do_free(self, a):
        "free <params...>  - nonlinear parameters to fit (e.g. free zero gain noise tail_frac); 'free none'"
        self.cfg.free = [] if a.strip() == "none" else a.split()

    def do_add(self, a):
        "add <El|El-K|El-L|El-M|El-Ka1 ...> ...  - add fit elements, shell families or single lines"
        for t in a.split():
            g = lines.build_groups([t], e0=self.cfg.e0, source=self.cfg.xs_source)
            if t not in self.cfg.elements: self.cfg.elements.append(t)
            print(f"  {t} -> " + ", ".join(f"{n} ({len(l)} line{'s' if len(l) > 1 else ''})" for n, l in g))

    def do_remove(self, a):
        "remove <El> ...  - remove fit elements ('remove all')"
        self.cfg.elements = [] if a.strip() == "all" else [e for e in self.cfg.elements if e not in a.split()]

    def do_lines(self, a):
        "lines <El>  - list all lines of an element: energies (Table 1-2), rates and cross sections (xraylib)"
        for sym in a.split():
            print(lines.describe(sym, self.cfg.e0, self.cfg.xs_source))

    def do_xsec(self, a):
        "xsec <El-Line> <E keV> [E ...]  - production cross section in barns/atom, e.g. 'xsec Fe-Ka1 10 20'\n"\
        "                                 (families Fe-K / Pb-L give the sum over the group)"
        t = a.split()
        tok, es = t[0], [float(x) for x in t[1:]]
        sym, fam, line = lines.parse_token(tok, self.cfg.e0)
        for E in es:
            if line:
                s, note = xdata.xs_line(sym, line, E, self.cfg.xs_source)
            else:
                s, notes = 0.0, set()
                for n, _, _ in lines.family_lines(sym, fam, self.cfg.e0, self.cfg.xs_source):
                    v, nt = xdata.xs_line(sym, n, E, self.cfg.xs_source); s += v
                    if nt: notes.add(nt)
                note = "; ".join(sorted(notes))
            print(f"  {tok} @ {E:g} keV: {s:.4g} b/atom" + (f"   [{note}]" if note else ""))

    def do_elements(self, a):
        "elements  - elements available and what data each has"
        for s in xdata.symbols():
            x = xdata.xl().get(s)
            print(f"  {s:2s} Z={(xdata.t12().get(s) or x)['Z']:3d}  Table1-2:{'y' if s in xdata.t12() else '-'}  "
                  f"xraylib:{'y' if x else '-'}  K-xs:{'y' if x and 'xs_K' in x else '-'}  L-xs:{'y' if x and 'xs_L' in x else '-'}")

    def do_fit(self, a):
        "fit  - fit current spectrum"
        if self._need("spec"):
            self.result = core.fit(self.spec, self.cfg)
            print(self.result.report())

    def do_show(self, a):
        "show [settings|fit]"
        if a.strip() == "fit" and self._need("res"):
            print(self.result.report())
        else:
            for k, v in vars(self.cfg).items(): print(f"  {k:12s} {v}")

    def do_plot(self, a):
        "plot [file.png]  - plot data/fit/background (shows window if no file)"
        if not self._need("res"): return
        import matplotlib
        if a.strip(): matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = self.result
        fig, (ax, rx) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]})
        ax.semilogy(r.energy, r.data, ".", ms=3, label="data")
        ax.semilogy(r.energy, r.fit, "r", label="fit")
        ax.semilogy(r.energy, r.background, "g--", label="background")
        for n, c in r.components.items():
            if not n.startswith("bg:") and c.max() > 0: ax.semilogy(r.energy, c + r.background * 0 + 1e-3, lw=.7)
        ax.set_ylim(max(0.5, r.data[r.data > 0].min() * .5) if (r.data > 0).any() else .5, None)
        ax.legend(loc="upper right"); ax.set_ylabel("counts")
        if self.spec is not None and self.spec.meta:
            box = io_ketek.summary(self.spec.meta, self.spec.name)
            if box:
                ax.text(0.01, 0.98, box, transform=ax.transAxes, fontsize=7, va="top", family="monospace",
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        rx.plot(r.energy, (r.data - r.fit) / (r.fit.clip(1)) ** .5, lw=.6); rx.set_xlabel("keV"); rx.set_ylabel("resid / sigma")
        if a.strip(): fig.savefig(a.strip(), dpi=130); print("saved", a.strip())
        else: plt.show()

    def do_save(self, a):
        "save <file.csv|file.json>  - fit curves (.csv) or settings (.json)"
        f = a.strip()
        if f.endswith(".json"): self.cfg.to_json(f)
        elif self._need("res"): self.result.save_csv(f)

    def do_config(self, a):
        "config <file.json>  - load settings"
        self.cfg = core.Config.from_json(a.strip())

    def do_calibrate(self, a):
        "calibrate <El-K>=<conc> ...  - derive sensitivities from fitted standard"
        if self._need("res"):
            std = {k: float(v) for k, v in (t.split("=") for t in a.split())}
            self.sens = core.sensitivities(self.result, std); print(self.sens)

    def do_quantify(self, a):
        "quantify [normalize]  - concentrations using 'calibrate' sensitivities\n"\
        "quantify theory [raw]  - relative atomic amounts from xraylib cross sections at e0 (normalised unless 'raw')\n"\
        "quantify matrix       - composition from the layer model (see 'matrix')"
        if not self._need("res"): return
        if "matrix" in a:
            try:
                r = matrix.solve(self.cfg.matrix, self.result)
            except (matrix.MatrixError, KeyError) as e:
                return print(f"matrix: {e.args[0] if e.args else e}")
            print(f"  K = {r['k']:.5g} ({'fitted' if r['k_fitted'] else 'given'}), {r['n_free']} quantities from {r['n_data']} line groups"
                  + (f", reduced chi2 {r['chi2red']:.2f}" if r["chi2red"] is not None else ""))
            for L in r["layers"]:
                th = "infinitely thick" if L["infinite"] else f"{L['thickness_um']:.4g} um" + (" (fitted)" if L["fit_thickness"] else "")
                print(f"  {L['name']}: {th}, {L['density']:g} g/cm3")
                for c in L["composition"]:
                    sg = f" +- {c['sigma']:.3g}" if c["sigma"] is not None else ""
                    print(f"      {c['element']:3s} {c['wt_pct']:9.4f} wt %{sg}   {c['at_pct']:8.4f} at %   [{c['mode']}]")
                for q in L["ratios"]:
                    es = lambda v, e: f"{v:.5g}" + (f" +- {e:.2g}" if e is not None else "")
                    print(f"      {q['element']}/{q['ref']}: {es(q['mass'], q['mass_sigma'])} by mass, {es(q['atom'], q['atom_sigma'])} by atoms")
            print("  group        measured cps  calculated cps  calc/meas")
            for g in r["groups"]:
                print(f"  {g['name']:10s} {g['measured']:13.5g} {g['predicted']:15.5g} {g['ratio'] if g['ratio'] is None else round(g['ratio'], 4)!s:>10}")
            for n in r["notes"]: print("  note:", n)
        elif "theory" in a:
            res = core.quantify_theory(self.result, normalize="raw" not in a)
            print(f"  e0 = {self.cfg.e0:g} keV, cross sections: {self.cfg.xs_source}; thin sample, no detector efficiency/matrix correction")
            for k, (v, note) in res.items(): print(f"  {k:10s} {v:.5g}" + (f"   [{note}]" if note else ""))
        else:
            for k, v in core.quantify(self.result, self.sens, "norm" in a).items(): print(f"  {k:10s} {v:.5g}")

    def do_batch(self, a):
        "batch <glob...> [> out.csv]  - fit many spectra, seeding each with the previous result"
        toks = shlex.split(a); out = None
        if ">" in toks: i = toks.index(">"); out = toks[i + 1]; toks = toks[:i]
        files = sorted(f for t in toks for f in glob.glob(t))
        res = core.fit_batch([core.load_spectrum(f) for f in files], self.cfg)
        names = sorted({n for r in res for n in r.areas})
        rows = ["file," + ",".join(f"{n},{n}_err" for n in names)]
        for f, r in zip(files, res):
            rows.append(f + "," + ",".join(f"{r.areas.get(n, (0, 0))[0]:.2f},{r.areas.get(n, (0, 0))[1]:.2f}" for n in names))
        txt = "\n".join(rows)
        if out: open(out, "w").write(txt + "\n"); print(f"{len(files)} spectra -> {out}")
        else: print(txt)
        if res: self.result = res[-1]

    def do_script(self, a):
        "script <file>  - run commands from file"
        for line in open(a.strip()):
            line = line.split("#")[0].strip()
            if line:
                print(f"{self.prompt}{line}")
                if self.onecmd(line): break

    def do_exit(self, a):
        "exit"
        return True
    do_quit = do_EOF = do_exit


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sh = Shell()
    if argv and argv[0] == "-c":
        for c in argv[1].split(";"): sh.onecmd(c.strip())
    elif argv:
        sh.do_script(argv[0])
    else:
        sh.cmdloop()


if __name__ == "__main__":
    main()
