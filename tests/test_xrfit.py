# SPDX-License-Identifier: GPL-3.0-or-later
import math
import unittest

import numpy as np

from xrfit import core, lines, xdata


class TestTables(unittest.TestCase):
    def test_table_1_2_values(self):  # read directly from the PDF
        t = xdata.t12()
        self.assertEqual(t["Fe"]["Ka1"], 6403.84)
        self.assertEqual(t["Cu"]["Kb1"], 8905.29)
        self.assertEqual(t["Pb"]["Lg1"], 14764.4)
        self.assertEqual(t["U"]["Ma1"], 3170.8)
        self.assertEqual((t["Ne"]["Ka1"], t["Ne"]["Ka2"], t["Ne"]["Kb1"]), (848.6, 848.6, None))
        self.assertIsNone(t["Xe"]["Lb1"])
        self.assertEqual(len(t), 93)

    def test_xraylib_page_values(self):
        fe = xdata.xl()["Fe"]
        self.assertEqual(fe["edge_keV"]["K"], 7.112)
        self.assertEqual(fe["rate"]["KL3"], 0.581)
        self.assertEqual(fe["fluor_yield_krause"]["K"], 0.340)
        self.assertAlmostEqual(xdata.xs_line("Fe", "Ka1", 8.0)[0], 4810)
        self.assertAlmostEqual(xdata.xs_line("Fe", "Ka1", 8.0, "krause")[0], 4890)
        self.assertAlmostEqual(xdata.xs_line("Fe", "Kb1", 10.0)[0], 373, delta=1e-6)

    def test_xs_physics(self):
        self.assertEqual(xdata.xs_line("Fe", "Ka1", 7.0)[0], 0.0)           # below K edge
        a = xdata.xs_line("Fe", "Ka1", 10)[0]
        b = xdata.xs_line("Fe", "Ka2", 10)[0]
        self.assertAlmostEqual(b / a, 0.297 / 0.581, delta=0.02)             # Ka2/Ka1 = rate ratio
        self.assertAlmostEqual(xdata.xs_line("Pb", "La1", 20)[0], 3450)       # regular grid point
        # monotonic between grid points above the edge
        es = np.linspace(8, 200, 60)
        v = [xdata.xs_line("Fe", "Ka1", e)[0] for e in es]
        self.assertTrue(all(x > y for x, y in zip(v, v[1:])))

    def test_mo_duplicate_edge_row_ignored(self):
        v = xdata.xs_line("Mo", "Ka1", 40.0)[0]
        self.assertAlmostEqual(v, 711, delta=1)
        self.assertLess(xdata.xs_line("Mo", "Ka1", 25.0)[0], 5000)


class TestGroups(unittest.TestCase):
    def test_weights(self):
        for tok in ("Fe-K", "Pb-L", "Sn-L", "Au-L", "Cu-K"):
            (name, ls), = lines.build_groups([tok])
            self.assertAlmostEqual(sum(w for _, w in ls), 1.0, places=9, msg=name)
        (name, ls), = lines.build_groups(["Fe-K"])
        self.assertAlmostEqual(ls[0][1], 0.581, places=3)
        self.assertAlmostEqual(ls[0][0], 6.40384)

    def test_default_family(self):
        self.assertEqual(lines.default_family("Fe", 20), "K")
        self.assertEqual(lines.default_family("Sn", 20), "L")   # K edge 29.2 keV not excited
        self.assertEqual(lines.default_family("Pb", 20), "L")
        self.assertEqual(lines.default_family("Pb", 100), "K")

    def test_single_lines_and_errors(self):
        self.assertEqual(lines.build_groups(["Pb-La2"])[0][1], [(10.4495, 1.0)])
        self.assertEqual(lines.build_groups(["U-La1"])[0][0], "U-La1")
        with self.assertRaises(KeyError): lines.build_groups(["U-L"])
        with self.assertRaises(KeyError): lines.build_groups(["Xx"])

    def test_free_ratios_split(self):
        names = [n for n, _ in lines.build_groups(["Pb-L", "Fe-K"], free_ratios=True)]
        self.assertEqual(names, ["Pb-La", "Pb-Lb", "Pb-Lg", "Fe-Ka", "Fe-Kb"])


class TestFitAndQuant(unittest.TestCase):
    def test_recovery_and_theory_quantification(self):
        rng = np.random.default_rng(7)
        cfg = core.Config(elements=["Fe", "Cu", "Pb-L"], zero=0.02, gain=0.01, noise=0.12, e0=20.0)
        ch = np.arange(2048.0)
        A, names, _ = core.design(cfg, cfg.params(), ch)
        live = 100.0
        n_true = {"Fe-K": 2.0, "Cu-K": 1.0, "Pb-L": 0.5}                       # relative atoms
        amp = np.array([n_true[k] * lines.group_sigma(k, 20.0)[0] * live * 2.0 for k in names])
        e = cfg.zero + cfg.gain * ch
        y = rng.poisson(A @ amp + 120 * np.exp(-e / 6) + 3)
        spec = core.Spectrum(y.astype(float), "synthetic", live_time=live)
        cfg2 = core.Config(elements=cfg.elements, zero=0.0, gain=0.0099, noise=0.11, emin=1.5, emax=18, e0=20.0)
        res = core.fit(spec, cfg2)
        for k, a in zip(names, amp):
            self.assertAlmostEqual(res.areas[k][0] / a, 1.0, delta=0.05, msg=k)
        q = core.quantify_theory(res)
        tot = sum(n_true.values())
        for k, v in n_true.items():
            self.assertAlmostEqual(q[k][0], v / tot, delta=0.03, msg=k)

    def test_unexcited_group_flagged(self):
        r = core.fit(core.Spectrum(np.random.default_rng(1).poisson(50, 2048).astype(float)),
                     core.Config(elements=["Cu-K"], emin=6, emax=12, gain=0.01))
        q = core.quantify_theory(r, e0=5.0)      # 5 keV is below the Cu K edge
        self.assertTrue(math.isnan(q["Cu-K"][0]))


if __name__ == "__main__":
    unittest.main()



def test_beam_scatter_and_polychromatic():
    import numpy as np
    from xrfit import core, demo, tube, webapp
    spec = demo.synthetic_spectrum()
    p = core.Config().params(); p.update(zero=0.0237, gain=0.0099951, noise=0.12, tail_frac=0.05)
    e = p["zero"] + p["gain"] * np.arange(len(spec.counts))
    Ec = 17.44 / (1 + 17.44 / core.ELECTRON_MC2)
    rng = np.random.default_rng(3)
    spec.counts = spec.counts + rng.poisson(3000 * core.peak(e, 17.44, p, p["gain"])
                                            + 1500 * core.peak(e, Ec, dict(p, noise=np.hypot(.12, .2)), p["gain"]))
    cfg = webapp.cfg_from_json({**demo.SUGGEST, "emax": 19, "beam": [{"energy": 17.44, "weight": 1, "scatter": True}]})
    r = core.fit(spec, cfg)
    assert abs(r.areas["Rayleigh 17.44"][0] - 3000) < 400 and abs(r.areas["Compton 17.44"][0] - 1500) < 400
    # a weighted beam averages the L-line cross sections; a tube table sums to 1 and keeps the anode K lines
    rows = tube.tube_beam("Rh", 40)
    assert abs(sum(x["weight"] for x in rows) - 1) < 1e-9 and any(x["scatter"] for x in rows)
    c2 = webapp.cfg_from_json({**demo.SUGGEST, "beam": rows})
    assert c2.e0 > 39 and core.fit(spec, c2).areas["Fe-K"][0] > 30000
    for bad in ([{"energy": -1, "weight": 1}], [{"energy": 10, "weight": 1, "use": False}]):
        try:
            webapp.cfg_from_json({"beam": bad}); assert False
        except webapp.ApiError:
            pass


def test_tube_characteristic_lines_share_one_scatter_group():
    import numpy as np
    from xrfit import core, demo, tube, webapp
    ref = demo.synthetic_spectrum()
    p = core.Config().params(); p.update(zero=0.0237, gain=0.0099951, noise=0.12, tail_frac=0.05)
    e = p["zero"] + p["gain"] * np.arange(len(ref.counts)); g = p["gain"]
    rows = tube.tube_beam("Mo", 35)
    lines = [r for r in rows if r["scatter"]]
    assert len(lines) >= 3
    tot = sum(r["weight"] for r in lines)
    pc = dict(p, noise=np.hypot(.12, .2))
    ray = sum(r["weight"] / tot * core.peak(e, r["energy"], p, g) for r in lines)
    com = sum(r["weight"] / tot * core.peak(e, r["energy"] / (1 + r["energy"] / core.ELECTRON_MC2), pc, g) for r in lines)
    spec = demo.synthetic_spectrum()
    spec.counts = ref.counts + np.random.default_rng(5).poisson(6000 * ray + 3000 * com)
    cfg = webapp.cfg_from_json({**demo.SUGGEST, "emax": 20.3, "beam": rows})
    r = core.fit(spec, cfg)
    assert {"Rayleigh", "Compton"} <= set(r.areas) and r.chi2red < 2
    base = core.fit(ref, webapp.cfg_from_json({**demo.SUGGEST, "emax": 20.3, "beam": [{"energy": 35, "weight": 1}]}))
    for n in ("Fe-K", "Cu-K", "Zn-K", "Pb-L"):
        assert abs(r.areas[n][0] - base.areas[n][0]) < 3 * r.areas[n][1]


def test_attenuation_matches_nist():
    from xrfit import attenuation as a
    assert abs(a.mu("Pb", 20.0) - 86.36) < 0.1                       # NIST XCOM, cm2/g
    water = 0.1119 * a.mu("H", 10.0) + 0.8881 * a.mu("O", 10.0)
    assert abs(water - 5.329) < 0.02


def test_composition_parser():
    import pytest
    from xrfit import matrix as M
    e = {x["element"]: x for x in M.parse_composition("Fe 70\nCr ?\nO balance")}
    assert e["Fe"]["value"] == 0.7 and e["Cr"]["mode"] == "unknown" and e["O"]["mode"] == "balance"
    sio2 = {x["element"]: x["value"] for x in M.parse_composition("SiO2")}
    assert abs(sio2["Si"] - 0.4674) < 1e-3
    for bad in ("Xx 3", "Fe 120", "Fe 3, Fe ?", "Fe"[:0]):
        with pytest.raises(M.MatrixError):
            M.parse_composition(bad)


def test_matrix_round_trip():
    import pytest
    from types import SimpleNamespace as NS
    from xrfit import core, matrix as M
    cfg = core.Config(elements=["Fe", "Cr", "Ni"], beam=[{"energy": 17.44, "weight": 1.0}])

    def result(mx_true, names, K=5e7):
        m = M.Model(mx_true, names, core.excitation(cfg), "xraylib")
        fr = [{e["element"]: e["value"] for e in L["entries"]} for L in m.layers]
        mass = [float("inf") if L["t_um"] is None else L["rho"] * L["t_um"] * 1e-4 for L in m.layers]
        f = m.predict(fr, mass, True)       # the "measurement" includes secondary excitation
        return NS(cfg=cfg, live_time=100.0, areas={n: (K * v * 100, (K * v * 100) ** 0.5) for n, v in f.items()})

    geo = {"alpha_in": 45, "alpha_out": 45}
    # bulk steel, every amount unknown: composition and K come back
    res = result({"layers": [{"composition": "Fe 70\nCr 18\nNi 12", "density": 7.9}], **geo}, ["Fe-K", "Cr-K", "Ni-K"])
    r = M.solve({"layers": [{"composition": "Fe ?\nCr ?\nNi ?", "density": 7.9}], "fit_k": True, **geo}, res)
    got = {c["element"]: c["wt_pct"] for c in r["layers"][0]["composition"]}
    assert abs(got["Fe"] - 70) < 0.05 and abs(got["Cr"] - 18) < 0.05 and abs(r["k"] / 5e7 - 1) < 1e-3
    # balance element with a given K
    r = M.solve({"layers": [{"composition": "Fe balance\nCr ?\nNi ?", "density": 7.9}], "fit_k": False, "k": 5e7, **geo}, res)
    assert abs({c["element"]: c["wt_pct"] for c in r["layers"][0]["composition"]}["Cr"] - 18) < 0.05
    # a balance element cannot be combined with a fitted K
    with pytest.raises(M.MatrixError):
        M.solve({"layers": [{"composition": "Fe balance\nCr ?", "density": 7.9}], "fit_k": True, **geo}, res)
    # 5 um Ni coating on Fe: the thickness is recovered
    res = result({"layers": [{"composition": "Ni", "density": 8.9, "thickness_um": 5}, {"composition": "Fe", "density": 7.87}], **geo},
                 ["Fe-K", "Ni-K"])
    r = M.solve({"layers": [{"composition": "Ni", "density": 8.9, "thickness_um": 2, "fit_thickness": True},
                            {"composition": "Fe", "density": 7.87}], "fit_k": False, "k": 5e7, **geo}, res)
    assert abs(r["layers"][0]["thickness_um"] - 5) < 0.01
    # the same element unknown in two layers is rejected
    with pytest.raises(M.MatrixError):
        M.solve({"layers": [{"composition": "Fe ?", "density": 7, "thickness_um": 1}, {"composition": "Fe ?", "density": 7}], **geo}, res)


def test_matrix_secondary_excitation_and_layer_ratios():
    import math
    from types import SimpleNamespace as NS
    from xrfit import core, matrix as M
    cfg = core.Config(beam=[{"energy": 17.44, "weight": 1.0}])
    exc = core.excitation(cfg)
    geo = {"alpha_in": 45, "alpha_out": 45}
    # Ni K lines excite Fe K in a 50/50 alloy: a large effect that must be there, and Ni cannot be enhanced by Fe
    m = M.Model({"layers": [{"composition": "Fe 50\nNi 50", "density": 8.3}], **geo}, ["Fe-K", "Ni-K"], exc, "xraylib")
    fr = [{"Fe": 0.5, "Ni": 0.5}]
    p0, p1 = m.predict(fr, [math.inf]), m.predict(fr, [math.inf], True)
    assert 0.2 < p1["Fe-K"] / p0["Fe-K"] - 1 < 0.45 and p1["Ni-K"] == p0["Ni-K"]
    # the same two elements unknown in the coating AND the substrate: the ratio in each layer comes back when
    # Kalpha and Kbeta are fitted as separate groups (two measurements per element)
    names = ["Cu-Ka", "Cu-Kb", "Zn-Ka", "Zn-Kb"]
    truth = {"layers": [{"composition": "Cu 80\nZn 20", "density": 8.7, "thickness_um": 5},
                        {"composition": "Cu 60\nZn 40", "density": 8.5}], **geo}
    m = M.Model(truth, names, exc, "xraylib")
    fr = [{"Cu": 0.8, "Zn": 0.2}, {"Cu": 0.6, "Zn": 0.4}]
    f = m.predict(fr, [8.7 * 5e-4, math.inf], True)
    res = NS(cfg=cfg, live_time=100.0, areas={n: (3e7 * v * 100, (3e7 * v * 100) ** 0.5) for n, v in f.items()})
    mx = {"layers": [{"composition": "Cu ?\nZn ?", "density": 8.7, "thickness_um": 5},
                     {"composition": "Cu ?\nZn ?", "density": 8.5}], "fit_k": False, "k": 3e7, **geo}
    r = M.solve(mx, res)
    ratio = [L["ratios"][0] for L in r["layers"]]            # Zn relative to Cu, by mass
    assert abs(ratio[0]["mass"] - 0.25) < 0.01 and abs(ratio[1]["mass"] - 2 / 3) < 0.02
    assert abs(r["layers"][0]["composition"][0]["at_pct"] + r["layers"][0]["composition"][1]["at_pct"] - 100) < 1e-6
    # only one line group per element cannot separate two layers: refused with a hint
    import pytest
    res1 = NS(cfg=cfg, live_time=100.0, areas={"Cu-K": (1e5, 300.0), "Zn-K": (5e4, 220.0)})
    with pytest.raises(M.MatrixError, match="Fit α, β and γ"):
        M.solve(mx, res1)


def test_beam_filters_harden_the_beam():
    import math
    import pytest
    from xrfit import core, matrix as M
    beam = [{"energy": 10.0, "weight": 1.0}, {"energy": 20.0, "weight": 1.0}]
    cfg = core.Config(beam=beam, beam_filters=[{"material": "Al", "density": "", "thickness_um": 200}])
    (e1, w1), (e2, w2) = core.excitation(cfg)
    # Beer-Lambert with the tabulated Al attenuation and density
    from xrfit import attenuation as a
    for e, w in ((e1, w1), (e2, w2)):
        assert abs(w - math.exp(-a.mu("Al", e) * a.density("Al") * 200e-4)) < 1e-9
    assert w2 / w1 > 3                                   # the high-energy end is favoured
    # filters stack
    cfg.beam_filters.append({"material": "Kapton", "density": "", "thickness_um": 50})
    (_, w1b), _ = core.excitation(cfg)
    assert w1b < w1
    # a compound needs a density, an unknown symbol is named, a wall of metal is reported
    with pytest.raises(M.MatrixError, match="density"):
        M.parse_filters([{"material": "SiO2", "density": "", "thickness_um": 5}])
    with pytest.raises(M.MatrixError, match="Xx"):
        M.parse_filters([{"material": "Xx", "density": "", "thickness_um": 5}])
    cfg.beam_filters = [{"material": "Pb", "density": "", "thickness_um": 5000}]
    with pytest.raises(ValueError):
        core.excitation(cfg)


def test_detector_filters_reduce_low_energy_efficiency():
    import math
    from xrfit import core, matrix as M
    exc = core.excitation(core.Config(beam=[{"energy": 17.44, "weight": 1.0}]))
    geo = {"alpha_in": 45, "alpha_out": 45}
    lay = [{"composition": "Fe 50\nCu 50", "density": 8.4}]
    base = M.Model({"layers": lay, **geo}, ["Fe-K", "Cu-K"], exc, "xraylib")
    filt = M.Model({"layers": lay, "detector": {"filters": [{"material": "Al", "density": "", "thickness_um": 50}]}, **geo},
                   ["Fe-K", "Cu-K"], exc, "xraylib")
    fr = [{"Fe": 0.5, "Cu": 0.5}]
    p0, p1 = base.predict(fr, [math.inf]), filt.predict(fr, [math.inf])
    r_fe, r_cu = p1["Fe-K"] / p0["Fe-K"], p1["Cu-K"] / p0["Cu-K"]
    assert r_fe < r_cu < 1                              # soft Fe K is cut harder than Cu K


def test_srm1157_example_is_fully_prefilled():
    import json
    from xrfit import webapp
    status, _, body, _ = webapp.dispatch("GET", "/api/example?name=srm1157", {}, b"")
    d = json.loads(body)
    sug = d["suggest"]
    assert status == 200 and d["live_time"] > 95 and "REDACTED" in json.dumps(d["meta"])
    assert sug["tube"]["anode"] == "Rh" and sug["tube"]["kv"] == 30.0 and sug["sum_peaks"]
    assert {"Fe", "Cr", "Mo", "V", "W-L", "Rh-L", "Ar"} <= set(sug["elements"])
    mx = sug["matrix"]
    assert mx["layers"][0]["composition"].count("?") == 9 and mx["fit_k"]
