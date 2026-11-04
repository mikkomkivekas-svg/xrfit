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

