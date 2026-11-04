#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build xrfit/data/attenuation.json from the XrayDB sqlite file (public domain, CC0).

    git clone --depth 1 https://github.com/xraypy/XrayDB
    python tools/build_attenuation.py XrayDB/xraydb.sqlite

Contents per element: atomic weight and density from XrayDB's element table, and the Elam, Ravel and Sieber (2002)
cross sections (cm^2/g) for photoabsorption, coherent and incoherent scattering as (ln E[eV], ln sigma, spline second
derivatives) so the original interpolation can be reproduced.
"""
import json
import os
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
out = {"_source": "XrayDB (CC0) compiled from Elam, Ravel and Sieber, Radiat. Phys. Chem. 63 (2002) 121; "
                  "energies in eV (ln), cross sections in cm^2/g (ln)", "elements": {}}
r5 = lambda v: [round(float(x), 5) for x in json.loads(v)]
for z, sym, mm, rho in db.execute("select atomic_number, element, molar_mass, density from elements where atomic_number <= 98"):
    ph = db.execute("select log_energy, log_photoabsorption, log_photoabsorption_spline from photoabsorption where element=?", (sym,)).fetchone()
    sc = db.execute("select log_energy, log_coherent_scatter, log_coherent_scatter_spline, log_incoherent_scatter, "
                    "log_incoherent_scatter_spline from scattering where element=?", (sym,)).fetchone()
    if not ph or not sc:
        continue
    out["elements"][sym] = {"Z": z, "A": mm, "rho": rho,
                            "photo": [r5(ph[0]), r5(ph[1]), r5(ph[2])],
                            "coh": [r5(sc[0]), r5(sc[1]), r5(sc[2])],
                            "incoh": [r5(sc[0]), r5(sc[3]), r5(sc[4])]}
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xrfit", "data", "attenuation.json")
json.dump(out, open(dst, "w"), separators=(",", ":"))
print(len(out["elements"]), "elements ->", dst, os.path.getsize(dst) // 1024, "kB")
