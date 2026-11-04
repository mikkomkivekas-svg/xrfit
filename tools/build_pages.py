#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the static GitHub Pages site into docs/ (UI + Pyodide worker + zipped Python package).

    python tools/build_pages.py

Then in the GitHub repo: Settings > Pages > Deploy from a branch > main, folder /docs.
"""
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "xrfit")
OUT = os.path.join(ROOT, "docs")
STATIC = os.path.join(PKG, "web", "static")

os.makedirs(OUT, exist_ok=True)
html = open(os.path.join(PKG, "web", "index.html"), encoding="utf-8").read()
tag = '<script src="shim.js"></script>\n<script>\n"use strict";'
assert html.count('<script>\n"use strict";') == 1
html = html.replace('<script>\n"use strict";', tag, 1)
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
for f in ("worker.js", "shim.js"):
    shutil.copy(os.path.join(STATIC, f), os.path.join(OUT, f))
open(os.path.join(OUT, ".nojekyll"), "w").close()

with zipfile.ZipFile(os.path.join(OUT, "xrfit-py.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(PKG):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", "web"))
        for f in sorted(fn):
            if f.endswith((".pyc",)):
                continue
            full = os.path.join(dp, f)
            info = zipfile.ZipInfo(os.path.relpath(full, ROOT), (2020, 1, 1, 0, 0, 0))  # stable bytes
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, open(full, "rb").read())
print("built", OUT, sorted(os.listdir(OUT)))
