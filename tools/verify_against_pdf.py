"""Independent check of xrfit/data/*.json against the PDFs using a different text extractor
(pdftotext). Compares, page by page, the multiset of numbers in the PDF with the numbers stored.

    python tools/verify_against_pdf.py Table_1-2.pdf xraylib_tables_v2.3.pdf
"""
import json
import re
import subprocess
import sys
from collections import Counter

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?(?![\w.])", re.I)


def pages(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
    return out.split("\f")


def nums(line):
    return [float(t.replace(",", "")) for t in NUM.findall(line)]


def check_t12(pdf, data):
    pdf_rows = {}
    for pg in pages(pdf):
        for line in pg.splitlines():
            m = re.match(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(.+)$", line)
            if m:
                pdf_rows[m.group(2)] = (int(m.group(1)), sorted(nums(m.group(3))))
    bad = []
    for sym, row in data.items():
        mine = sorted(v for k, v in row.items() if k != "Z" and v is not None)
        if sym not in pdf_rows or pdf_rows[sym][0] != row["Z"] or pdf_rows[sym][1] != mine:
            bad.append(sym)
    return len(data), bad


def check_xl(pdf, data):
    bad, n = [], 0
    for pg in pages(pdf):
        m = re.search(r"Element:\s*(\w+)\s*,\s*Z=\s*(\d+)", pg)
        if m:  # data page: everything after the title block
            sym = m.group(1)
            got = Counter()
            for line in pg.splitlines():
                if re.match(r"\s*(Edge \[|Fluor Yield|Jump Factor|Line energy|Radiative rate)", line):
                    got.update(nums(line.split("]")[-1] if "[" in line else re.sub(r"^\D+(\(\w+\))?", "", line)))
            e = data[sym]
            mine = Counter()
            for blk in ("edge_keV", "fluor_yield_xraylib", "fluor_yield_krause", "jump"):
                mine.update(v for v in e[blk].values() if v is not None)
            mine.update(v for v in e["line_keV"].values() if v is not None)
            mine.update(v for v in e["rate"].values() if v is not None)
            # pdftotext keeps the "(xraylib)"/"(Krause)" label digits out; compare as sets of values
            if Counter({k: v for k, v in got.items()}) != mine:
                bad.append(("A", sym, sorted((got - mine).items())[:3], sorted((mine - got).items())[:3]))
            n += 1
            continue
        m = re.search(r"for\s+(\w+)\s+\(Z=\s*(\d+)\)", pg)
        if m and "cross" in pg[:300]:
            sym = m.group(1)
            got = Counter()
            for line in pg.splitlines():
                if re.match(r"^\s*\d+\.\d+\s", line):
                    got.update(nums(line))
            e, mine = data[sym], Counter()
            for sect in ("xs_K", "xs_L"):
                t = e.get(sect)
                if not t: continue
                mine.update(t["E_keV"])
                for k, v in t.items():
                    if k == "E_keV": continue
                    for src in ("xraylib", "krause"):
                        mine.update(x for x in v[src] if x is not None)
            if got != mine:
                bad.append(("B", sym, sorted((got - mine).items())[:3], sorted((mine - got).items())[:3]))
            n += 1
    return n, bad


if __name__ == "__main__":
    t12 = json.load(open("xrfit/data/xdb_table_1_2.json"))
    xl = json.load(open("xrfit/data/xraylib_v23.json"))
    n, bad = check_t12(sys.argv[1], t12)
    print(f"Table 1-2: {n} element rows compared, mismatches: {bad}")
    n, bad = check_xl(sys.argv[2], xl)
    print(f"xraylib tables: {n} pages compared, mismatches: {len(bad)}")
    for b in bad[:10]: print("  ", b)
