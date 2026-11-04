# xrfit

PyMca-style XRF spectrum fitting with a JaBS-style scriptable command interpreter.

## Install / run
    pip install numpy scipy matplotlib
    pip install -e .
    xrfit                      # interactive prompt
    xrfit examples/demo.xrf    # run a script
    xrfit -c "load a.txt; add Fe Cu; fit"

## Implemented
- Gaussian peaks with energy-dependent width: FWHM^2 = noise^2 + 2.3548^2 * eps * Fano * E
- Hypermet low-energy tail, step function, Si-escape peaks, optional sum (pile-up) peaks
- Fitted energy calibration (zero, gain), resolution (noise, Fano), tail, step, escape
- Variable projection: nonlinear params by bounded TRF, line-group areas by bounded linear LSQ (>= 0)
- Poisson re-weighting, parameter and area uncertainties, reduced chi^2
- Backgrounds: SNIP (LLS), polynomial, SNIP+polynomial, none
- Line groups from the bundled tables (see *Data* below): K family weighted by xraylib radiative rates,
  L family weighted by xraylib production cross sections at the excitation energy `e0`; single lines
  (`Pb-La2`) and free Ka/Kb or La/Lb/Lg ratios (`set free_ratios on`)
- Batch fitting (previous result seeds next), CSV export, plot
- Quantification: against a standard (`calibrate`/`quantify`) or from theory (`quantify theory`, relative atomic
  amounts = net rate / production cross section at `e0`; thin sample, no detector/matrix correction)

## Input formats
- KETEK `.vspc` (JSON; `examples/sample_ketek.vspc` is a real export with the serial numbers redacted: `RawData`, `Calibration`, `Configuration.Parameters`, `MeasurementInfo`) - checked against a VICO-Scope export;
  live time and keV/bin are taken from the file. Also the `Channel,RawData` CSV export, Amptek .mca, plain text.
- `info` / `info all` show file name, serial, start/end time, real/live/dead time, count rates, slow-filter
  peaking and gap time, SDD temperature and calibration. `plot` prints the key items on the figure.

## Data (bundled, extracted from the two source PDFs)
- `xrfit/data/xdb_table_1_2.json` - X-Ray Data Booklet Table 1-2: Ka1 Ka2 Kb1 La1 La2 Lb1 Lb2 Lg1 Ma1 in eV, Z=3-95.
- `xrfit/data/xraylib_v23.json` - XRAYLIB tables v2.3 (2003): edges, fluorescence yields (xraylib and Krause),
  jump factors, line energies, fractional radiative rates, K- and L-shell XRF cross sections (barns/atom) on
  the tabulated grid, each as xraylib value and Krause value, Z=5-90.
- Line energies use Table 1-2 where listed (Kb3, Lb3, Ll from xraylib). Cross sections are log-log
  interpolated on the tabulated grid; `xsec` prints a note when a value is extrapolated (e.g. between an
  edge and the first grid point). `set xs_source krause` switches to Krause's column.
- `python tools/build_data.py <Table_1-2.pdf> <xraylib_tables_v2.3.pdf>` regenerates the JSON;
  `python tools/verify_against_pdf.py ...` re-checks every number with an independent extractor;
  `python -m unittest tests.test_xrfit` runs the tests.
- Commands: `elements`, `lines Fe`, `xsec Fe-Ka1 8 10 20`, `set e0 <keV>`, `set xs_source xraylib|krause`.

Known limits of the source data
- xraylib 2.3 uses jump approximations for L lines; the PDF itself says values for L1-related lines are
  discrepant. Cross sections of L lines stop at the K-edge energy (extrapolated above it).
- Rows at the L1/L2/L3 edge energies in the PDF are not smooth; they are ignored for interpolation.
  The Mo K-shell table has a second row at exactly the K edge (pre-edge xraylib value); also ignored.
- La2, Lb2 and Ma1 are in Table 1-2 but have no rate or cross section in xraylib: single-line tokens only.
- Z>90 and Z<5 have no xraylib data (use single lines, e.g. `U-La1`). Ca L2/L3 lines have no cross sections.
- 2003 values; current xraylib releases differ slightly.

## Not implemented (vs. PyMca)
Full fundamental-parameters quantification (matrix absorption, secondary excitation, multilayers; only the
thin-sample theory above), tube/excitation-source spectrum, detector/filter attenuation layers, Compton/Rayleigh scatter peaks,
HDF5/EDF/SPEC I/O, GUI, XRF mapping, photoabsorption/scattering cross-section tables beyond those listed above.

## JaBS compatibility
Same working style as JaBS (command interpreter, `.xrf` scripts like `.jabs` scripts, batch runs).
It is *not* linked to JaBS/jibal: JaBS is a C code for RBS/ERD/NRA and has no XRF model.
For true integration, expose jibal element/mass data via ctypes in `lines.py`.

## License
Copyright (C) 2026 the xrfit authors. xrfit is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version. It is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; see the [LICENSE](LICENSE) file for details.

The bundled data in `xrfit/data/` is extracted from third-party publications (X-Ray Data Booklet Table 1-2,
LBNL; XRAYLIB tables v2.3, ESRF/Univ. of Sassari, 2003). The GPL applies to the xrfit code; the data remain
subject to their original sources' terms and should be cited accordingly.
