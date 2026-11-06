# xrfit

PyMca-style XRF spectrum fitting with a JaBS-style scriptable command interpreter.

## Install / run
    pip install numpy scipy matplotlib
    pip install -e .
    xrfit                      # interactive prompt
    xrfit examples/demo.xrf    # run a script
    xrfit -c "load a.txt; add Fe Cu; fit"

## Incident spectrum

Like PyMca's beam table, the excitation is a list with one energy per line: **energy (keV)**, **weight** (relative photon intensity) and an optional word **scatter**, for example `17.44 1.0 scatter`. One line is a monochromatic beam; several lines describe a tube or a multi-line source. In the browser it is a plain text box; delete a line to remove an energy.

- The weights set the line intensities: L-line family ratios and the theory quantification use each cross section averaged over the beam, and a bare element symbol picks K lines when the highest enabled energy reaches the K edge.
- Lines marked **scatter** add two fitted groups, Rayleigh at the beam energies and Compton at `E / (1 + E (1 - cos θ) / 511 keV)`. Set the scattering angle θ (default 90°). All scatter lines share one Rayleigh amplitude and one Compton amplitude, with relative sizes set by their weights (so Rh Ka1, Ka2 and Kb are fitted as one pattern, not as competing peaks). Scatter groups are left out of the quantification. The Compton peak is widened by `compton_broad` (keV, added in quadrature to the detector width; it can be fitted).
- **Load from file** reads a text file in the same format (`#` comments, a header line at the top is skipped). **Generate from X-ray tube** fills the list from a simplified model: Kramers continuum plus the anode K lines, with no window or filter absorption. PyMca uses the Ebel model for this; xrfit does not have the attenuation data it needs, so for a quantitative tube spectrum compute it elsewhere and load it.
- Shell: `beam`, `beam add 17.44 1 scatter`, `beam load tube.txt`, `beam tube Rh 40 24 25`, `beam clear`; `set scatter_angle 90`; `free compton_broad`.

Not modelled: polarisation of the beam, and the sample, filter and detector efficiency terms that PyMca's fundamental-parameter fit applies to the incident spectrum.

## Calibrating the tube on a standard

The simplified tube model has one free number that matters for elements just above the anode lines: the share of
photons in the anode K lines. Tick **Fit the anode line share** with every amount of a standard given (no `?`), and
the calculation fits K and the share together; **Use this share** regenerates the incident spectrum with it. For the
SRM 1157 example it gives about 3 %, against 25 % as a first guess. Mo K, which only the anode lines and the hard
continuum can excite, moves from 0.8 to about 3.3 wt % (certified 4.9) when the share is set this way.

## Reference values

Each layer has an optional **Reference values** box (for example a certificate: `Fe balance`, `Cr 4.36`, ...). The
result table then shows the reference, renormalised to the elements of the layer, and the difference to the fit. The
SRM 1157 example carries the NBS certificate values.

## Instrument lines

Lines that come from the instrument and not the sample (the tube anode's L lines, argon from the air path) can be fitted
and marked **not sample** on their chip (`instrument` in the settings). They keep their net area, but they get no atomic
share and the matrix calculation ignores them. The K lines of the anode work differently: scatter-flagged rows of the
incident spectrum are fitted as one Rayleigh and one Compton peak.

## Filters

Both lists use the same tabulated attenuation data as the matrix model (Elam, Ravel and Sieber, via XrayDB) and
take an element (`Al`), a formula (`SiO2`, density required) or a preset (Kapton, Mylar, Polypropylene).

- **Beam filters** sit between the source and the sample. Their transmission multiplies the weight of every
  incident energy, so the beam hardens, the line intensities, Rayleigh and Compton peaks follow, and the matrix
  calculation sees the filtered beam. Use them to describe the tube window and any primary filter on top of the
  simplified tube spectrum. In the shell: `filter beam add Al 100`, `filter T 5 10 20`.
- **Detector filters** sit between the sample and the detector. They multiply the detection efficiency in the
  matrix model (`filter detector add Be 25`), next to the Si thickness, Be window and air path.

## Sample matrix

Describe the sample as 1 to 6 layers (browser: choose the number of layers first, then fill in each; shell: `matrix`). Layer 1 faces the beam and the detector, and **the last layer is infinitely thick**. Each layer has a composition, a density and (except the last) a thickness.

Composition lines: `Fe 70` (wt %), `Cr ?` (amount found from the fitted lines), `O balance` (whatever is left to 100 %, for elements XRF does not see), or a formula such as `SiO2` or `Fe2O3 40`. A single formula or element is the whole layer; the density of a pure element can be left blank.

**What is calculated.** After a fit, *Calculate composition* (shell: `quantify matrix`) predicts the rate of every fitted line group from the layers and compares it with the measured net rate. For element *i* with mass fraction *w* in layer *j*, line *l* and beam energy *E0*:

`rate = K · σ_l(E0) · N_A/A_i · w · T_above · (1 − exp(−χM)) / (χ sin α_in) · ε(E_l)`, with `χ = μ_j(E0)/sin α_in + μ_j(E_l)/sin α_out`

where σ is the production cross section used elsewhere in xrfit, μ the mass attenuation of the layer, M = density × thickness (infinite for the last layer), T_above the attenuation of the layers above, and ε the detector path (air, Be window, Si thickness). The beam can be polychromatic (the incident spectrum list). Unknown amounts, **K** (flux × solid angle, lumped) and layer thicknesses marked *fit* are found by least squares with a start from a Criss–Birks style iteration. K can be fitted when the amounts sum to 100 %; with a balance element, give K from a standard (all amounts fixed, *Fit K* on, then copy the value). Uncertainties come from the fit covariance (shown only when there are fewer unknowns than line groups).

**Secondary excitation** (a line of one element exciting another, such as Ni K lines enhancing Fe K) is included by default: every line of the sample whose energy is above a fitted line's edge is treated as an isotropic source, the sample is cut into thin cells, and the cell-to-cell transfer uses exact double integrals of the exponential-integral kernel. For a bulk Fe-Ni alloy it agrees with a brute-force numerical integration to 0.03 %. Switch it off with `matrix secondary off`.

**The same element in several layers.** An element may be unknown in more than one layer (for example Cu and Zn in both a coating and its substrate) as long as there are at least as many independent measurements of it as layers: fit α, β and γ lines separately, so Kα and Kβ (absorbed differently, hence sensitive to depth) are two data points. Every layer reports weight and atomic percent and the ratios of its other elements to its first unknown element (by mass and by atoms), with uncertainties from the fit covariance, which are given also when the problem is exactly determined (then they are statistical only). Strongly correlated quantities are flagged.

**Still left out:** tertiary excitation, scatter, roughness, and feeding the matrix-corrected line ratios back into the peak fit (inside an L group the Lα:Lβ ratio is fixed from the cross sections at the beam energy). A balance element needs K from a standard, because with only relative intensities K and the absolute amounts cannot be separated. Absorption data are the Elam, Ravel and Sieber tables (cm²/g, 0.1 to 800 keV) via the public-domain XrayDB compilation (`tools/build_attenuation.py`).

## Browser interface

```bash
pip install -e .
xrfit-web            # opens http://127.0.0.1:8765 in your browser
```

Options: `--port N`, `--no-browser`, `--host`. Open or drop a spectrum (KETEK `.vspc`, CSV, text columns), add elements (`Fe`, `Pb-L`, `Cu-Ka1`), press **Fit spectrum** (Ctrl+Enter). The page shows the metadata (name, measurement time, slow filter), the plot with residuals and line markers (wheel zoom, drag pan, double-click reset, arrow keys step a cursor), net areas with atomic shares, a line-data lookup, batch fitting, CSV download and settings save/load. Light and dark themes are supported.

The server is single-user and has no login. It only accepts requests addressed to localhost and requires a custom header on writes, so keep it bound to `127.0.0.1` unless you add your own protection.

### Hosted version (GitHub Pages)

The same interface also runs with no server: `docs/` holds a static build in which the Python backend runs in the browser through [Pyodide](https://pyodide.org) (numpy and scipy compiled to WebAssembly). Spectra stay on your computer. The first visit downloads about 30 MB from the jsDelivr CDN and takes several seconds to start.

Enable it once under **Settings → Pages → Deploy from a branch → `main` / `/docs`**. The site is then at `https://<user>.github.io/xrfit/`. After changing anything in `xrfit/`, rebuild with `python tools/build_pages.py` and commit `docs/`.

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
- KETEK `.vspc` (JSON; `examples/srm1157_tool_steel.vspc` is a real export (SRM 1157 tool steel, Rh tube at 30 kV) with the serial numbers redacted: `RawData`, `Calibration`, `Configuration.Parameters`, `MeasurementInfo`) - checked against a VICO-Scope export;
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
- Commands: `elements`, `lines Fe`, `xsec Fe-Ka1 8 10 20`, `set e0 <keV>`, `set xs_source xraylib|krause`, `beam ...` (incident spectrum, below).

Known limits of the source data
- xraylib 2.3 uses jump approximations for L lines; the PDF itself says values for L1-related lines are
  discrepant. Cross sections of L lines stop at the K-edge energy (extrapolated above it).
- Rows at the L1/L2/L3 edge energies in the PDF are not smooth; they are ignored for interpolation.
  The Mo K-shell table has a second row at exactly the K edge (pre-edge xraylib value); also ignored.
- La2, Lb2 and Ma1 are in Table 1-2 but have no rate or cross section in xraylib: single-line tokens only.
- Z>90 and Z<5 have no xraylib data (use single lines, e.g. `U-La1`). Ca L2/L3 lines have no cross sections.
- `xrfit/data/attenuation.json` - mass attenuation (photo, coherent, incoherent) of Z=1-98 and atomic weights/densities, from Elam, Ravel and Sieber (2002) as compiled in XrayDB (public domain, CC0): https://github.com/xraypy/XrayDB. Rebuild with `tools/build_attenuation.py`.
- 2003 values; current xraylib releases differ slightly.

## Not implemented (vs. PyMca)
Tertiary excitation, an Ebel tube model with window and filter, beam polarisation,
HDF5/EDF/SPEC I/O, XRF mapping.

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
