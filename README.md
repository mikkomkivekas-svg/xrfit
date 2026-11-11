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
- **Load from file** reads a text file in the same format (`#` comments, a header line at the top is skipped). **Generate from X-ray tube** fills the list from a simplified model: Kramers continuum plus the anode K lines (or the L lines when the voltage is below the K edge, for example a 5 kV Rh tube), with no window or filter absorption; use the beam filters for the tube window. Use more continuum steps at low voltage, so that the Al and Si K edges are resolved. PyMca uses the Ebel model for this; xrfit does not have the attenuation data it needs, so for a quantitative tube spectrum compute it elsewhere and load it.
- Shell: `beam`, `beam add 17.44 1 scatter`, `beam load tube.txt`, `beam tube Rh 40 24 25`, `beam clear`; `set scatter_angle 90`; `free compton_broad`.

Not modelled: polarisation of the beam, and the sample, filter and detector efficiency terms that PyMca's fundamental-parameter fit applies to the incident spectrum.

## Escape peaks and the long tail

The Si escape peak is calculated for every line above the Si K edge (Reed and Ware: Si fluorescence yield, jump ratio and Kα share, with the Si attenuation at the line and at Si Kα). The escape ratio is 1.2 % at Rh Lα, 1.1 % at Ar Kα and 0.3 % at Fe Kα. **Si escape scale** (`esc_scale`, default 1) multiplies it and can be fitted to correct for the real crystal and dead layer; **Extra escape fraction** (`esc_frac`) adds a constant ratio on top.

Strong soft lines (Rh L, Ar K) sit on a low-energy shelf. By default the shelf is taken by the background, **SNIP plus a cubic polynomial**, and the long tail is off. **Long tail fraction / length** (`tail2_frac`, `beta2`) are an option that adds a second, long exponential tail next to the short one. Use it with care: it is added on top of a SNIP curve that already follows the shelf, so it counts the shelf twice. On a 5 kV kaolinite spectrum it lifted the baseline under Al Kα by 30–50 counts and halved the Al area, and it made the resolution parameters drift when it was switched off without a flexible background. On a 5 kV kaolinite spectrum (600 ns peaking time) the calculated escape gives a fitted scale of 1.1–1.2.

## Calibration tab

Fits a straight line `signal = a + b × amount` to standards of known amount (film mass per area, thickness, concentration) and reads unknowns off it.

1. Choose the standards' spectra and press **Read spectra**. The signal is either the counts in an energy window (optionally minus a line drawn between the window edges) or the net peak area of one line group from a full fit with the current Elements settings. **Counts per second** divides by live time.
2. Type the known amount next to each spectrum. A spectrum with no amount is treated as an unknown. Untick a row to leave it out of the fit.
3. The page shows slope and intercept with errors, R², the scatter about the line in amount units, a 3σ detection limit from the counting error of the lowest standard, and the amount read from the line for every spectrum, with its uncertainty.

Options: force the line through zero, and weight by counting error. The result downloads as a table (CSV) and as calibration constants (JSON). The energy window uses the zero and gain you enter, which start from the Elements settings.

## Geometry

Three angles describe the geometry, and xrfit keeps them as separate settings: the beam angle and the detector angle to the sample surface (Sample matrix; they set the path lengths in the sample), and the scattering angle (Beam; it sets the Compton shift). With tube, sample normal and detector in one plane, the position angles from the normal give them directly: beam = 90° − tube angle, detector = 90° − detector angle, and the scattering angle is 180° − (tube angle + detector angle) when tube and detector are on opposite sides of the normal, or 180° − |difference| on the same side. **Set the angles from the tube and detector positions** in the Matrix tab (or `matrix geometry <tube> <detector> [same]` in the shell) fills in all three. For a tube along the normal and a detector 47° from it: beam 90°, detector 43°, scattering angle 133°, which is the soft-line example's geometry. The defaults (45°, 45°, 90°) fit no particular instrument, so set them before using the matrix model.

## Beam spot size

A broad excitation spot is not seen uniformly. Fluorescence from the edge of a 40 mm spot reaches the detector through a longer, slanted air path and at a smaller solid angle than fluorescence from the centre. Under **Sample matrix → Detector and path**, set the **beam spot diameter** (a uniform disc, or the FWHM of a Gaussian) and the matrix calculation averages the air absorption and solid angle over the spot: each ring of radius ρ is weighted by its share of the emitted light, with a solid-angle factor (d/r)³ and air transmission exp(−μr), where r = √(d² + ρ²) and d is the air path. A spot of 0 reproduces the earlier point-source model exactly.

The effect depends on energy, so it does not cancel into the instrument factor K. For a 40 mm disc at 30 mm it lowers Al Kα to about half the point-source value and Rh Lα to about 0.7. A model without it reads the extra absorption as a longer air path when you fit the air distance.

The **Distance calculator** in the same section lists, for Al, Si, S, Rh L, Ar and Fe lines, the on-axis air transmission, the spot-averaged factor, the weight of the spot edge against its centre, and the gain in detected rate if the detector moves to another distance (with the spot, and for a point source). It assumes the sample is flat, uniformly emitting and faces the detector; it does not include dead time.

## Calibrating the air path

When the detector looks at the sample through air (soft lines such as Al-K and Si-K at low kV), the distance matters a lot: 30 mm of air passes about 1.2 % of Al-K and 6 % of Si-K, so a 1 mm error changes the Al signal by roughly 16 %. Tick **Fit the air path** in the Matrix tab, give every amount of a standard that has two lines far apart in energy (kaolinite: Al-K and Si-K, Al/Si mass ratio about 0.96), and the effective distance is fitted together with K. The result shows start value, fitted value and uncertainty; **Use this air path** copies it into the detector settings. Shell: set `matrix` key `fit_air` true in the JSON; result field `air_path`.

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

**Defaults.** The Si escape scale is a free parameter and α, β and γ lines of a shell are fitted as separate groups (`free_ratios`), so that the Kα and Kβ of the sample elements (for example Si, whose Kβ lies next to the Si escape and Rh L structure) are not tied to tabulated ratios. Untick **Fit α, β and γ lines separately** for the older combined groups. The bundled examples keep the combined groups.

**Argon from the air** is added automatically. When an air path is set (Sample matrix → Detector and path, or `matrix.detector.air_mm`), the fit includes an `Ar-K` group marked as an instrument line, unless you have already added an Ar token yourself. The select **Argon from air** in the Elements panel (`argon` in the settings, `set argon auto|on|off` in the shell) changes this: `on` fits it even with no air path entered, `off` leaves it out. This group is always one tied Kα + Kβ group, even when the sample elements are split into α and β: only its Kα height is fitted, and the Kβ line (and its escape peak at 1.45 keV, 36 eV below Al Kα) follows from the tabulated ratio. Its escape peaks follow from the calculated Si escape ratio. The Ar Kβ escape peak at 1.45 keV is the small bump you see between the Si Kα peak and the Ar Kα escape peak at 1.22 keV.

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

### Thick reference

For thin films of a single element, a bulk sample of the same material (for example an Al foil for Al films) can serve as the reference. Under **Thick reference** in the Calibration tab, choose the reference spectrum, the line (`Al-K`), and the film, reference and substrate compositions and densities. The film mass per area comes from the ratio of the film's signal to the reference signal, converted by the matrix model: for a film of mass m on the substrate the model gives R(m) = signal ÷ signal of the infinitely thick reference, which is inverted for every spectrum. The table gives the film mass (g/m²), the mass of the element, and the thickness (mass ÷ density).

Position, air path, solid angle and detector response cancel in the ratio, so measure the reference with the same settings as the films. The scale of R(m) depends on how deep the excitation reaches, so it needs the real incident spectrum (Beam settings, for example the Rh L lines of the tube): for Al with the 5 kV Rh spectrum a thin film gives about 32 % of the foil signal per g/m², but a single 5 keV energy gives only 8.5 %. If you have a film of known mass, tick **Scale to films with a known amount**: the model is scaled to the known masses (least squares through the origin), which removes this uncertainty. A film whose signal is at or above the reference is not a thin film and is flagged.

## Soft-line example

The **Si coating, soft lines** button loads a KETEK spectrum of a Si-containing coating on a substrate (5 kV Rh tube, 30 mm of air, 600 ns peaking time) with every setting filled in, as a reference for similar data: energy calibration (gain 0.00487 keV/channel from the Si K, Rh L and Ar K peaks, since the file calls itself uncalibrated), resolution (noise and Fano free), Si escape scale free, no long tail and a SNIP + cubic polynomial background (a long tail on top of SNIP counts the shelf twice and halves the Al Kα area), alpha and beta lines separate, the Rh L lines (Lα, Lβ1 and the weak Ll at 2.377 keV, which otherwise inflates S) marked as instrument lines, the 30 mm air path and 40 mm spot in the detector section, and the Ar K group added automatically. The fit gives a reduced χ² of about 4.6.

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
