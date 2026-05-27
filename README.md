# A tool of Hierarchical cOre ideNtification and Kinematic property AssIgnment (**HONKAI**) for Dense Cores

An automated analysis pipeline for interstellar medium (ISM) molecular line data, built on [ScousePy](https://github.com/jdhenshaw/scousepy) + [ACORNS](https://github.com/jdhenshaw/acorns).

Supports spectral fitting, hierarchical structure identification, and multi-scale physical property computation (mass, virial mass, Jeans length, density, etc.) with visualization for radio telescope FITS data cubes.

## Features

- **Spectral fitting**: Interactive/automated fitting of molecular line data cubes via ScousePy
- **Hierarchical clustering**: Multi-scale structure identification in 2D (PP) or 3D (PPV) space using ACORNS (Agglomerative Clustering for ORganising Nested Structures)
- **Multi-wavelength cross-analysis (DUAL mode)**: Extract structures from one FITS file and map them onto another FITS file (different resolution / molecular line) for physical property computation
- **Multi-scale analysis**: Full hierarchy analysis from root nodes to leaf nodes
- **Visualization**: 3D cluster trees, dendrograms, contour maps, core mass function (CMF), Jeans length analysis, and more

## Directory Structure

```
.
├── config.py                  # Global configuration (paths, physical constants, clustering parameters, etc.)
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── src/
│   ├── loader.py              # FITS data loading
│   ├── processor.py           # ScousePy fitting & ACORNS clustering
│   ├── analyzer.py            # Physical property computation (single-file mode)
│   ├── multi_wavelength_analyzer.py  # Multi-wavelength cross-analysis (dual-file mode)
│   ├── visualizer.py          # Visualization
│   └── utils.py               # Utility functions
├── examples/
│   ├── run.py                 # ACORNS standalone example
│   ├── plot.py                # ACORNS standalone visualization example
│   ├── run_2d.py
│   └── plot_2d.py
└── data/
    └── inputs/                # Place your .fits files here
```

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. ScousePy & ACORNS

ScousePy and ACORNS must be installed from GitHub:

```bash
# Install ScousePy
git clone https://github.com/jdhenshaw/scousepy.git
cd scousepy
pip install -e .
cd ..

# Install ACORNS
git clone https://github.com/jdhenshaw/acorns.git
cd acorns
pip install -e .
cd ..
```

> If these libraries are already installed in your environment via other means (e.g., .egg) and can be imported normally, you may skip this step.

## Usage

### 1. Prepare Data

Place your `.fits` files into the `data/inputs/` directory.

### 2. Configure Parameters

Edit `config.py`:

- **`ANALYSIS_MODE`**: Choose `'SINGLE'` (single-file analysis) or `'DUAL'` (dual-file cross-analysis)
- **Input file (single-file mode)**: Set `FITS_DATA` to your filename
- **Input files (dual-file mode)**: Set `STRUCTURE_FITS_PATH` (structure file A) and `PHYSICS_FITS_PATH_DUAL` (physics file B) respectively
- **ACORNS clustering parameters**: Adjust `MIN_HEIGHT_MULTIPLE`, `STOP`, etc. to control structure identification sensitivity
- **Physical parameters**: Modify `DIST_PC` (source distance), `FREQ` (transition frequency), `MOLECULE_NAME`, etc.
- **Beam size**: If `BMAJ/BMIN` is missing from the FITS header, specify manually via `DUAL_B_BEAM_SIZE_ARCSEC`

### 3. Run

```bash
python main.py
```

**Single-file mode** executes: data loading → ScousePy fitting (if `.dat` is missing) → ACORNS clustering → physical property computation → visualization.

**Dual-file mode** executes: ACORNS clustering on structure file A → structure mapping to file B → spectral fitting on file B → multi-scale cross-analysis and visualization.

Results are saved under `data/outputs/` in scale-stratified subdirectories.

### 4. Standalone Examples

```bash
cd examples
python run.py   # Run ACORNS clustering example
python plot.py  # Visualize clustering results
```

## FITS File Requirements

### General
- `CTYPE1`/`CTYPE2`: Recommended `RA---TAN` / `DEC--TAN`
- `CUNIT1`/`CUNIT2`: Recommended `deg`

### 3D Data Cubes (PPV) — Additional
- `CTYPE3`: Velocity axis type (e.g., `VRAD`)
- `CUNIT3`: Recommended `km/s`
- `BUNIT`: Recommended `Jy/beam`

### Beam Information
- `BMAJ`/`BMIN`: FWHM angle (unit: deg). Can be manually specified via config if missing.

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | Numerical computation |
| matplotlib | Plotting |
| pandas | Tabular data |
| astropy | FITS/WCS handling |
| spectral_cube | Data cube operations |
| pyvista | 3D visualization |
| Pillow | Image processing |
| scousepy | Spectral fitting |
| acorns | Hierarchical clustering |

## Known Limitations

- When two FITS files have different pixel scales, WCS coordinate mapping may cause pixel overlap or gaps. This is currently mitigated by sub-pixel sampling (`WCS_MAPPING_SUBPIXEL_DIVISIONS`) and an overlap threshold (`WCS_MAPPING_MIN_OVERLAP`).
- Some structures in file A may have no corresponding emission in file B. This is handled via Spearman correlation matching and a fallback strategy.


---

## Citing

If you use this pipeline in your research, please cite the following software and references:

| Package | Citation |
|---|---|
| **Astropy** | Astropy Collaboration, 2013, A&A, 558, A33; 2018, AJ, 156, 123 |
| **ScousePy** | Henshaw et al., 2016, MNRAS, 457, 2675 |
| **ACORNS** | Henshaw et al., 2019, MNRAS, 485, 2457 |
| **spectral-cube** | https://doi.org/10.5281/zenodo.592982 |
| **PyVista** | Sullivan & Kaszynski, 2019, JOSS, 4, 1450 |
| **NumPy** | Harris et al., 2020, Nature, 585, 357 |
| **SciPy** | Virtanen et al., 2020, Nature Methods, 17, 261 |
| **Matplotlib** | Hunter, 2007, CSE, 9, 90 |
| **Pandas** | McKinney, 2010, Proc. 9th Python in Science Conf. |

## License

MIT License
