# config.py
import numpy as np
import os

# ================= Library Path Configuration =================
# scousepy and acorns should be installed via pip install -e . after cloning from GitHub.
# Once properly installed, they register in Python's site-packages automatically — no manual path needed.
# If you still need to load via .egg or local paths, append them to the list below.
# See README.md for installation instructions.
LIB_PATHS = []

# ================= Pipeline Control Flags =================
# Set to True to run ScousePy interactive fitting.
# Set to False to skip fitting and directly use existing .dat files for clustering.
RUN_SCOUSE_FITTING = True

# ================= Analysis Mode =================
# 'SINGLE': Single-file analysis (original workflow, uses analyzer.py)
# 'DUAL':   Dual-file analysis (structure from File A, physical properties from File B, uses multi_wavelength_analyzer.py)
ANALYSIS_MODE = 'DUAL'



# ================= Plotting Parameters =================
# Histogram bin count settings
# If set to None, automatically calculated based on data size (sqrt(N) * 1.3), constrained between [10, 20]
# If set to an integer, that count is forced

# Nearest neighbor separation histogram (plot_separation_hist)
SEPARATION_HIST_BINS = 15

# Core Mass Function (CMF) histogram (plot_core_mass_function)
CMF_HIST_BINS = 15

# ================= Files & Directories =================
DIR0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')  # Data root directory (project-local data/)
# New folder structure
INPUT_DIR = os.path.join(DIR0, 'inputs')             # Input file directory (.fits)
INTERMEDIATE_DIR = os.path.join(DIR0, 'intermediate/') # Intermediate products (ScousePy output, .npy matrices, etc.)
OUTPUT_DIR = os.path.join(DIR0, 'outputs')           # Final results directory (plots, property CSVs)

HEADER_NAME = 'header.txt'       # FITS header info text file (optional, for specific loaders)
FITS_DATA = 'sdc38p7_a1_850dn.fits'          # Primary FITS file for analysis (Physics/Spectra or Single Mode File)
FITS_NAME = 'sdc38p7_a1_850dn'               # Filename without extension, used in ScousePy settings
GRAPH_TITLE = 'dust intensity peak' # Plot title
GRAPH_NAME = 'dust.pdf'          # Output filename for certain plots (legacy variable)
SAVE_FILENAME = 'dust.acorn'     # ACORNS clustering save filename (Single Mode)
DATA_FILENAME = os.path.join(INTERMEDIATE_DIR, 'best_fit_solutions.dat') # ScousePy fitting result path
PROP_FILENAME = 'property.txt'   # Property computation output file (Single Mode)
LI_CORE_FILE = 'corelist_omc2b.txt' # Li et al. core list file for comparison (optional)

# Manual beam size input (unit: arcsec)
# - None: auto-read BMAJ/BMIN from FITS header (header unit: deg)
# - single value: treated as circular beam (BMAJ=BMIN, unit: arcsec)
# - tuple/list: (BMAJ_arcsec, BMIN_arcsec), each element in arcsec
# Example: 14.0 or (14.0, 12.5)
# SINGLE mode (loader.py uses FITS_DATA)
SINGLE_BEAM_SIZE_ARCSEC = None

# ================= Dual-File Mode Configuration (effective only when ANALYSIS_MODE = 'DUAL') =================
# Structure-defining file (File A, higher resolution / used for structure identification)
STRUCTURE_FITS_PATH = os.path.join(INPUT_DIR, 'sdc38p7_a1_850dn.fits') 
STRUCTURE_FITS_TYPE = 'PP'  # Structure file type: 'PP' (2D) or 'PPV' (3D)
# Physics file (File B, used for spectral fitting and physical property computation)
# If empty, defaults to FITS_DATA defined above.
PHYSICS_FITS_PATH_DUAL = os.path.join(INPUT_DIR, 'sdc38p7_a1_13co_small.fits') 

# DUAL mode: File A beam (unit: arcsec)
# - Currently used mainly for config differentiation; may be extended to A-side physical quantities.
DUAL_A_BEAM_SIZE_ARCSEC = 15

# DUAL mode: File B beam (unit: arcsec)
# - Used by multi_wavelength_analyzer.py for mass/density calculations.
DUAL_B_BEAM_SIZE_ARCSEC = 50

# ================= Input FITS Header Requirements & Unit Notes (DUAL Mode) =================
# The following applies to STRUCTURE_FITS_PATH (A) and PHYSICS_FITS_PATH_DUAL (B).
#
# [Coordinate / WCS: recommended for both A and B]
# 1) CTYPE1/CTYPE2: typical values RA---TAN / DEC--TAN
# 2) CUNIT1/CUNIT2: recommended deg
# 3) CDELT1/CDELT2: angular pixel scale, same unit as CUNIT (commonly deg/pix)
#
# [Velocity axis: required for file B]
# 1) CTYPE3: velocity axis type (e.g., VRAD)
# 2) CUNIT3: recommended km/s
# 3) CDELT3: velocity channel width, same unit as CUNIT3
#
# [Intensity unit: recommended to be explicit for file B]
# 1) BUNIT: recommended Jy/beam (current physical property calculations assume this)
#    - If not Jy/beam, update the brightness temperature conversion logic accordingly.
#
# [Beam: recommended for file B]
# 1) BMAJ/BMIN: FWHM, unit deg (FITS standard)
#    - Purpose: Jy/beam -> K brightness temperature conversion
# 2) If BMAJ/BMIN are missing, specify manually via DUAL_B_BEAM_SIZE_ARCSEC (unit arcsec)
#
# [Area conversion notes]
# 1) Geometric area S uses B-image pixel area (computed from WCS)
# 2) Beam area is used only for brightness temperature conversion, not for pixel geometric area.
#
# [Fallback for missing fields]
# 1) BMAJ/BMIN missing: try DUAL_B_BEAM_SIZE_ARCSEC
# 2) Pixel area computation failure: fall back to CDELT1 approximation
# 3) If all above are missing, mass-related results will have significant uncertainty.

# A->B pixel coverage mapping parameters (robust matching across different pixel scales / resolutions)
# Each A pixel is split into subpixel_divisions^2 sub-pixels for WCS sampling before mapping to B.
# Higher values = more accurate but slower. Recommended range: 3~9
WCS_MAPPING_SUBPIXEL_DIVISIONS = 5

# Minimum overlap threshold: only B pixels with coverage ratio >= this value participate in matching.
# E.g., 0.01 means at least 1% of the B pixel must be covered by the A structure.
WCS_MAPPING_MIN_OVERLAP = 0.01

# When the filling_factor from "structure matching" is below this threshold,
# automatically fall back to "directly use A->B region Scouse results" mode.
# Recommended range: [0, 1].
WCS_MAPPING_FALLBACK_FILLING_FACTOR_THRESHOLD = 0.5

# Spearman matching parameters (A=2D, B=3D)
# True: when multiple candidate B leaf structures exist, preferentially select via Spearman intensity correlation.
SPEARMAN_MATCHING_ENABLED = True

# Spearman minimum valid sample count.
SPEARMAN_MIN_POINTS = 5

# Spearman tie tolerance: |rho_i - rho_best| <= this value is considered a tie; fall back to old tie-break.
SPEARMAN_TIE_TOL = 1.0e-3

# Resolution flag: currently only grid/overlap mapping is performed, no beam matching. Default writes 'not_matched' or 'unknown'.
RESOLUTION_MATCH_FLAG_DEFAULT = 'not_matched'

# Multi-scale structure analysis configuration (DUAL mode)
# Structure hierarchy definition:
#   0 -> largest parent structure (trunk/root, dendrogram color level 0)
#   1 -> secondary structures (level 1)
#   2 -> sub-structures of secondary structures (level 2)
# Multiple scales can be analyzed simultaneously, e.g., [0, 1, 2].
# If set to None, automatically iterates all levels present in the current forest.
ANALYSIS_STRUCTURE_SCALES = None

# Scale directory labels (for output subdirectory naming)
# ANALYSIS_STRUCTURE_SCALE_LABELS = {
#     0: 'scale_0',
#     1: 'scale_1',
#     2: 'scale_2'
# }

# True: strictly select nodes at the target level. If a tree has no nodes at that level, it is skipped at that scale.
# False: if a tree has no nodes at the target level, fall back to the deepest reachable level for that tree.
ANALYSIS_STRUCTURE_SCALE_STRICT = True

# B-anchored facet plot parameters
B_ANCHORED_PLOTS_ENABLED = True
B_ANCHORED_SUBPLOT_ROWS = 3
B_ANCHORED_SUBPLOT_COLS = 3

# A global data contour percentiles (for overlay on B plots)
B_ANCHORED_A_CONTOUR_PERCENTILES = [85, 92, 97]


# ================= Plot Output Filenames =================
PLOT_3D_TREES = '3d_structure_trees.pdf'                        # 3D cluster tree visualization
PLOT_COMPLEX_TREES_3D = '3d_structure_complex_trees.pdf'        # 3D visualization of complex tree structures
PLOT_BRIGHT_TREES_3D = '3d_structure_bright_trees.pdf'          # Dendrogram of bright cores only
PLOT_DENDROGRAM = 'dendrogram_structure.pdf'                    # Dendrogram
PLOT_DENDROGRAM_WITH_INDEX = 'dendrogram_bright_cores_indexed.pdf' # Numbered dendrogram
PLOT_N_COMPONENTS_MAP = 'map_velocity_components.pdf'           # Velocity component count distribution
PLOT_LARGE_STRUCTURES_OUTLINE = 'map_outlines_large_structures.pdf' # Outlines of large-scale structures
PLOT_BRIGHT_STRUCTURES_OUTLINE = 'map_outlines_bright_structures.pdf' # Outlines of bright structures
PLOT_ELLIPSE_CENTERS = 'map_ellipse_centers.pdf'                # Ellipse fitting center distribution
PLOT_JEANS_LENGTH_ANALYSIS = 'analysis_jeans_length_regression.pdf' # Jeans length regression analysis
PLOT_SCALED_SEPARATION_HIST = 'hist_scaled_separation.pdf'      # Scaled separation histogram
PLOT_COMPARISON_TWO_PANEL = 'comparison_density_jeans_separation.pdf' # Density/Jeans/separation comparison
PLOT_SEPARATION_HISTOGRAM = 'hist_nearest_neighbor_separation.pdf' # Nearest-neighbor separation histogram
PLOT_RHO_VS_SEPARATION = 'analysis_density_vs_separation.pdf'   # Density vs. separation analysis
PLOT_CORE_MASS_FUNCTION_PNG = 'analysis_core_mass_function.png' # Core Mass Function (CMF) plot

# ================= Physical Constants =================
MH = 1.67e-24     # Hydrogen atom mass (g)
M0 = 1.67e-24*2.33 # Mean molecular mass (g), assuming mean molecular weight 2.33 (H2 + He)
HP = 6.626e-27    # Planck constant (erg s)
KB = 1.38e-16     # Boltzmann constant (erg/K)
GR = 6.67e-8      # Gravitational constant (cm^3 g^-1 s^-2)
PC = 3.086e18     # Parsec (cm)
AU = 1.496e13     # Astronomical unit (cm)
C = 2.998e10      # Speed of light (cm/s)
YEAR = 86400.0*365 # Seconds in a year
MSUN = 1.99e33    # Solar mass (g)
LSUN = 3.826e33   # Solar luminosity (erg/s)
JY = 1.0e-23      # Jansky (erg s^-1 cm^-2 Hz^-1)
RSUN = 6.96e10    # Solar radius (cm)
TBG = 2.73        # Cosmic microwave background temperature (K)
EV = 1.602e-12    # Electron volt (erg)
DIST_PC = 470     # Source distance (pc); modify per source (e.g., Orion ~400-470)
DISTANCE = 470 * PC # Source distance (cm)
D2S = np.sqrt(8 * np.log(2)) # FWHM to sigma conversion factor (FWHM = sigma * D2S)
DAS = 420 * AU    # Possibly a specific scale or linear beam size (needs verification)
SSB = 1.8047e-5   # Stefan-Boltzmann constant or other constant (needs verification)

# ================= ACORNS Clustering Parameters (Single Mode) =================
PIXEL_SIZE = 1          # Pixel size (unit: pixels)
MIN_RADIUS_PIX = 1     # Points farther apart than this distance cannot be merged into same structure (same unit as PIXEL_SIZE)
MIN_HEIGHT_MULTIPLE = 3 # Threshold multiplier: defines min core height as MIN_HEIGHT_MULTIPLE * RMS (smaller = finer)
VELO_LINK = 0.1         # Data velocity resolution
DV_LINK = 0.2           # If you would also like to link using the LW as an additional criterion
RELAX = np.array([3.0, 2.0, 0.5]) # for interactive set to 0.0 and set interactive = True when calling acorns
STOP = 3.               # Stopping criterion (RMS multiple at which tree growth is cut off)

# ================= ACORNS Clustering Parameters (DUAL Mode - File A structure analysis) =================
A_PIXEL_SIZE = 1
A_MIN_RADIUS_PIX = 1
A_MIN_HEIGHT_MULTIPLE = 1
A_VELO_LINK = 0.1
A_DV_LINK = 0.2
A_RELAX = np.array([3.0, 2.0, 0.5])
A_STOP = 7.

# ================= ACORNS Clustering Parameters (DUAL Mode - File B structure analysis) =================
B_PIXEL_SIZE = 1
B_MIN_RADIUS_PIX = 1.1
B_MIN_HEIGHT_MULTIPLE = 3
B_VELO_LINK = 0.332
B_DV_LINK = 0.664
B_RELAX = np.array([3.0, 2.0, 0.5])
B_STOP = 5.

# ================= Molecular Line Parameters (modify this block to switch molecules) =================
# Used for plot legends / output labels.
MOLECULE_NAME = '13CO'
MOLECULE_TRANSITION = '1-0'
MOLECULE_PLOT_LABEL = f'{MOLECULE_NAME} Cores'

# Line fundamental constants
# Example is 13CO (1-0). Change to target transition parameters when switching molecules.
FREQ = 110.201354e9     # Rest frequency (Hz)
LAMDA = C / FREQ        # Wavelength (cm)
A_UL = 6.335e-8         # Einstein A_ul (s^-1)

# Quantum numbers & statistical weights
# Uses J_u -> J_l notation. For pure rotational lines, statistical weight is typically g_J = 2J + 1.
UPPER_J = 1
LOWER_J = 0
# Legacy variable alias (historical code uses NJ=J_lower)
NJ = LOWER_J
GU = 2 * UPPER_J + 1
GL = 2 * LOWER_J + 1

# Abundance & excitation conditions
X_MOL = 1.5e-6          # Abundance relative to H2 (e.g., 13CO typically 1e-6~2e-6)
TEX = 15                # Excitation temperature (K)

# Line strength corrections
# RI: relative integrated intensity correction for fitted component (typically 1.0 without hyperfine)
# LINE_STRENGTH_RATIO: total line strength / fitted component strength (typically 1.0 without hyperfine)
RI = 1.0
LINE_STRENGTH_RATIO = 1.0

# Partition function configuration
# PARTITION_FUNCTION_MODE:
# - 'interp': interpolate from temperature-Qrot table below (recommended)
# - 'linear_rotor': linear rotor approximation Qrot ~= kT/(hB) + 1/3
PARTITION_FUNCTION_MODE = 'interp'

# Only used when PARTITION_FUNCTION_MODE='interp'.
# A usable set of approximate table values for 13CO (can be replaced with literature/CDMS/LAMDA data).
PARTITION_FUNCTION_TEMPS = [9.0, 18.0, 37.0, 75.0]
PARTITION_FUNCTION_QROTS = [3.89, 7.43, 14.52, 28.7]

# Only used when PARTITION_FUNCTION_MODE='linear_rotor'.
# If None, B is auto-estimated from FREQ and UPPER_J.
ROTATIONAL_CONSTANT_HZ = None

# Kinematic constants
CS = 0.23               # Isothermal sound speed (km/s)
MU = 2.33               # Mean molecular weight, used for mass calculations
P_IC = 2e7 * KB         # External pressure (erg cm^-3)