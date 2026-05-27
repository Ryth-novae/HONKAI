# src/processor.py
import numpy as np
import sys
import os
import config as cfg
from src import utils

# Try to import libraries
try:
    from acorns import Acorns
except ImportError:
    print("Warning: 'acorns' library not found.")

try:
    from scousepy import scouse
    from scousepy.io import output_ascii_indiv
except ImportError:
    print("Warning: 'scousepy' library not found.")

@utils.trace_error
def run_scouse_fitting():
    """
    Run ScousePy semi-automatic fitting process (Stage 1 - Stage 4).
    Note: This is an interactive process and will pop up a GUI window.
    
    Restores logic from lines 186-194 of the original code.
    """
    print("\n>>> Starting SCOUSEPY fitting process...")
    print("    NOTE: This requires GUI interaction.")

    # 1. Run setup (generate config files)
    # filename is the FITS filename, datadir is the directory
    config_file = scouse.run_setup(cfg.FITS_NAME, cfg.INPUT_DIR, outputdir=cfg.INTERMEDIATE_DIR)
    
    # 2. Stage 1: Identify spectral coverage
    s = scouse.stage_1(config=config_file, interactive=True)

    # 3. Stage 2: Fit average spectrum (SAA)
    s = scouse.stage_2(config=config_file)

    # 4. Stage 3: Auto-fit all spectra
    s = scouse.stage_3(config=config_file)

    # 5. Stage 4: Quality control and inspection
    s = scouse.stage_4(config=config_file, bitesize=True)

    # 6. Export results
    # Export fitting results as ASCII .dat for ACORNS
    print(f"Saving SCOUSEPY results to {cfg.INTERMEDIATE_DIR}...")
    output_ascii_indiv(s, cfg.INTERMEDIATE_DIR)

    print("SCOUSEPY fitting completed.")

@utils.trace_error
def run_acorns_clustering():
    """
    Load scousepy fitting results and run/load ACORNS clustering.
    """
    print("Preparing ACORNS clustering data...")

    # Check if .dat file exists
    if not os.path.exists(cfg.DATA_FILENAME):
        if cfg.RUN_SCOUSE_FITTING:
            print("Data file not found, but RUN_SCOUSE_FITTING is True. Proceeding...")
        else:
            raise FileNotFoundError(f"Best fit data '{cfg.DATA_FILENAME}' not found. Please set RUN_SCOUSE_FITTING = True in config.py to generate it first.")

    # 1. Load .dat file (ScousePy output)
    dataarr_raw = np.loadtxt(cfg.DATA_FILENAME, skiprows=1)

    # 2. Data preprocessing
    # Extract specific columns: x, y, amp, err_amp, shift, width, rms
    dataarr0 = np.array(dataarr_raw[:, np.array([1, 2, 3, 4, 5, 7, 9])]).T
    dataarr_acorns_process = np.array(dataarr0[np.array([0, 1, 2, 3, 4, 5]), :])

    # Full transpose
    dataarr_acorns = np.array(dataarr_raw).T

    # Compute clustering parameters
    min_height = cfg.MIN_HEIGHT_MULTIPLE * np.mean(dataarr_acorns[9, :])
    min_radius = cfg.MIN_RADIUS_PIX
    cluster_criteria = np.array([min_radius, cfg.VELO_LINK, cfg.DV_LINK])
    relax = cfg.RELAX
    stop = cfg.STOP

    # 3. Run or load ACORNS
    save_path = os.path.join(cfg.INTERMEDIATE_DIR, cfg.SAVE_FILENAME)

    # If ACORNS file does not exist, or forced rerun, then run process
    # Default: load if file exists, unless .acorn is manually deleted
    if os.path.exists(save_path):
        print(f"Loading existing ACORNS forest from {save_path}...")
        A = Acorns.load_from(save_path)
    else:
        print("Existing ACORNS tree not found. Running Acorns.process (this may take time)...")
        # Note: this requires dataarr_acorns_process (6-row data)
        A = Acorns.process(dataarr_acorns_process, cluster_criteria, method="PPV",
                           min_height=min_height, pixel_size=cfg.PIXEL_SIZE,
                           relax=relax, stop=stop)
        A.save_to(save_path)

    return A, dataarr_acorns, dataarr_raw