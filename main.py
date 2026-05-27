# main.py
import sys
import os
import matplotlib
import numpy as np
# ==========================================
# Note: Matplotlib Backend Configuration
# ==========================================
# If best_fit_solutions.dat does not exist, the program will launch the ScousePy interactive GUI.
# The interactive interface requires a graphical backend (e.g., TkAgg, Qt5Agg); 'Agg' will not work.
# If you are certain the file exists and only need to generate plots headlessly, uncomment the line below:
# matplotlib.use('Agg')

import config as cfg

# ==========================================
# 1. Configure environment paths
# If scousepy/acorns are installed via pip install -e ., LIB_PATHS is empty and this block is skipped.
# If you still need to manually specify .egg or local paths, append them in config.py LIB_PATHS.
# ==========================================
for lib in cfg.LIB_PATHS:
    sys.path.append(lib)

# ==========================================
# 2. Import submodules
# ==========================================
from src import loader, processor, analyzer, visualizer, utils
from src.multi_wavelength_analyzer import MultiWavelengthAnalyzer

@utils.trace_error
def main():
    print("\n" + "="*50)
    print("       N2H+ Pipeline Processing Started       ")
    print("="*50 + "\n")
    
    # Ensure directories exist
    for d in [cfg.INPUT_DIR, cfg.INTERMEDIATE_DIR, cfg.OUTPUT_DIR]:
        if not os.path.exists(d):
            print(f"Creating directory: {d}")
            os.makedirs(d, exist_ok=True)

    # ================= Check Analysis Mode =================
    if cfg.ANALYSIS_MODE == 'DUAL':
        print(f"Mode: DUAL File Analysis")
        print(f"This mode uses structural info from one FITS to analyze spectra in another.")
        
        struct_path = cfg.STRUCTURE_FITS_PATH
        phys_path = cfg.PHYSICS_FITS_PATH_DUAL
        if phys_path is None:
            phys_path = os.path.join(cfg.INPUT_DIR, cfg.FITS_DATA)
            
        if not os.path.exists(struct_path):
             print(f"Error: Structure file not found: {struct_path}")
             return
             
        if not os.path.exists(phys_path):
             print(f"Error: Physics file not found: {phys_path}")
             return

        # Initialize Analyzer
        mw_analyzer = MultiWavelengthAnalyzer(
            structure_fits_path=struct_path,
            output_dir=cfg.OUTPUT_DIR,
            intermediate_dir=cfg.INTERMEDIATE_DIR,
            physics_fits_path=phys_path,
            structure_pixel_size=cfg.A_PIXEL_SIZE,
            structure_min_radius_pix=cfg.A_MIN_RADIUS_PIX,
            structure_min_height_multiple=cfg.A_MIN_HEIGHT_MULTIPLE,
            structure_velo_link=cfg.A_VELO_LINK,
            structure_dv_link=cfg.A_DV_LINK,
            structure_relax=cfg.A_RELAX,
            structure_stop=cfg.A_STOP,
            physics_pixel_size=cfg.B_PIXEL_SIZE,
            physics_min_radius_pix=cfg.B_MIN_RADIUS_PIX,
            physics_min_height_multiple=cfg.B_MIN_HEIGHT_MULTIPLE,
            physics_velo_link=cfg.B_VELO_LINK,
            physics_dv_link=cfg.B_DV_LINK,
            physics_relax=cfg.B_RELAX,
            physics_stop=cfg.B_STOP
        )
        
        # 1. Structure Analysis (ACORNS on File A)
        # Supports 'PP' (2D) and 'PPV' (3D) modes. 
        # In 'PPV' mode, this step includes ScousePy fitting on File A.
        mw_analyzer.run_structure_analysis(
            method=cfg.STRUCTURE_FITS_TYPE, 
            save_name='structure_A.acorn'
        )
        
        # 2. Spectral Fitting (ScousePy on File B masked by A)
        # This maps structure A to B, generating a mask, and runs ScousePy on B
        dat_path = mw_analyzer.run_scouse_on_structures()
        
        # 3/4. Multi-scale analysis and visualization
        scales_raw = getattr(cfg, 'ANALYSIS_STRUCTURE_SCALES', [0])
        if scales_raw is None:
            scales = mw_analyzer.get_available_structure_scale_levels()
            if len(scales) == 0:
                scales = [0]
            print(f"ANALYSIS_STRUCTURE_SCALES=None -> using all available levels: {scales}")
        else:
            if np.isscalar(scales_raw):
                scales_raw = [scales_raw]

            scales = []
            for s in scales_raw:
                try:
                    s_int = int(s)
                except Exception:
                    continue
                if s_int < 0:
                    continue
                if s_int not in scales:
                    scales.append(s_int)
            if len(scales) == 0:
                scales = [0]

        label_map = getattr(cfg, 'ANALYSIS_STRUCTURE_SCALE_LABELS', {})
        base_output_dir = cfg.OUTPUT_DIR

        for scale_level in scales:
            label = label_map.get(scale_level, f'level{scale_level}')
            scale_output_dir = os.path.join(base_output_dir, f'scale_{scale_level}_{label}')
            os.makedirs(scale_output_dir, exist_ok=True)

            selected_nodes = mw_analyzer.get_structure_nodes_for_scale(scale_level)
            print(
                f"\n>>> Scale analysis: level={scale_level} label={label} "
                f"nodes={len(selected_nodes)} output={scale_output_dir}"
            )

            mw_analyzer.output_dir = scale_output_dir
            df_results = mw_analyzer.map_and_analyze(
                selected_nodes=selected_nodes,
                scale_level=scale_level
            )
            mw_analyzer.plot_structures(
                selected_nodes=selected_nodes,
                scale_level=scale_level,
                run_plot6=False
            )

        # 5. Always-on extra groups
        extra_groups = [
            ('leaves_all', 'scale_leaves_all'),
            ('roots_nonleaf', 'scale_roots_nonleaf')
        ]

        for group_name, group_dirname in extra_groups:
            group_output_dir = os.path.join(base_output_dir, group_dirname)
            os.makedirs(group_output_dir, exist_ok=True)

            selected_nodes = mw_analyzer.get_structure_nodes_for_group(group_name)
            print(
                f"\n>>> Extra-group analysis: group={group_name} "
                f"nodes={len(selected_nodes)} output={group_output_dir}"
            )
            if len(selected_nodes) == 0:
                print(f"Skip group={group_name}: no nodes selected.")
                continue

            mw_analyzer.output_dir = group_output_dir
            mw_analyzer.map_and_analyze(
                selected_nodes=selected_nodes,
                scale_level=0
            )
            mw_analyzer.plot_structures(
                selected_nodes=selected_nodes,
                scale_level=0,
                run_plot6=(group_name == 'leaves_all')
            )

        # Restore base output directory on analyzer object
        mw_analyzer.output_dir = base_output_dir
        
        print("\n>>> Analysis Complete. Multi-scale results saved to output subdirectories.")
        return
    else:
        print(f"Mode: SINGLE File Analysis (Standard Pipeline)")

    # ------------------------------------------------------
    # Step 1: Data Loading & Preprocessing
    # ------------------------------------------------------
    print(">>> Step 1: Loading Data...")
    # scu1: SpectralCube object, intensity: 2D intensity map, grid: WCS object, meta_data: dict containing dx, dy, beam, etc.
    scu1, intensity, grid, meta_data = loader.load_data()
    
    # ------------------------------------------------------
    # Step 1.5: ScousePy Fitting
    # ------------------------------------------------------
    # Logic: check if the data file exists. If not, force fitting.
    if not os.path.exists(cfg.DATA_FILENAME):
        print(f"\n[!] Missing data file: {cfg.DATA_FILENAME}")
        print(">>> Step 1.5: Initiating SCOUSEPY Fitting (Interactive Mode)...")

        # Check if the current backend supports interactive mode (simple guard)
        backend = matplotlib.get_backend()
        if 'Agg' in backend:
            print(f"WARNING: Current matplotlib backend is '{backend}', which implies non-interactive mode.")
            print("         If the GUI does not appear, please comment out 'matplotlib.use(\"Agg\")' in main.py.")

        # Run fitting (generates best_fit_solutions.dat)
        processor.run_scouse_fitting()
    else:
        print(f"\n>>> Step 1.5: Found {cfg.DATA_FILENAME}")
        print("          Skipping fitting process. Using existing data.")
        
    # ------------------------------------------------------
    # Step 2: Run/Load ACORNS Clustering Results
    # ------------------------------------------------------
    print("\n>>> Step 2: Running/Loading ACORNS Clustering...")
    # Note: processor.py needs to return dataarr_raw (raw txt data) for plotting Ncomps
    # A: Forest object
    # dataarr_acorns: processed transposed array (for computation)
    # dataarr_raw: raw loaded array (for Ncomps plotting)
    try:
        A, dataarr_acorns, dataarr_raw = processor.run_acorns_clustering()
    except ValueError:
        # If processor.py hasn't been updated to return 3 values, try the old version
        print("Warning: processor.run_acorns_clustering returned 2 values. 'dataarr_raw' missing.")
        # Reload raw data for plotting
        dataarr_raw = np.loadtxt(cfg.DATA_FILENAME, skiprows=1)
        ret = processor.run_acorns_clustering()
        A = ret[0]
        dataarr_acorns = ret[1]

    # ------------------------------------------------------
    # Step 3: Basic Structure Visualization
    # ------------------------------------------------------
    print("\n>>> Step 3: Visualizing Basic Structures...")
    
    # 1. 3D plot of bright core structures
    visualizer.plot_bright_trees_3d(A, dataarr_acorns, meta_data)

    # 2. Dendrogram with indices for bright cores
    visualizer.plot_dendrogram_with_index(A)
    
    # 3. 2D velocity component count map (Ncomps)
    visualizer.plot_ncomps_2d(intensity, grid, dataarr_raw)
    
    # 4. Structure outline contours
    visualizer.plot_outlines(intensity, grid, A, dataarr_acorns, meta_data['scu0_shape'])
    
    # ------------------------------------------------------
    # Step 4: Compute Physical Properties & Error Analysis
    # ------------------------------------------------------
    print("\n>>> Step 4: Calculating Core Properties & Error Analysis...")
    
    # Compute core physical properties (mass, density, Jeans length, etc.) and ellipse fitting
    # df_props: result DataFrame
    # leave_num: valid leaf index list
    # ellipse_centers: fitted center coordinates (pixel)
    df_props, leave_num, ellipse_centers = analyzer.calculate_core_properties(
        A, dataarr_acorns, scu1, meta_data
    )
    
    # Identify sources with large errors (via neighbor density comparison)
    # big_err_idx: list of source indices with large errors
    big_err_idx, _ = analyzer.identify_big_error_sources(df_props, ellipse_centers, meta_data)
    
    # Plot marked center map (highlighting large-error sources)
    visualizer.plot_ellipse_centers(
        intensity, grid, ellipse_centers, big_err_idx, 
        A, dataarr_acorns, leave_num, meta_data['scu0_shape']
    )
    
    # ------------------------------------------------------
    # Step 5: Statistical Analysis (Jeans Length)
    # ------------------------------------------------------
    print("\n>>> Step 5: Performing Statistical Analysis...")
    
    # Plot Jeans length analysis (with linear regression and outlier removal)
    visualizer.plot_jeans_analysis(df_props)
    
    visualizer.plot_separation_hist(df_props)

    visualizer.plot_rho_vs_separation_standalone(df_props)
    
    # Plot Core Mass Function (CMF)
    visualizer.plot_core_mass_function(df_props)

    print("\n" + "="*50)
    print("       Pipeline Completed Successfully.       ")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()