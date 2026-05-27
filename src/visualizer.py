# src/visualizer.py
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import config as cfg
from src import utils
from matplotlib.patches import Ellipse
try:
    import cv2
except ImportError:
    cv2 = None

@utils.trace_error
def plot_bright_trees_3d(A, dataarr_acorns, meta_data):
    """
    Plot only bright tree structures with significant height variation (>0.24)
    Replaces original plot_3d_trees and plot_complex_trees_3d
    """
    print("Plotting bright trees 3D...")
    dx, dy = meta_data['dx'], meta_data['dy']
    xmin, ymin = meta_data['xmin'], meta_data['ymin']
    velo_axis = meta_data['velo_axis']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # -------------------------------------------------------------------------
    # Fix issue where RA/DEC scale is compressed
    # -------------------------------------------------------------------------
    # Calculate actual coordinate range
    raw_x = dataarr_acorns[1, :] * dx + xmin
    raw_y = dataarr_acorns[2, :] * dy + ymin
    x_span = np.ptp(raw_x)
    y_span = np.ptp(raw_y)
    
    # Set 3D Box visual aspect ratio
    # Z-axis (Velocity) visual height set to average of X/Y spans to avoid being too flat or too tall
    if x_span > 0 and y_span > 0:
        # matplotlib 3.3+ supports set_box_aspect
        try:
            ax.set_box_aspect((x_span, y_span, (x_span + y_span)*0.5))
        except AttributeError:
            # Legacy version compatibility (may not be as effective)
            pass
    # -------------------------------------------------------------------------

    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('DEC [deg]')
    ax.set_zlabel('Velocity [km/s]')
    if len(velo_axis) > 0:
        ax.set_zlim(np.min(velo_axis).value, np.max(velo_axis).value)

    # Background
    ax.scatter(dataarr_acorns[1, :]*dx+xmin, dataarr_acorns[2, :]*dy+ymin,
               dataarr_acorns[5, :], marker='o', s=1., c='black', linewidth=0., alpha=0.2)

    # Count qualifying trees
    n = 0
    for tree in A.forest:
        if np.max(A.forest[tree].cluster_vertices[1]) > 0.24:
            n += 1

    colour = iter(cm.rainbow(np.linspace(0, 1, n if n>0 else 1)))

    for tree in A.forest:
        is_complex = np.max(A.forest[tree].cluster_vertices[1]) > 0.24
        
        if A.forest[tree].trunk.leaf_cluster:
            members = A.forest[tree].trunk.cluster_members
            if is_complex:
                c = next(colour)
                ax.scatter(dataarr_acorns[1, members]*dx+xmin, dataarr_acorns[2, members]*dy+ymin, 
                           dataarr_acorns[5, members], marker='o', s=2, c='None', edgecolors=c, alpha=0.6)
            else:
                # Simple trees in black
                ax.scatter(dataarr_acorns[1, members]*dx+xmin, dataarr_acorns[2, members]*dy+ymin, 
                           dataarr_acorns[5, members], marker='.', s=0.01, c='None', edgecolors='k', alpha=0.01)
        else:
            if is_complex:
                c = next(colour)
                for blade in A.forest[tree].leaves:
                    members = blade.cluster_members
                    ax.scatter(dataarr_acorns[1, members]*dx+xmin, dataarr_acorns[2, members]*dy+ymin, 
                               dataarr_acorns[5, members], marker='o', s=2, c='None', edgecolors=c, alpha=0.6)
            else:
                for blade in A.forest[tree].leaves:
                    members = blade.cluster_members
                    ax.scatter(dataarr_acorns[1, members]*dx+xmin, dataarr_acorns[2, members]*dy+ymin, 
                               dataarr_acorns[5, members], marker='.', s=0.01, c='None', edgecolors='k', alpha=0.01)

    plt.tight_layout()
    
    # Smart check: if backend is non-interactive (Agg) or display fails, auto-save
    backend = plt.get_backend().lower()
    if backend == 'agg':  # Strict match 'agg' to avoid false matches on interactive backends
        print(f"Non-interactive backend ({backend}) detected. Saving figure instead.")
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_BRIGHT_TREES_3D))
        plt.close()
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Interactive display failed ({e}). Saving figure to disk.")
            plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_BRIGHT_TREES_3D))
            plt.close()



@utils.trace_error
def plot_dendrogram_with_index(A):
    """
    Dendrogram of Bright Cores with indices
    Highlight trees with height > 0.24 in color with indices; others in black.
    """
    print("Plotting dendrogram with indices...")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Sv [Jy Beam^-1]')

    # Count Bright Cores for color assignment
    bright_trees = []
    for tree in A.forest:
        if np.max(A.forest[tree].cluster_vertices[1]) > 0.24:
            bright_trees.append(tree)
    
    # Generate colors
    colour = iter(cm.rainbow(np.linspace(0, 1, len(bright_trees))))
    
    count = 0.0
    for tree in A.forest:
        is_bright = False
        if np.max(A.forest[tree].cluster_vertices[1]) > 0.24:
            is_bright = True
            c = next(colour)
        else:
            c = 'k' # Black
        
        # Record tree top for labeling
        max_y = -np.inf
        max_x = 0
        
        # Draw vertical line
        for j in range(len(A.forest[tree].cluster_vertices[0])):
            x_vals = A.forest[tree].cluster_vertices[0][j] + count
            y_vals = A.forest[tree].cluster_vertices[1][j]
            ax.plot(x_vals, y_vals, c=c)
            
            # Update max for label
            if len(y_vals) > 0:
                local_max_idx = np.argmax(y_vals)
                if y_vals[local_max_idx] > max_y:
                    max_y = y_vals[local_max_idx]
                    max_x = x_vals[local_max_idx]
        
        # Draw horizontal line
        for j in range(len(A.forest[tree].horizontals[0])):
            x_vals = A.forest[tree].horizontals[0][j] + count
            y_vals = A.forest[tree].horizontals[1][j]
            ax.plot(x_vals, y_vals, c=c)
            
        # Add index labels (Bright Cores only)
        if is_bright and max_y > -np.inf:
            # Label above tree top
            ax.text(max_x, max_y, f'{tree}', ha='center', va='bottom', fontsize=8, color=c)
            
        count += len(A.forest[tree].leaves)
        
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_DENDROGRAM_WITH_INDEX), bbox_inches='tight', dpi=500, format="PDF")
    plt.close()

@utils.trace_error
def plot_ncomps_2d(intensity, grid, dataarr_raw):
    """
    Plot velocity component distribution map
    Corresponds to original lines 367-425 (ncomp.pdf)
    """
    print("Plotting Ncomps 2D map...")
    fig = plt.figure(figsize=(12, 12))
    # Use WCS projection
    ax1 = fig.add_subplot(111, projection=grid)
    ax1.imshow(intensity.value, cmap='Greys')
    
    # dataarr_raw is the raw array from loadtxt
    # Column 0 is ncomps (1,2,3,4...)
    # Column 1 is x, Column 2 is y
    
    # Ensure dataarr_raw is a numpy array
    data = np.array(dataarr_raw)
    ncomps = data[:, 0]
    xs = data[:, 1]
    ys = data[:, 2]
    
    colors_map = {1: 'green', 2: 'yellow', 3: 'cyan', 4: 'red'}
    
    for n_comp, color in colors_map.items():
        mask = (ncomps == n_comp)
        if np.sum(mask) > 0:
            # Plot scatter
            ax1.scatter(xs[mask], ys[mask], c=color, s=5, label=f'{n_comp} comps', alpha=1)
            
    # Set axes
    lon = ax1.coords[0]
    lat = ax1.coords[1]
    lon.set_axislabel('RA (J2000)', minpad=0.7)
    lat.set_axislabel('DEC (J2000)', minpad=0.7)
    
    # Simple legend (via dummy points)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=f'{n} comps') 
                       for n, c in colors_map.items()]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_N_COMPONENTS_MAP), bbox_inches='tight', dpi=500)
    plt.close()

@utils.trace_error
def plot_outlines(intensity, grid, A, dataarr_acorns, scu0_shape):
    """
    Plot structure outlines.
    Fully restore original lines 429-509 logic, generating two figures:
    1. c10_outlines.pdf: Large structures with area > 10 pixels
    2. c_outlines.pdf: Bright structures with height > 0.24
    """
    print("Plotting outlines (Large > 10px & Bright > 0.24)...")
    
    vector = np.vectorize(np.int_)
    field_shape = scu0_shape[1:] # shape (y, x)
    
    # ==========================================
    # 1. Plot large-scale structures (Area > 10 pixels)
    # Corresponds to original lines 429-473
    # ==========================================
    # fig1 = plt.figure(figsize=(12, 12))
    # ax1 = fig1.add_subplot(111, projection=grid)
    # ax1.imshow(intensity.value, cmap='Greys')
    
    # Count qualifying structures for colors
    # n_large = 0
    # for tree in A.forest:
    #     for blade in A.forest[tree].leaves:
    #         if len(blade.cluster_members) > 10:
    #             n_large += 1
    
    # colour_large = iter(cm.rainbow(np.linspace(0, 1, n_large if n_large>0 else 1)))
    
    # for tree in A.forest:
    #     for blade in A.forest[tree].leaves:
    #         if len(blade.cluster_members) > 10:
    #             co = next(colour_large)
    #             mask = np.zeros(field_shape)
    # Note coordinate indexing: dataarr[2]=y, dataarr[1]=x
    #             mask[vector(dataarr_acorns[2, blade.cluster_members]), 
    #                  vector(dataarr_acorns[1, blade.cluster_members])] = 1
                
    #             x1, y1 = utils.edge_fun(mask)
    #             ax1.scatter(x1, y1, color=co, marker='.', s=1)

    # lon = ax1.coords[0]
    # lat = ax1.coords[1]
    # lon.set_axislabel('RA (J2000)', minpad=0.7)
    # lat.set_axislabel('DEC (J2000)', minpad=0.7)
    
    # plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_LARGE_STRUCTURES_OUTLINE), bbox_inches='tight', dpi=500)
    # plt.close(fig1)

    # ==========================================
    # 2. Plot bright structures (Peak > 0.24)
    # Corresponds to original lines 476-505
    # ==========================================
    fig2 = plt.figure(figsize=(12, 12))
    ax2 = fig2.add_subplot(111, projection=grid)
    ax2.imshow(intensity.value, cmap='Greys')
    
    # Count qualifying structures
    n_bright = 0
    for tree in A.forest:
        if np.max(A.forest[tree].cluster_vertices[1]) > 0.24:
            n_bright += 1
    
    colour_bright = iter(cm.rainbow(np.linspace(0, 1, n_bright if n_bright>0 else 1)))
    
    for tree in A.forest:
        if np.max(A.forest[tree].cluster_vertices[1]) > 0.24:
            co = next(colour_bright)
            for blade in A.forest[tree].leaves:
                mask = np.zeros(field_shape)
                mask[vector(dataarr_acorns[2, blade.cluster_members]), 
                     vector(dataarr_acorns[1, blade.cluster_members])] = 1
                
                x1, y1 = utils.edge_fun(mask)
                ax2.scatter(x1, y1, color=co, marker='.', s=1)
                
    lon = ax2.coords[0]
    lat = ax2.coords[1]
    lon.set_axislabel('RA (J2000)', minpad=0.7)
    lat.set_axislabel('DEC (J2000)', minpad=0.7)

    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_BRIGHT_STRUCTURES_OUTLINE), bbox_inches='tight', dpi=500)
    plt.close(fig2)

@utils.trace_error
def plot_ellipse_centers(intensity, grid, ellipse_centers, big_err_idx, A, dataarr_acorns, leave_num, scu0_shape):
    """
    Plot ellipse centers, mark large error sources, and draw core outlines
    """
    print("Plotting ellipse centers, outlines and error flags...")
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(111, projection=grid)
    ax1.imshow(intensity.value, cmap='Greys')
    
    # Prepare outlines
    vector = np.vectorize(np.int_)
    field_shape = scu0_shape[1:]
    
    # Plot outlines in leave_num order (matching ellipse_centers)
    # Color assignment matches plot_outlines (bright structures): one color per tree
    n_bright_trees = len(leave_num)
    colour_bright = iter(cm.rainbow(np.linspace(0, 1, n_bright_trees if n_bright_trees > 0 else 1)))
    
    for tree_idx in leave_num:
        # Assign a color to each tree
        co = next(colour_bright)
        
        # Iterate over all leaves in this tree
        for blade in A.forest[tree_idx].leaves:
            mask = np.zeros(field_shape)
            mask[vector(dataarr_acorns[2, blade.cluster_members]), 
                 vector(dataarr_acorns[1, blade.cluster_members])] = 1
            
            x1, y1 = utils.edge_fun(mask)
            # Draw outline with per-tree color
            ax1.scatter(x1, y1, color=co, marker='.', s=0.5, alpha=0.8)

            # Fit ellipse and draw outline (using cv2.fitEllipse)
            if cv2 is None:
                continue
            if len(x1) < 5 or len(y1) < 5:
                continue
            try:
                pts = np.vstack([x1, y1]).T.astype(np.float32)
                (xc, yc), (a_fit, b_fit), theta = cv2.fitEllipse(pts)
                a_axis, b_axis = max(a_fit, b_fit), min(a_fit, b_fit)
                e = Ellipse(
                    (xc, yc),
                    width=a_axis,
                    height=b_axis,
                    angle=theta,
                    fill=False,
                    edgecolor=co,
                    linewidth=0.8,
                    transform=ax1.get_transform(grid)
                )
                ax1.add_patch(e)
            except Exception:
                continue

    centers = np.array(ellipse_centers)
    if len(centers) > 0:
        # Plot all centers with index labels
        for i, center in enumerate(centers):
            ax1.scatter(center[1], center[0], marker='x', color='blue', s=10, linewidths=0.5)  # Blue markersX
            # Label all point indices
            ax1.text(center[1], center[0], f'{i}', color='blue', fontsize=4, ha='right', va='top')
        
        # Mark large error sources (more prominent red markers)
        for idx in big_err_idx:
            if idx < len(centers):
                ax1.scatter(centers[idx, 1], centers[idx, 0], marker='X', color='r', s=20)  # Red for distinction
                ax1.text(centers[idx, 1], centers[idx, 0], f'ERR<{idx}>', color='r', fontsize=6, ha='left', va='bottom')
            
    lon = ax1.coords[0]
    lat = ax1.coords[1]
    lon.set_axislabel('RA (J2000)')
    lat.set_axislabel('DEC (J2000)')

    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_ELLIPSE_CENTERS), bbox_inches='tight', dpi=500)
    plt.close()
@utils.trace_error

def plot_jeans_analysis(df):
    """
    Plot Jeans length analysis with linear regression and outlier rejection
    Corresponds to original lines 1093-1163
    """
    print("Plotting Jeans analysis with regression...")
    fig, ax = plt.subplots()
    
    x = df['closest_dist_pc'].values
    y = df['L_J_pc'].values
    y_cy = df['L_J_cy_pc'].values

    # Determine errors
    try:
        y_err_u = df['L_J_pc_err_u'].values - y
        y_err_l = y - df['L_J_pc_err_l'].values
        y_err_cy_u = df['L_J_cy_pc_err_u'].values - y_cy
        y_err_cy_l = y_cy - df['L_J_cy_pc_err_l'].values
        
        y_err = [np.maximum(0, y_err_l), np.maximum(0, y_err_u)]
        y_err_cy = [np.maximum(0, y_err_cy_l), np.maximum(0, y_err_cy_u)]
        
        has_error = True
    except KeyError:
        has_error = False
        y_err = None
        y_err_cy = None
    
    # Plot base scatter
    if has_error:
        ax.errorbar(x, y, yerr=y_err, fmt='.', color='b', ecolor='b', alpha=0.4, label='Homogeneous (Data)')
        ax.errorbar(x, y_cy, yerr=y_err_cy, fmt='.', color='r', ecolor='r', alpha=0.2, label='Cylinder (Data)')
    else:
        ax.scatter(x, y, label='Homogeneous (Data)', marker='.', color='b', alpha=0.4)
        ax.scatter(x, y_cy, label='Cylinder (Data)', marker='.', color='r', alpha=0.1)
    
    # Proportional function: y = k * x (intercept=0)
    def proportional_func(x, k):
        return k * x

    # --- Restore regression logic ---
    if len(x) > 2:
        # Define arrays for storing fit results
        datasets = [
            {'x': x, 'y': y, 'color': 'b', 'name': 'Homo'},
            {'x': x, 'y': y_cy, 'color': 'r', 'name': 'Cyl'}
        ]

        x_line = np.linspace(0, np.max(x), 100)

        for ds in datasets:
            x_data = ds['x']
            y_data = ds['y']
            
            # Initial fit
            try:
                popt, pcov = curve_fit(proportional_func, x_data, y_data)
                
                # Iteration to remove outliers
                num_iterations = 2
                tolerant = 3
                mask = np.ones_like(x_data, dtype=bool)
                
                slope = popt[0]
                perr = np.sqrt(np.diag(pcov))[0]

                for i in range(num_iterations):
                    fitted_vals = proportional_func(x_data, slope)
                    residuals = y_data - fitted_vals
                    std_dev = np.std(residuals)
                    
                    if std_dev > 0:
                        mask = np.abs(residuals) < (tolerant - i) * std_dev
                        if np.sum(mask) > 2:
                             popt, pcov = curve_fit(proportional_func, x_data[mask], y_data[mask])
                             slope = popt[0]
                             perr = np.sqrt(np.diag(pcov))[0]
                
                # Plot Fit line
                y_line = proportional_func(x_line, slope)
                ax.plot(x_line, y_line, color=ds['color'], linestyle='--', 
                        label=f"{ds['name']} Fit: slope={slope:.2f}")

                # Plot Confidence Interval (3-sigma)
                if perr > 0:
                    y_upper = proportional_func(x_line, slope + 3*perr)
                    y_lower = proportional_func(x_line, slope - 3*perr)
                    ax.fill_between(x_line, y_lower, y_upper, color=ds['color'], alpha=0.1)

            except Exception as e:
                print(f"Fitting failed for {ds['name']}: {e}")

    # Reference line y=x
    mx = np.max(x) if len(x) > 0 else 0.1
    ax.plot([0, mx], [0, mx], 'grey', linestyle='-', label='Theory=Obs (Slope=1)')
    
    ax.set_xlabel('Closest Distance (pc)')
    ax.set_ylabel('Jeans Length (pc)')

    # Reorder legend to put Data first
    handles, labels = ax.get_legend_handles_labels()
    display_order = []
    other_indices = []
    for i, label in enumerate(labels):
        if '(Data)' in label:
             display_order.append(i)
        else:
             other_indices.append(i)
    display_order.extend(other_indices)
    
    ordered_handles = [handles[i] for i in display_order]
    ordered_labels = [labels[i] for i in display_order]
    
    ax.legend(ordered_handles, ordered_labels, fontsize=8, loc='upper right')
    ax.set_ylim(0, 0.15)
    
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_JEANS_LENGTH_ANALYSIS), bbox_inches='tight', dpi=500)
    plt.close()




@utils.trace_error

def plot_separation_hist(df_props):
    """
    Plot nearest neighbor separation histogram
    Corresponds to original lines 1166-1180
    Output: separate_hist.pdf
    """
    if df_props is None or df_props.empty: return
    print("Plotting separation histogram...")
    
    fig, ax = plt.subplots()
    
    # Extract distance data (pc)
    dists = df_props['closest_dist_pc'].values
    # Filter zeros
    dists = dists[dists > 0]
    
    # Dynamically compute bins
    if cfg.SEPARATION_HIST_BINS is not None:
        num_bins = cfg.SEPARATION_HIST_BINS
    else:
        num_bins = int(1.3 * np.sqrt(len(dists)))  # Slightly more than sqrt(N)
        if num_bins < 10: num_bins = 10
        if num_bins > 20: num_bins = 20
    
    # Plot histogram
    ax.hist(dists, bins=num_bins, color='grey', alpha=0.5, edgecolor='black')
    
    ax.set_xlabel('Separation (pc)')
    ax.set_ylabel('Number')
    ax.legend()
    
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_SEPARATION_HISTOGRAM), bbox_inches='tight', dpi=500)
    plt.close()
@utils.trace_error

def plot_rho_vs_separation_standalone(df_props):
    """
    Plot density vs. separation diagram (with fit)
    Corresponds to original lines 1183-1240
    Output: rho-obs.pdf
    """
    if df_props is None or df_props.empty: return
    print("Plotting Rho vs Separation (Standalone)...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rho = df_props['rho'].values
    # Convert to number density (cm^-3)
    # rho_err_u, rho_err_l are min/max values, not error amplitudes
    rho_num = rho / (cfg.MU * cfg.MH)
    
    # Process error bars
    if 'rho_err_u' in df_props.columns and 'rho_err_l' in df_props.columns:
        rho_u = df_props['rho_err_u'].values / (cfg.MU * cfg.MH)
        rho_l = df_props['rho_err_l'].values / (cfg.MU * cfg.MH)
        
        # Compute asymmetric errors: error = [left_err, right_err]
        x_err_low = rho_num - rho_l
        x_err_high = rho_u - rho_num
        
        # Avoid negative errors (should not happen, but safety check)
        x_err_low = np.maximum(0, x_err_low)
        x_err_high = np.maximum(0, x_err_high)
        
        x_err = [x_err_low, x_err_high]
    else:
        x_err = None

    dist = df_props['closest_dist_pc'].values
    
    # Filter invalid values
    mask = (rho_num > 0) & (dist > 0)
    x = rho_num[mask]
    y = dist[mask]
    
    if x_err is not None:
        x_err_plot = [x_err[0][mask], x_err[1][mask]]
        # Draw error bars
        ax.errorbar(
            x,
            y,
            xerr=x_err_plot,
            fmt='o',
            color='k',
            ecolor='gray',
            elinewidth=1,
            capsize=2,
            alpha=0.5,
            label=getattr(cfg, 'MOLECULE_PLOT_LABEL', 'Cores')
        )
    else:
        ax.scatter(
            x,
            y,
            marker='o',
            color='k',
            alpha=0.5,
            s=10,
            label=getattr(cfg, 'MOLECULE_PLOT_LABEL', 'Cores')
        )
    
    # Prepare theoretical curves
    x_line = np.logspace(np.log10(np.min(x)*0.8), np.log10(np.max(x)*1.2), 100)
    
    # Spherical Jeans: L ~ 1/sqrt(rho)
    # coeff = sqrt(pi * cs^2 / (G * mu * mh)) / pc
    coeff_sph = np.sqrt(np.pi * (cfg.CS * 1e5)**2 / (cfg.GR * cfg.MU * cfg.MH)) / cfg.PC
    y_sph = coeff_sph * (x_line ** -0.5)
    
    # Cylindrical Jeans
    coeff_cyl = 20 * cfg.CS * 1e5 / np.sqrt(4 * np.pi * cfg.GR * cfg.MU * cfg.MH) / cfg.PC
    y_cyl = coeff_cyl * (x_line ** -0.5)
    
    ax.plot(x_line, y_sph, 'b--', label=r'Spherical: $\lambda = %.4f n^{-0.5}$' % coeff_sph)
    ax.plot(x_line, y_cyl, 'r--', label=r'Cylindrical: $\lambda = %.4f n^{-0.5}$' % coeff_cyl)
    
    # Power-law fit: log(y) = a + b * log(x)
    # Original code fixes b = -0.5
    if len(x) > 5:
        def fit_func_fixed_slope(log_x, a):
            return a - 0.5 * log_x
        
        log_x = np.log(x)
        log_y = np.log(y)
        
        try:
            popt, pcov = curve_fit(fit_func_fixed_slope, log_x, log_y)
            
            # Compute errors using covariance matrix
            perr = np.sqrt(np.diag(pcov))
            a_fit = popt[0]
            a_err = perr[0]
            
            # Outlier rejection loop (original code logic)
            for _ in range(2):
                residuals = log_y - fit_func_fixed_slope(log_x, *popt)
                std_res = np.std(residuals)
                if std_res == 0: break
                mask_fit = np.abs(residuals) < 2 * std_res 
                if np.sum(mask_fit) > 5:
                    popt, pcov = curve_fit(fit_func_fixed_slope, log_x[mask_fit], log_y[mask_fit])
                    a_fit = popt[0]
                    perr = np.sqrt(np.diag(pcov))
                    a_err = perr[0]
            
            k_fit = np.exp(a_fit)
            y_fit = k_fit * (x_line ** -0.5)
            
            # Compute confidence intervals (upper and lower bounds based on std err of 'a')
            # y = exp(a) * x^-0.5
            # y_upper = exp(a + delta) * x^-0.5
            # y_lower = exp(a - delta) * x^-0.5
            # Use 3-sigma errors
            y_fit_upper = np.exp(a_fit + 3 * a_err) * (x_line ** -0.5)
            y_fit_lower = np.exp(a_fit - 3 * a_err) * (x_line ** -0.5)
            
            # Label fit formula
            fit_label = r'Fit: $\lambda = %.4f \cdot n^{-0.5}$' % k_fit
            
            ax.plot(x_line, y_fit, 'k-', alpha=0.8, linewidth=2, label=fit_label)
            
            # Draw error range (semi-transparent shading)
            ax.fill_between(x_line, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label='Fit Error (3$\sigma$)')
            
        except Exception as e:
            print(f"Fitting failed: {e}")

    ax.set_xscale('log')
    ax.set_xlabel(r'$n_{H_2} (cm^{-3})$')
    ax.set_ylabel('Separation (pc)')
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.2)
    
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, cfg.PLOT_RHO_VS_SEPARATION), bbox_inches='tight', dpi=500)
    plt.close()
    
@utils.trace_error
def plot_core_mass_function(df_props):
    """
    Plot Core Mass Function (CMF) with power-law fit
    """
    print("\nPlotting Core Mass Function (CMF)...")
    
    # Prepare data
    masses = df_props['Mc'].values
    masses = masses[masses > 0] 
    
    if len(masses) < 5:
        print("Not enough cores for CMF statistics.")
        return

    # Create save directory (products)
    product_dir = cfg.OUTPUT_DIR
    if not os.path.exists(product_dir):
        try:
            os.makedirs(product_dir, exist_ok=True)
        except OSError:
            pass

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use logarithmic binning
    log_masses = np.log10(masses)
    min_val = np.min(log_masses)
    max_val = np.max(log_masses)
    
    # Determine number of bins
    if cfg.CMF_HIST_BINS is not None:
        num_bins = cfg.CMF_HIST_BINS
    else:
        num_bins = int(1.3 * np.sqrt(len(masses)))  # Slightly more than sqrt(N)
        if num_bins < 6: num_bins = 6
        if num_bins > 20: num_bins = 20
    
    bins = np.logspace(min_val, max_val, num=num_bins)
    
    # Plot histogram (Counts per log bin)
    counts, bins_edges, patches = ax.hist(masses, bins=bins, color='skyblue', alpha=0.7, edgecolor='black', label='Data')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$dN / d \log M$ (Number of Cores)')
    ax.set_title(f'Core Mass Function (N={len(masses)})')
    
    # --- Fit high-mass end ---
    # Find peak location
    bin_centers = 10**(0.5 * (np.log10(bins_edges[1:]) + np.log10(bins_edges[:-1])))
    peak_idx = np.argmax(counts)
    
    if len(counts) - peak_idx >= 3:
        x_fit = bin_centers[peak_idx:]
        y_fit = counts[peak_idx:]
        
        # Remove zeros
        mask = y_fit > 0
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]
        
        if len(x_fit) >= 3:
            log_x = np.log10(x_fit)
            log_y = np.log10(y_fit)
            
            # Linear regression: log(dN) = k * log(M) + b
            slope, intercept, r_val, p_val, std_err = linregress(log_x, log_y)
            
            # Generate fit line
            # Cover fit region
            x_line = np.logspace(np.log10(np.min(x_fit)), np.log10(np.max(x_fit)), 100)
            y_line = 10**(intercept + slope * np.log10(x_line))
            
            ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Fit Slope = {slope:.2f}$\pm${std_err:.2f}')
            
            print(f"CMF Slope (high-mass end): {slope:.2f} (Salpeter expectation: -1.35)")

    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    save_path = os.path.join(product_dir, cfg.PLOT_CORE_MASS_FUNCTION_PNG)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"CMF generated: {save_path}")