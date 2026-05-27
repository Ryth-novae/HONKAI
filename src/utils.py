# src/utils.py
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
import config as cfg
import functools
import traceback

def trace_error(func):
    """
    Decorator to catch exceptions and print the function name where the error occurred.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n[!] Error occurred in function: '{func.__name__}'")
            print(f"    Module: {func.__module__}")
            print(f"    Error message: {str(e)}")
            # print(traceback.format_exc()) # Optional: print full traceback here
            raise e
    return wrapper

def Jp(Tex):
    """
    Calculate radiation temperature J(T) at a given excitation temperature.
    Corresponds to the Jp function in original code.
    Formula: T0 / (exp(T0/Tex) - 1)
    """
    T0 = cfg.HP * cfg.FREQ / cfg.KB
    # Prevent division by zero
    if Tex == 0: 
        return 0
    
    # Avoid overflow
    try:
        denom = np.exp(T0 / Tex) - 1
        if denom == 0:
            return 0
        Tp = T0 / denom
    except OverflowError:
        Tp = 0
        
    return Tp

def get_partition_function(Tex):
    """
    Calculate molecular partition function Qrot(Tex) from config.
    Supports:
    - Interpolation table mode: PARTITION_FUNCTION_MODE='interp'
    - Linear rotor approximation: PARTITION_FUNCTION_MODE='linear_rotor'
    """
    try:
        tex = float(Tex)
    except Exception:
        tex = float(cfg.TEX)

    mode = str(getattr(cfg, 'PARTITION_FUNCTION_MODE', 'interp')).strip().lower()

    if mode == 'interp':
        temps = np.asarray(getattr(cfg, 'PARTITION_FUNCTION_TEMPS', []), dtype=float)
        qrots = np.asarray(getattr(cfg, 'PARTITION_FUNCTION_QROTS', []), dtype=float)

        valid = (
            temps.ndim == 1 and qrots.ndim == 1 and
            len(temps) == len(qrots) and len(temps) >= 2
        )
        if valid:
            mask = np.isfinite(temps) & np.isfinite(qrots) & (qrots > 0)
            temps = temps[mask]
            qrots = qrots[mask]

            if len(temps) >= 2:
                order = np.argsort(temps)
                temps = temps[order]
                qrots = qrots[order]

                kind = 'cubic' if len(temps) >= 4 else 'linear'
                q_fun = interp1d(temps, qrots, kind=kind, fill_value='extrapolate')
                return max(float(q_fun(tex)), 0.0)

    if mode in ('linear_rotor', 'linear'):
        B_hz = getattr(cfg, 'ROTATIONAL_CONSTANT_HZ', None)
        if B_hz is None:
            try:
                upper_j = int(getattr(cfg, 'UPPER_J', int(getattr(cfg, 'NJ', 0)) + 1))
                if upper_j > 0 and float(cfg.FREQ) > 0:
                    B_hz = float(cfg.FREQ) / (2.0 * upper_j)
            except Exception:
                B_hz = None

        if B_hz is not None and float(B_hz) > 0:
            q_lin = (cfg.KB * tex) / (cfg.HP * float(B_hz)) + (1.0 / 3.0)
            return max(float(q_lin), 0.0)

    # Backward-safe fallback to legacy table if config is incomplete.
    t_legacy = np.array([9.0, 18.0, 37.0, 75.0], dtype=float)
    q_legacy = np.array([40.88, 78.55, 154.0, 304.95], dtype=float)
    q_fun = interp1d(t_legacy, q_legacy, kind='cubic', fill_value='extrapolate')
    return max(float(q_fun(tex)), 0.0)

def edge_fun(area):
    """
    Extract edge pixel coordinates from a binary mask.
    
    Args:
        area: 2D numpy array (0/1 mask)
    Returns:
        x1, y1: Arrays of x and y coordinates of edge pixels
    """
    area0 = area.copy()
    # Find coordinates of all points with value 1
    iys, ixs = np.where(area[1:-1, 1:-1] == 1)
    # Restore to original coordinates (slice 1:-1 shifted indices by 1)
    iys = iys + 1
    ixs = ixs + 1
    ns = len(iys)
    
    # Dilate: set 3x3 neighborhood of all 1-valued points to 1
    for i in np.arange(ns):
        area0[(iys[i]-1):(iys[i]+2), (ixs[i]-1):(ixs[i]+2)] = 1
        
    # Edge = dilated region - original region
    edge_area = area0 - area
    
    # Note: np.where returns (row, col), i.e. (y, x)
    # But plotting usually needs (x, y)
    y1, x1 = np.where(edge_area == 1)
    
    return x1, y1

def check_overlap(coords1, coords2, threshold=0.1):
    """
    Check if two regions overlap.
    
    Args:
        coords1: List or array [y_coords, x_coords] (original code uses dataarr slices)
        coords2: List or array [y_coords, x_coords]
        threshold: Overlap area threshold (default > 0 means overlap, original may have specific logic)
        
    Returns:
        bool: Whether they overlap
    """
    # Extract bounding box
    # coords input is typically a list of pixel indices
    y1, x1 = coords1
    y2, x2 = coords2
    
    min_x1, max_x1 = np.min(x1), np.max(x1)
    min_y1, max_y1 = np.min(y1), np.max(y1)
    
    min_x2, max_x2 = np.min(x2), np.max(x2)
    min_y2, max_y2 = np.min(y2), np.max(y2)
    
    # Compute overlap region width and height
    overlap_w = min(max_x1, max_x2) - max(min_x1, min_x2)
    overlap_h = min(max_y1, max_y2) - max(min_y1, min_y2)
    
    # If width or height <= 0, no overlap
    if overlap_w <= 0 or overlap_h <= 0:
        return False
        
    # Simple rectangle overlap check
    # Original may have pixel-level overlap ratio; BBox overlap is for fast filtering
    return True

def calculate_potential_energy(r, f_obs, M):
    """
    Calculate gravitational potential energy of an ellipsoid.
    Corresponds to original code starting at line 751.
    
    Args:
        r: Radius (cm)
        f_obs: Observed axis ratio
        M: Mass (g)
    """
    a_density = 1.6  # Density profile index
    # Alpha shape factor
    alpha = (1 - a_density / 3) / (1 - 2 * a_density / 5)
    
    # Compute intrinsic axis ratio f
    def calc_f(f_obs):
        # Use hypergeometric function for projection correction
        return (2 / np.pi) * f_obs * hyp2f1(0.5, 0.5, 1.5, 1 - f_obs**2)
    
    f = calc_f(f_obs)
    
    # Eccentricity e
    if f <= 1:
        e = np.sqrt(1 - f**2)
    else:
        e = 0  # Sphere
        
    # Beta shape factor
    if e != 0:
        beta = np.arcsin(e) / e
    else:
        beta = 1
    
    potential_energy = - (3 / 5) * alpha * beta * cfg.GR * M**2 / r
    return potential_energy

def calculate_virial_mass(r, f_obs, sigma):
    """
    Calculate virial mass.
    Accounts for geometric shape factors.
    
    Args:
        r: Radius (cm)
        f_obs: Observed axis ratio
        sigma: Velocity dispersion (cm/s)
    """
    a_density = 1.6
    alpha = (1 - a_density / 3) / (1 - 2 * a_density / 5)
    
    def calc_f(f_obs):
        return (2 / np.pi) * f_obs * hyp2f1(0.5, 0.5, 1.5, 1 - f_obs**2)
        
    f = calc_f(f_obs)
    
    if f <= 1:
        e = np.sqrt(1 - f**2)
    else:
        e = 0
        
    if e != 0:
        beta = np.arcsin(e) / e
    else:
        beta = 1
    
    # Avoid division by zero
    if alpha * beta == 0:
        return 0
        
    virial_mass = (5 * sigma**2 * r) / (alpha * beta * cfg.GR)
    return virial_mass

def poly_flux_fun(im0, abox=[], vm=[]):
    """
    Interactive polygon flux calculation tool.
    Allows users to click regions on a plot to compute total flux.
    
    Args:
        im0: 2D image data (numpy array)
        abox: Display range [xmin, xmax, ymin, ymax] (optional)
        vm: Display vmin, vmax [min, max] (optional)
        
    Returns:
        flux0: Total flux of selected region
        imsk: Generated mask array
    """
    print("Interactive Mode: Please click points to define a polygon.")
    print("  - Left Click: Add point")
    print("  - Middle Click: Remove last point")
    print("  - Right Click: Finish and calculate")
    
    ny, nx = im0.shape
    
    # Simple contour level
    lv0 = np.arange(0.5, 1, 0.1) * im0.max()
    
    fig = plt.figure(figsize=(8, 6))
    ax0 = fig.add_subplot(111)
    
    # Set display range
    if len(abox) == 4:
        im1 = im0[abox[2]:abox[3], abox[0]:abox[1]]
        vmg = [0, im1.max()]
        ax0.axis(abox)
    else:
        vmg = [im0.min(), im0.max()]
        
    # Display image
    if len(vm) == 2:
        sc0 = ax0.imshow(im0, cmap="terrain_r", origin='lower', norm=colors.LogNorm(vmin=vm[0], vmax=vm[1]))
    else:
        sc0 = ax0.imshow(im0, cmap="terrain_r", origin='lower', vmin=vmg[0], vmax=vmg[1])
        
    plt.colorbar(sc0, ax=ax0)
    # Overlay contours
    ax0.contour(im0, levels=lv0, colors='black', origin='lower', linewidths=1.5, alpha=0.6)
    
    # === Interaction ===
    # ginput: n=points, mouse_pop=undo key, mouse_stop=stop key
    # Note: will error or hang on headless servers
    try:
        pos = plt.ginput(n=2000, show_clicks=True, timeout=30000, mouse_pop=2, mouse_stop=3)
    except Exception as e:
        print(f"Interactive input failed (Running headless?): {e}")
        return 0, np.zeros_like(im0)
    
    # Generate polygon mask
    ims = Image.new('L', (nx, ny), 0) # 'L' for 8-bit pixels, black background
    if len(pos) > 2:
        ImageDraw.Draw(ims).polygon(pos, outline=1, fill=1)
    
    imsk = np.array(ims)
    
    # Compute flux
    flux0 = np.sum(imsk * im0)
    area = np.sum(imsk)
    
    print(f'Calculation Result: Area={area:.2f} pixels, Total Flux={flux0:.4e}')
    
    plt.close()
    return flux0, imsk