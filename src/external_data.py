# src/external_data.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from astropy import units as u
from scipy.interpolate import interp1d
import os
import config as cfg
from src import utils

def load_li_cores():
    """
    Read Li et al. (2013) core list (corelist_omc2b.txt)
    Corresponds to lines 1243-1264 of original code
    """
    try:
        # Assume file is under DIR0 or directly in the current path
        # Based on original code logic, it reads 'corelist_omc2b.txt'
        import os
        filepath = os.path.join(cfg.INPUT_DIR, cfg.LI_CORE_FILE)
            
        with open(filepath, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Warning: Li core file '{cfg.LI_CORE_FILE}' not found. Skipping Li comparison.")
        return None

    # Initialize arrays
    core_coord = [] # Store projected coordinates [y, x] (arcsec)
    core_temp = []
    core_d = []     # Radius (arcsec)
    core_vol = []
    
    # Auxiliary variables for projection calculation (requires meta_data; store raw values or rely on external input for now)
    # To maintain independence, parse raw data here first; projection handled later in analyze function
    raw_data = []

    # Read from the third line onwards
    for line in lines[2:]:
        items = line.split()
        if len(items) < 5: continue
        
        ra = float(items[1])
        dec = float(items[2])
        temp = float(items[3])
        dcore = float(items[4]) # arcsec
        
        # Compute volume (original code line 1263)
        # vcore = (4/3)*pi*(390*pc*tan(dcore...))**3
        # Note: original code used 390pc (OMC-2/3 distance), while cfg.DISTANCE is 470pc
        # Keep original hardcoded 390 for full reproduction, or use cfg.DISTANCE
        # Original code: 390 * pc * np.tan(...) 
        # Use original value 390 here for reproduction
        dist_li = 390 * cfg.PC
        radius_cm = dist_li * np.tan(dcore * 360 / (3600 * 2 * np.pi))
        vcore = (4/3) * np.pi * (radius_cm)**3
        
        raw_data.append({
            'ra': ra, 'dec': dec, 'temp': temp, 
            'dcore': dcore, 'vcore': vcore
        })
        
    return raw_data

def analyze_li_comparison(li_raw_data, A, leave_num, dataarr_acorns, meta_data, scu1):
    """
    Execute core logic for Li et al. (2013) comparison analysis
    Includes:
    1. Coordinate projection and distance matching
    2. Compute average density (avg_rho) within each Li core region
    3. Round 2: Recompute N2H+ properties based on Li temperature (column density, Jeans length, etc.)
    """
    if not li_raw_data: return None
    print("Executing full Li et al. (2013) comparison analysis...")

    # ================= 1. Coordinate system preparation =================
    dx, dy = meta_data['dx'], meta_data['dy']
    xmin, ymin = meta_data['xmin'], meta_data['ymin']
    xmax = meta_data['xmax']
    
    # Build projected coordinates for Li cores (unit: arcsec)
    # Original code logic: [(dec-ymin)*3600, (xmax-ra)*3600]
    # Note: original code uses core_coord.append(...)
    li_coords = []
    li_temps = []
    li_ds = []
    li_vols = []
    
    for item in li_raw_data:
        y_proj = (item['dec'] - ymin) * 3600
        x_proj = (xmax - item['ra']) * 3600
        li_coords.append([y_proj, x_proj])
        li_temps.append(item['temp'])
        li_ds.append(item['dcore'])
        li_vols.append(item['vcore'])
        
    li_coords = np.array(li_coords)
    li_temps = np.array(li_temps)
    li_ds = np.array(li_ds)
    
    # Build projected coordinates for N2H+ cores
    # Only use valid leaves from leave_num
    n2hp_coords = []
    n2hp_indices = []  # Store (tree_idx, leaf_obj, original_idx)
    
    # Pre-compute basic properties (Mass, Area) for all leaves for Step 1
    # To avoid redundant computation, only calculate coordinates and assignment distances
    
    # Original index counter (for ellipse_centers alignment, etc.)
    global_idx = 0 
    
    # Temporarily store N2H+ basic data to avoid recomputation in Step 1
    n2hp_basic_props = [] 
    
    # Helper constants
    S = meta_data['S']
    beam_area = meta_data['beam_area']
    
    for j in leave_num:
        for blade in A.forest[j].leaves:
            # Extract coordinates (Pixel -> Arcsec projection)
            # dataarr: 1=x, 2=y
            xc = np.mean(dataarr_acorns[1, blade.cluster_members])
            yc = np.mean(dataarr_acorns[2, blade.cluster_members])
            
            # Projection logic must match Li
            ra_val = xc * dx + xmin
            dec_val = yc * dy + ymin
            
            y_proj = (dec_val - ymin) * 3600
            x_proj = (xmax - ra_val) * 3600
            
            n2hp_coords.append([y_proj, x_proj])
            n2hp_indices.append((j, blade))
            
            # Pre-compute Mass (based on standard Tex=15K, for Step 1 weighting)
            # Original code used Mc[i] (mass at standard temperature) in Step 1
            # Simplified: extract data, defer N_tot calculation to loop
            
            n2hp_basic_props.append({
                'members': blade.cluster_members,
                'xc': xc, 'yc': yc,
                'idx': global_idx
            })
            global_idx += 1
            
    n2hp_coords = np.array(n2hp_coords)
    
    # Compute distance matrix (Li x N2H+)
    # cdist input shape (mA, n), (mB, n) -> (mA, mB)
    dist_matrix = cdist(li_coords, n2hp_coords)
    
    # Compute N2H+ distances (for nearest neighbor calculation)
    # Note: uses projected arcsec distance; original used ellipse centers pixel distance
    # For exact reproduction, we need the pixel distance matrix
    n2hp_pixel_centers = np.array([[p['xc'], p['yc']] for p in n2hp_basic_props])
    dist_matrix_n2hp_pixel = cdist(n2hp_pixel_centers, n2hp_pixel_centers)
    
    # ================= 2. Step 1: Compute mean density avg_rho =================
    # Corresponds to original lines 1267-1323
    
    avg_rho = []
    rho_err_upper = []
    rho_err_lower = []
    
    # Pre-compute all N2H+ masses and areas at Tex=15K (for Step 1)
    # Optimization: original code computed these inside the loop
    std_masses = []
    std_areas = []
    
    # Standard parameters
    Qrot_std = utils.get_partition_function(cfg.TEX)
    c0 = 8*np.pi / cfg.LAMDA**3 / cfg.A_UL
    c1 = cfg.GL / cfg.GU
    c2_std = 1 / (utils.Jp(cfg.TEX) - utils.Jp(cfg.TBG))
    c3_std = 1 / (1 - np.exp(-cfg.HP*cfg.FREQ/(cfg.KB*cfg.TEX)))
    c4_std = Qrot_std / cfg.GL
    rts = cfg.LINE_STRENGTH_RATIO
    factor_std = 1e5 * c0 * c1 * c2_std * c3_std * c4_std * rts / cfg.X_MOL
    
    for item in n2hp_basic_props:
        mask = np.zeros(dataarr_acorns.shape[1], dtype=bool)
        mask[item['members']] = True
        clump = dataarr_acorns[:, mask]
        
        amp = clump[3, :]
        width = clump[7, :]
        disp = width / (2*np.sqrt(2*np.log(2)))
        
        flux = amp * u.Jy / beam_area.to(u.sr)
        equiv = u.brightness_temperature(cfg.FREQ*u.Hz)
        T_br = flux.to(u.K, equivalencies=equiv).value
        
        I_tot = T_br * np.abs(disp) * np.sqrt(2*np.pi) / cfg.RI # km/s
        N_tot = I_tot * factor_std
        
        m = cfg.MU * cfg.MH * np.sum(N_tot) * S
        std_masses.append(m)
        std_areas.append(len(item['members']) * S)
        
    std_masses = np.array(std_masses)
    std_areas = np.array(std_areas)
    
    # Loop over Li cores to compute avg_rho
    for k in range(len(li_coords)):
        dists = dist_matrix[k, :]
        # Condition: distance < core_d * sqrt(2) and current Li core is nearest
        
        mass_tem = []
        area_tem = []
        
        for i in range(len(n2hp_coords)):
            if dists[i] < li_ds[k] * np.sqrt(2):
                # Check if belongs to current Li core
                # Check distance from this N2H+ core to all Li cores
                dists_to_all_li = dist_matrix[:, i]
                # Filter Li cores satisfying the radius condition
                valid_li = np.where(dists_to_all_li < li_ds * np.sqrt(2))[0]
                
                if len(valid_li) > 0:
                    min_dist = np.min(dists_to_all_li[valid_li])
                    if dists[i] == min_dist:
                        mass_tem.append(std_masses[i])
                        area_tem.append(std_areas[i])
        
        if len(mass_tem) == 0:
            avg_rho.append(0)
            rho_err_upper.append(0)
            rho_err_lower.append(0)
        else:
            total_mass = np.sum(mass_tem)
            total_area = np.sum(area_tem)
            
            # Compute mean density (original code logic)
            # avg_rho = M / ((4/3)pi * R_eff^3)
            # R_eff = sqrt(Total_Area / pi)
            r_eff_tem = np.sqrt(total_area / np.pi)
            vol_tem = (4/3) * np.pi * (r_eff_tem)**3
            
            avg_rho_val = total_mass / vol_tem
            avg_rho.append(avg_rho_val)
            
            # Error calculation (original lines 1310-1315)
            # Find min/max mass/area ratio cores, subtract from total to estimate bounds
            ratios = np.array(mass_tem) / np.array(area_tem)
            min_idx = np.argmin(ratios)
            max_idx = np.argmax(ratios)
            
            m_minus_min = total_mass - mass_tem[min_idx]
            a_minus_min = total_area - area_tem[min_idx]
            rho_u = m_minus_min / ((4/3)*np.pi*(np.sqrt(a_minus_min/np.pi))**3)
            
            m_minus_max = total_mass - mass_tem[max_idx]
            a_minus_max = total_area - area_tem[max_idx]
            rho_l = m_minus_max / ((4/3)*np.pi*(np.sqrt(a_minus_max/np.pi))**3)
            
            rho_err_upper.append(rho_u)
            rho_err_lower.append(rho_l)
            
    # Special correction (original line 1323)
    if len(avg_rho) > 12:
        avg_rho[12] = 1.85e5 * cfg.MU * cfg.MH
        rho_err_upper[12] = avg_rho[12]
        rho_err_lower[12] = avg_rho[12]

    # ================= 3. Step 2: Round 2 physical property re-calculation =================
    # Corresponds to original lines 1326-1512
    
    # Load ellipse parameter matrix (for geometric correction: virial / potential energy)
    try:
        ellipse_params_path = os.path.join(cfg.INTERMEDIATE_DIR, 'ellipse_params_matrix.npy')
        ellipse_params_matrix = np.load(ellipse_params_path)
    except FileNotFoundError:
        print("Warning: ellipse_params_matrix.npy not found. Virial/Potential Energy will be skipped for Li comparison.")
        ellipse_params_matrix = None

    results = {
        'index': [], 'x': [], 'y': [], 'ra': [], 'dec': [], 
        'N_tot': [], 'Mc': [], 'M_J': [], 'M_BE': [],
        'M_vir': [], 'alpha_vir': [], 'E_pot': [],
        'velocity': [], 'width': [], 'area': [], 'eccentricity': [],
        'R_eff': [], 'R_eff_au': [], 'cs_eff': [],
        'dist_pc': [], 'dist_au': [], 'rho': [],'rho_err_u': [], 'rho_err_l': [], 'L_J_pc': [], 'L_J_pc_err_u': [], 'L_J_pc_err_l': [],
        'L_J_cy_pc': [], 'L_J_cy_pc_err_u': [], 'L_J_cy_pc_err_l': []
    }
    
    idx_counter = 0
    
    for k in range(len(li_coords)):
        dists = dist_matrix[k, :]
        
        # Temporarily store N2H+ results for this Li Core to compute avg N_tot/R_eff
        # Original Round 2 logic:
        # For each qualifying N2H+ core, compute and store in results
        # rho uses Step 1 computed avg_rho[k] (important!)
        
        # Get temperature for this Li core
        Tex = li_temps[k]
        
        # Update temperature-dependent coefficients
        Qrot = utils.get_partition_function(Tex)
        c2 = 1 / (utils.Jp(Tex) - utils.Jp(cfg.TBG))
        c3 = 1 / (1 - np.exp(-cfg.HP*cfg.FREQ/(cfg.KB*Tex)))
        c4 = Qrot / cfg.GL
        factor_li = 1e5 * c0 * c1 * c2 * c3 * c4 * rts / cfg.X_MOL
        
        for i in range(len(n2hp_coords)):
            # Same assignment logic
            if dists[i] < li_ds[k] * np.sqrt(2):
                dists_to_all_li = dist_matrix[:, i]
                valid_li = np.where(dists_to_all_li < li_ds * np.sqrt(2))[0]
                
                is_closest = False
                if len(valid_li) > 0:
                    if dists[i] == np.min(dists_to_all_li[valid_li]):
                        is_closest = True
                
                if is_closest:
                    # === Begin computing N2H+ core properties ===
                    item = n2hp_basic_props[i]
                    mask = np.zeros(dataarr_acorns.shape[1], dtype=bool)
                    mask[item['members']] = True
                    clump = dataarr_acorns[:, mask]
                    
                    # 1. Basic quantities
                    amp = clump[3, :]
                    amp_err = clump[4, :]
                    width = clump[7, :]
                    disp = width / (2*np.sqrt(2*np.log(2)))
                    
                    flux = amp * u.Jy / beam_area.to(u.sr)
                    equiv = u.brightness_temperature(cfg.FREQ*u.Hz)
                    T_br = flux.to(u.K, equivalencies=equiv).value
                    
                    I_tot = T_br * np.abs(disp) * np.sqrt(2*np.pi) / cfg.RI
                    
                    # 2. N_tot & Mass (using Li temperature)
                    N_tot0 = I_tot * factor_li
                    # Filter zeros
                    valid_N = N_tot0 != 0
                    if np.sum(valid_N) > 0:
                        N_mean = np.mean(N_tot0[valid_N])
                    else:
                        N_mean = 0
                        
                    m = cfg.MU * cfg.MH * np.sum(N_tot0) * S
                    
                    # 3. Kinematics
                    sigma_v = np.sum(disp) / len(item['members'])
                    cs_eff = np.sqrt(sigma_v**2 + 0.92*(cfg.CS**2))
                    
                    # 4. Geometry
                    area0 = len(item['members']) * S
                    R_eff = np.sqrt(area0 / np.pi)
                    
                    # 5. Key: density uses avg_rho[k]
                    rho_val = avg_rho[k]
                    rho_u = rho_err_upper[k]
                    rho_l = rho_err_lower[k]
                    
                    # 6. Jeans length
                    # L_J = sqrt(pi * cs^2 / (G * rho))
                    # Note: original uses cs (sound speed) or cs_eff (effective sound speed)?
                    # Original line 1464: L_J0_Li = (np.pi*(cs*1e5)**2/(Gr*rho0_Li))**(1/2)
                    # Uses thermal sound speed cs (0.23 km/s), not cs_eff!!
                    
                    L_J0 = (np.pi * (cfg.CS*1e5)**2 / (cfg.GR * rho_val))**(1/2)
                    L_J0_u = (np.pi * (cfg.CS*1e5)**2 / (cfg.GR * rho_l))**(1/2)  # rho in denominator, inverted
                    L_J0_l = (np.pi * (cfg.CS*1e5)**2 / (cfg.GR * rho_u))**(1/2)
                    
                    L_cy0 = 20 * cfg.CS * 1e5 / np.sqrt(4 * np.pi * cfg.GR * rho_val)
                    L_cy0_u = 20 * cfg.CS * 1e5 / np.sqrt(4 * np.pi * cfg.GR * rho_l)
                    L_cy0_l = 20 * cfg.CS * 1e5 / np.sqrt(4 * np.pi * cfg.GR * rho_u)
                    
                    # 7. Nearest neighbor distance
                    # Use pixel distance matrix
                    dists_pix = dist_matrix_n2hp_pixel[item['idx'], :]
                    sorted_idx = np.argsort(dists_pix)
                    
                    # Find nearest
                    sort_k = 1
                    while sort_k < len(sorted_idx) and (dists_pix[sorted_idx[sort_k]] == 0):
                        sort_k += 1
                    
                    if sort_k < len(sorted_idx):
                        closest_neighbor_idx = sorted_idx[sort_k]
                        
                        # Overlap check
                        # Get bounding ranges of both cores
                        # Simplified: utils.check_overlap needs raw coordinate arrays
                        # Original: array1 = [y, x], array2 = [y, x]
                        # Need to re-extract here
                        blade_curr = n2hp_indices[item['idx']][1]
                        blade_neigh = n2hp_indices[closest_neighbor_idx][1]
                        
                        arr1 = [dataarr_acorns[2, blade_curr.cluster_members], dataarr_acorns[1, blade_curr.cluster_members]]
                        arr2 = [dataarr_acorns[2, blade_neigh.cluster_members], dataarr_acorns[1, blade_neigh.cluster_members]]
                        
                        dist_val_pix = dists_pix[closest_neighbor_idx]
                        if utils.check_overlap(arr1, arr2):
                            dist_val_pix *= np.sqrt(2)
                            
                        # Convert to pc
                        dist_deg = dist_val_pix * dx
                        dist_pc = np.sin(dist_deg * np.pi / 180) * cfg.DISTANCE / cfg.PC
                    else:
                        dist_pc = 0
                        
                    # 8. M_J & M_BE
                    M_J = np.pi**(5/2) * (cs_eff*1e5)**3 / (6 * cfg.GR**(3/2) * rho_val**(1/2))
                    M_BE = 1.18 * (cs_eff*1e5)**4 / (cfg.P_IC**(1/2) * cfg.GR**(3/2))
                    
                    # 9. Virial & Potential Energy & Eccentricity
                    M_vir = 0
                    alpha_vir = 0
                    E_pot = 0
                    ecc = 0
                    
                    if ellipse_params_matrix is not None and item['idx'] < len(ellipse_params_matrix):
                        # ellipse_params: [xc, yc, a, b, theta]
                        ell_params = ellipse_params_matrix[item['idx']]
                        a_axis = ell_params[2]
                        b_axis = ell_params[3]
                        
                        f_obs = b_axis / a_axis if a_axis > 0 else 1
                        ecc = np.sqrt(1 - f_obs**2) if f_obs <= 1 else 0
                        
                        M_vir_val = utils.calculate_virial_mass(R_eff, f_obs, sigma_v * 1e5) # sigma_v in km/s -> cm/s
                        E_pot_val = utils.calculate_potential_energy(R_eff, f_obs, m) # m in g
                        
                        M_vir = M_vir_val / cfg.MSUN
                        E_pot = E_pot_val
                        if m > 0:
                            alpha_vir = M_vir_val / m
                    
                    # Compute Velocity (shift mean)
                    shift = clump[5, :]
                    velocity_val = np.sum(shift) / len(item['members'])

                    # Store in results
                    results['index'].append(idx_counter)
                    results['x'].append(item['xc'])
                    results['y'].append(item['yc'])
                    results['ra'].append(n2hp_coords[i][1]/3600 + xmin)  # Convert back to RA deg (approx)
                    results['dec'].append(n2hp_coords[i][0]/3600 + ymin)
                    results['N_tot'].append(N_mean)
                    results['Mc'].append(m / cfg.MSUN)
                    results['M_J'].append(M_J / cfg.MSUN)
                    results['M_BE'].append(M_BE / cfg.MSUN)
                    results['M_vir'].append(M_vir)
                    results['alpha_vir'].append(alpha_vir)
                    results['E_pot'].append(E_pot)
                    results['velocity'].append(velocity_val)
                    results['width'].append(sigma_v)
                    results['area'].append(area0)
                    results['eccentricity'].append(ecc)
                    results['R_eff'].append(R_eff)
                    results['R_eff_au'].append(R_eff / cfg.AU)
                    results['cs_eff'].append(cs_eff)
                    results['dist_pc'].append(dist_pc)
                    results['dist_au'].append(dist_pc * cfg.PC / cfg.AU)
                    results['rho'].append(rho_val)  # cm^-3? g/cm^3? Original is g/cm^3
                    results['rho_err_u'].append(rho_u)
                    results['rho_err_l'].append(rho_l)
                    results['L_J_pc'].append(L_J0 / cfg.PC)
                    results['L_J_pc_err_u'].append(L_J0_u / cfg.PC)
                    results['L_J_pc_err_l'].append(L_J0_l / cfg.PC)
                    
                    results['L_J_cy_pc'].append(L_cy0 / cfg.PC)
                    results['L_J_cy_pc_err_u'].append(L_cy0_u / cfg.PC)
                    results['L_J_cy_pc_err_l'].append(L_cy0_l / cfg.PC)
                    
                    idx_counter += 1
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    # Format rho column (convert to cm^-3 number density for CSV output)
    # rho_num = rho / (mu * mh)
    df['rho_num'] = df['rho'] / (cfg.MU * cfg.MH)
    
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'property_Li.csv'), index=False)
    print("Li comparison analysis completed.")
    
    return df