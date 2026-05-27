# src/analyzer.py (Final Complete Version)
import numpy as np
import pandas as pd
try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python (cv2) not found. Ellipse fitting will fallback to circular approximation.")

from scipy.spatial.distance import cdist
from astropy import units as u
import config as cfg
from src import utils
import os
from tqdm import tqdm

@utils.trace_error
def identify_big_error_sources(df_props, ellipse_centers, meta_data):
    """
    Identify and mark potential large error sources by comparing density differences with nearest neighbor cores.
    """
    print("Identifying sources with large density errors...")
    rho = df_props['rho'].values
    big_err_idx = []
    
    # Reconstruct distance matrix (using pixel coordinates)
    centers = np.array(ellipse_centers)
    if len(centers) == 0: 
        return [], []
    
    distances = cdist(centers, centers)
    
    for i in range(len(rho)):
        dists = distances[i, :]
        sorted_indices = np.argsort(dists)
        
        # Find the nearest neighbor that is not itself
        sort_k = 1
        while sort_k < len(sorted_indices) and (sorted_indices[sort_k] == i or dists[sorted_indices[sort_k]] == 0):
            sort_k += 1
        
        # Restore logic from pipeline_original.py:
        # If the nearest neighbor density difference is too large, keep searching until a match is found or the first 10 are exhausted
        found = False
        while sort_k < len(sorted_indices) and sort_k <= 10:
            neighbor_idx = sorted_indices[sort_k]
            
            # Avoid division by zero
            if rho[neighbor_idx] != 0:
                ratio = rho[i] / rho[neighbor_idx]
                if 0.66 <= ratio <= 1.5:
                    found = True
                    break
            
            sort_k += 1
        
        if not found:
             big_err_idx.append(i)
                
    return big_err_idx, []


@utils.trace_error
def calculate_core_properties(A, dataarr_acorns, scu1, meta_data):
    """
    Loop 1: Compute basic physical properties (Mass, Density, Virial, Jeans) & error propagation
    Loop 2: Distance calculation (with overlap correction) & Jeans length
    """
    print("Calculating core properties, geometry and error propagation...")
    
    # Extract valid leaf (Core) indices
    leave_num = []
    for i in np.arange(len(A.forest)):
        if np.max(A.forest[i].cluster_vertices[1]) > 0.24:
            leave_num.append(i)

    # Prepare storage containers
    props = {
        'index': [], 'x': [], 'y': [], 'ra': [], 'dec': [],
        'Mc': [], 'M_vir': [], 'alpha_vir': [], 'M_J': [], 'M_BE': [],
        'E_pot': [],
        'velocity': [], 'width': [],
        'area': [], 'cs_eff': [], 'rho': [], 'R_eff': [], 'R_eff_au': [], 
        'eccentricity': [],
        'closest_dist_pc': [], 'closest_dist_au': [], 'L_J_pc': [], 'L_J_cy_pc': [],
        # Error columns
        'L_J_pc_err_u': [], 'L_J_pc_err_l': [],
        'L_J_cy_pc_err_u': [], 'L_J_cy_pc_err_l': [],
        'rho_err_u': [], 'rho_err_l': [],
        'cs_eff_err_u': [], 'cs_eff_err_l': []
    }
    
    # Auxiliary variables
    S = meta_data['S']
    beam_area = meta_data['beam_area']
    dx, dy = meta_data['dx'], meta_data['dy']
    xmin, ymin = meta_data['xmin'], meta_data['ymin']
    
    # Constant calculations
    Qrot = utils.get_partition_function(cfg.TEX)
    c0 = 8*np.pi / cfg.LAMDA**3 / cfg.A_UL
    c1 = cfg.GL / cfg.GU
    c2 = 1 / (utils.Jp(cfg.TEX) - utils.Jp(cfg.TBG))
    c3 = 1 / (1 - np.exp(-cfg.HP*cfg.FREQ/(cfg.KB*cfg.TEX)))
    c4 = Qrot / (cfg.GL)
    # rts = (1.074+5.371+3.222)/1.074
    rts = cfg.LINE_STRENGTH_RATIO
    
    # ------------------- Step 1: Ellipse fitting and saving -------------------
    print("Fitting ellipses and saving matrix...")
    ellipse_centers = [] # Pixel coordinates
    ellipse_params_list = [] # [xc, yc, a, b, theta]
    all_cluster_members = [] # [FIX] Store all member indices for later overlap checking
    
    npy_path = os.path.join(cfg.INTERMEDIATE_DIR, 'ellipse_params_matrix.npy')
    
    if os.path.exists(npy_path):
        print(f"Loading existing ellipse parameters from {npy_path}...")
        ellipse_params_matrix = np.load(npy_path).T #.T temp
        ellipse_params_list = ellipse_params_matrix.tolist()
        
        # Rebuild centers and member list
        for params in ellipse_params_list:
            ellipse_centers.append([params[0]/(dx*3600), params[1]/(dy*3600)]) #temp
            
        # Iterate only to collect members (skip costly fitEllipse)
        for j in leave_num:
            for blade in A.forest[j].leaves:
                all_cluster_members.append(blade.cluster_members)
                
    else:
        for j in tqdm(leave_num, desc="Processing Cores"):
            for blade in A.forest[j].leaves:
                # Store members
                all_cluster_members.append(blade.cluster_members)
                
                # Fitting preparation
                vector = np.vectorize(np.int_)
                shape_2d = scu1.shape[1:] 
                mask = np.zeros(shape_2d)
                
                # dataarr_acorns: 1=x, 2=y
                mask[vector(dataarr_acorns[2, blade.cluster_members]), vector(dataarr_acorns[1, blade.cluster_members])] = 1
                
                loop_points = np.transpose(utils.edge_fun(mask))
                
                xc, yc, a_ell, b_ell, theta = 0, 0, 0, 0, 0
                
                if len(loop_points) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(loop_points)
                        (xc, yc), (a_ell, b_ell), theta = ellipse
                        ellipse_centers.append([xc, yc])
                    except:
                        # Fitting failed (rare edge case)
                        xc = np.mean(dataarr_acorns[1, blade.cluster_members])
                        yc = np.mean(dataarr_acorns[2, blade.cluster_members])
                        ellipse_centers.append([xc, yc])
                else:
                    xc = np.mean(dataarr_acorns[1, blade.cluster_members])
                    yc = np.mean(dataarr_acorns[2, blade.cluster_members])
                    # Small points default to circle
                    r_approx = np.sqrt(len(blade.cluster_members)/np.pi)
                    a_ell, b_ell = r_approx*2, r_approx*2
                    ellipse_centers.append([xc, yc])
                
                ellipse_params_list.append([xc, yc, a_ell, b_ell, theta])
        
        # Save parameter matrix
        np.save(npy_path, np.array(ellipse_params_list))
            
    ellipse_centers = np.array(ellipse_centers)
    
    # Compute distance matrix
    if len(ellipse_centers) > 0:
        distances_matrix = cdist(ellipse_centers, ellipse_centers)
    else:
        distances_matrix = []

    # ------------------- Step 2: Physical quantity computation -------------------
    idx_counter = 0
    
    for j in leave_num:
        for blade in A.forest[j].leaves:
            # Data extraction
            mask_arr = np.zeros_like(dataarr_acorns)
            mask_arr[:, blade.cluster_members] = 1
            clump = dataarr_acorns * mask_arr
            
            amplitude = clump[3,:]
            amplitude_err = clump[4,:]
            shift = clump[5,:]
            linewidth = clump[7,:]
            linewidth_err = clump[8,:]
            
            # Line width to velocity dispersion
            factor_sig = 1.0 / (2*np.sqrt(2*np.log(2)))
            dispersion = linewidth * factor_sig
            dispersion_err = linewidth_err * factor_sig
            
            # Flux to brightness temperature
            equiv = u.brightness_temperature(cfg.FREQ*u.Hz)
            flux = amplitude * u.Jy / beam_area.to(u.sr)
            flux_err = amplitude_err * u.Jy / beam_area.to(u.sr)
            
            T_br = flux.to(u.K, equivalencies=equiv)
            T_br_err = flux_err.to(u.K, equivalencies=equiv)
            
            # Integrated intensity (with error)
            term_v_max = np.abs(dispersion + dispersion_err)
            term_v_min = np.abs(dispersion - dispersion_err)
            
            I_tot = T_br * np.abs(dispersion) * u.km/u.s * np.sqrt(2*np.pi) / cfg.RI
            I_tot_max = (T_br + T_br_err) * term_v_max * u.km/u.s * np.sqrt(2*np.pi) / cfg.RI
            I_tot_min = (T_br - T_br_err) * term_v_min * u.km/u.s * np.sqrt(2*np.pi) / cfg.RI
            
            # Column density
            factor = 1e5 * c0 * c1 * c2 * c3 * c4 * rts / cfg.X_MOL
            N_tot0 = I_tot.value * factor
            N_tot0_max = I_tot_max.value * factor
            N_tot0_min = I_tot_min.value * factor
            
            # Mass
            m = cfg.MU * cfg.MH * np.sum(N_tot0) * S
            m_max = cfg.MU * cfg.MH * np.sum(N_tot0_max) * S
            m_min = cfg.MU * cfg.MH * np.sum(N_tot0_min) * S
            
            props['Mc'].append(m / cfg.MSUN)
            
            # Kinematics and sound speed
            sigma_v_mean = np.sum(dispersion) / len(blade.cluster_members)
            sigma_v_err_mean = np.sum(dispersion_err) / len(blade.cluster_members)
            
            cs_eff0 = np.sqrt(sigma_v_mean**2 + 0.92*(cfg.CS**2))
            cs_eff0_max = np.sqrt((sigma_v_mean + sigma_v_err_mean)**2 + 0.92*(cfg.CS**2))
            cs_eff0_min = np.sqrt((sigma_v_mean - sigma_v_err_mean)**2 + 0.92*(cfg.CS**2))
            
            props['cs_eff'].append(cs_eff0)
            props['cs_eff_err_u'].append(cs_eff0_max)
            props['cs_eff_err_l'].append(cs_eff0_min)
            props['velocity'].append(np.sum(shift)/len(blade.cluster_members))
            props['width'].append(sigma_v_mean)
            
            # Geometry and density
            area0 = len(blade.cluster_members) * S
            R_eff0 = np.sqrt(area0 / np.pi)
            
            props['area'].append(area0)
            props['R_eff'].append(R_eff0)
            props['R_eff_au'].append(R_eff0 / cfg.AU)
            
            rho0 = m / ((4/3) * np.pi * R_eff0**3)
            rho0_max = m_max / ((4/3) * np.pi * R_eff0**3)
            rho0_min = m_min / ((4/3) * np.pi * R_eff0**3)
            
            props['rho'].append(rho0)
            props['rho_err_u'].append(rho0_max)
            props['rho_err_l'].append(rho0_min)
            
            # Geometric eccentricity
            ell_params = ellipse_params_list[idx_counter]
            a_axis = ell_params[2]
            b_axis = ell_params[3]
            f_obs = b_axis / a_axis if a_axis > 0 else 1
            ecc = np.sqrt(1 - f_obs**2) if f_obs <= 1 else 0
            props['eccentricity'].append(ecc)
            
            # Virial Mass
            M_vir = utils.calculate_virial_mass(R_eff0, f_obs, sigma_v_mean * 1e5)
            props['M_vir'].append(M_vir / cfg.MSUN)
            props['alpha_vir'].append((M_vir / m) if m > 0 else 0)

            # Gravitational potential energy
            E_pot = utils.calculate_potential_energy(R_eff0, f_obs, m)
            props['E_pot'].append(E_pot)

            # Jeans analysis
            M_J0 = np.pi**(5/2) * (cs_eff0*1e5)**3 / (6 * cfg.GR**(3/2) * rho0**(1/2))
            M_BE0 = 1.18 * (cs_eff0*1e5)**4 / (cfg.P_IC**(1/2) * cfg.GR**(3/2))
            props['M_J'].append(M_J0 / cfg.MSUN)
            props['M_BE'].append(M_BE0 / cfg.MSUN)
            
            L_J0 = (np.pi * (cs_eff0*1e5)**2 / (cfg.GR * rho0))**(1/2)
            L_J0_max = (np.pi * (cs_eff0_max*1e5)**2 / (cfg.GR * rho0_min))**(1/2)
            L_J0_min = (np.pi * (cs_eff0_min*1e5)**2 / (cfg.GR * rho0_max))**(1/2)
            
            L_J_cy0 = 20 * cs_eff0 * 1e5 / (4 * np.pi * cfg.GR * rho0)**(1/2)
            L_J_cy0_max = 20 * cs_eff0_max * 1e5 / (4 * np.pi * cfg.GR * rho0_min)**(1/2)
            L_J_cy0_min = 20 * cs_eff0_min * 1e5 / (4 * np.pi * cfg.GR * rho0_max)**(1/2)
            
            props['L_J_pc'].append(L_J0 / cfg.PC)
            props['L_J_pc_err_u'].append(L_J0_max / cfg.PC)
            props['L_J_pc_err_l'].append(L_J0_min / cfg.PC)
            
            props['L_J_cy_pc'].append(L_J_cy0 / cfg.PC)
            props['L_J_cy_pc_err_u'].append(L_J_cy0_max / cfg.PC)
            props['L_J_cy_pc_err_l'].append(L_J_cy0_min / cfg.PC)
            
            # Distance calculation (with overlap correction)
            if len(distances_matrix) > 0:
                dists = distances_matrix[idx_counter, :]
                sorted_indices = np.argsort(dists)
                sort_k = 1
                while sort_k < len(sorted_indices) and (sorted_indices[sort_k] == idx_counter or dists[sorted_indices[sort_k]] == 0):
                    sort_k += 1
                
                if sort_k < len(sorted_indices):
                    closest_idx = sorted_indices[sort_k]
                    dist_pix = dists[closest_idx]
                    
                    # [FIX] Strict overlap detection
                    # Get pixel coordinate lists for current and neighbor
                    coords_curr = [dataarr_acorns[2, blade.cluster_members], dataarr_acorns[1, blade.cluster_members]]
                    members_neigh = all_cluster_members[closest_idx]
                    coords_neigh = [dataarr_acorns[2, members_neigh], dataarr_acorns[1, members_neigh]]
                    
                    if utils.check_overlap(coords_curr, coords_neigh):
                        dist_pix *= np.sqrt(2)
                        
                    dist_deg = dist_pix * dx 
                    dist_pc = np.sin(dist_deg * np.pi / 180) * cfg.DISTANCE / cfg.PC
                    props['closest_dist_pc'].append(dist_pc)
                    props['closest_dist_au'].append(dist_pc * cfg.PC / cfg.AU)
                else:
                    props['closest_dist_pc'].append(0)
                    props['closest_dist_au'].append(0)
            else:
                props['closest_dist_pc'].append(0)
                props['closest_dist_au'].append(0)
            
            # Coordinate storage
            x0 = ellipse_centers[idx_counter][0]
            y0 = ellipse_centers[idx_counter][1]
            props['x'].append(x0)
            props['y'].append(y0)
            props['ra'].append(x0 * dx + xmin)
            props['dec'].append(y0 * dy + ymin)
            props['index'].append(idx_counter)
            
            idx_counter += 1

    df = pd.DataFrame(props)
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'property.csv'), index=False)
    
    return df, leave_num, ellipse_centers