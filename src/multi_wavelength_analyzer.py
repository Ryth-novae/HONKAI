
import numpy as np
import pandas as pd
from acorns import Acorns
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.stats import sigma_clipped_stats
import os
import json
try:
    import cv2
except ImportError:
    cv2 = None
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import config as cfg
from src import utils

import shutil
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.patches import Ellipse
try:
    import pyvista as pv
except ImportError:
    pv = None

class MultiWavelengthAnalyzer:
    def __init__(
        self,
        structure_fits_path,
        output_dir,
        intermediate_dir=None,
        physics_fits_path=None,
        physics_scouse_dat_path=None,
        structure_pixel_size=1,
        structure_min_radius_pix=3.332,
        structure_min_height_multiple=2,
        structure_velo_link=0.1,
        structure_dv_link=0.2,
        structure_relax=np.array([3.0, 2.0, 0.5]),
        structure_stop=3.0,
        physics_pixel_size=1,
        physics_min_radius_pix=3.332,
        physics_min_height_multiple=2,
        physics_velo_link=0.1,
        physics_dv_link=0.2,
        physics_relax=np.array([3.0, 2.0, 0.5]),
        physics_stop=3.0
    ):
        self.structure_path = structure_fits_path
        self.output_dir = output_dir
        self.intermediate_dir = intermediate_dir if intermediate_dir is not None else output_dir
        self.physics_fits_path = physics_fits_path
        self.physics_dat_path = physics_scouse_dat_path
        self.structure_pixel_size = structure_pixel_size
        self.structure_min_radius_pix = structure_min_radius_pix
        self.structure_min_height_multiple = structure_min_height_multiple
        self.structure_velo_link = structure_velo_link
        self.structure_dv_link = structure_dv_link
        self.structure_relax = structure_relax
        self.structure_stop = structure_stop

        self.physics_pixel_size = physics_pixel_size
        self.physics_min_radius_pix = physics_min_radius_pix
        self.physics_min_height_multiple = physics_min_height_multiple
        self.physics_velo_link = physics_velo_link
        self.physics_dv_link = physics_dv_link
        self.physics_relax = physics_relax
        self.physics_stop = physics_stop
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)

        self.physics_data_map = {} # (x, y) -> list of components
        self.forest = None
        self.structure_wcs = None
        self.structure_shape = None
        self.is_2d = True
        self.physics_forest = None
        self.physics_dataarr_acorns = None
        self.physics_dataarr_raw = None
        self.physics_component_to_leaf = {}
        self.physics_leaf_area = {}

        try:
            self.mapping_subpixel_divisions = max(1, int(getattr(cfg, 'WCS_MAPPING_SUBPIXEL_DIVISIONS', 5)))
        except Exception:
            self.mapping_subpixel_divisions = 5

        try:
            self.mapping_min_overlap = float(getattr(cfg, 'WCS_MAPPING_MIN_OVERLAP', 0.01))
        except Exception:
            self.mapping_min_overlap = 0.01

        try:
            self.physics_max_component_snr = float(getattr(cfg, 'B_MAX_COMPONENT_SNR', 1.0e4))
        except Exception:
            self.physics_max_component_snr = 1.0e4
        self.physics_max_component_snr = max(1.0, self.physics_max_component_snr)

        try:
            self.mapping_fallback_filling_threshold = float(
                getattr(cfg, 'WCS_MAPPING_FALLBACK_FILLING_FACTOR_THRESHOLD', 0.5)
            )
        except Exception:
            self.mapping_fallback_filling_threshold = 0.5
        self.mapping_fallback_filling_threshold = min(
            1.0, max(0.0, self.mapping_fallback_filling_threshold)
        )
        
        # Load physics meta data (Beam, etc.)
        self.meta_data_B = self._load_physics_metadata()

    def _cluster_depth_from_trunk(self, node, trunk):
        """
        Compute hierarchy depth for a cluster node relative to a tree trunk.
        Depth 0 means trunk, larger values are lower levels in the dendrogram.
        """
        if node is trunk:
            return 0

        depth = 0
        current = node
        visited = set()

        while current is not None and current is not trunk:
            if id(current) in visited:
                break
            visited.add(id(current))

            parent = getattr(current, 'antecedent', None)
            if parent is None:
                break

            depth += 1
            current = parent

        return depth

    def get_structure_nodes_for_scale(self, scale_level=0):
        """
        Return de-duplicated structure nodes for a hierarchy level.

        Parameters
        ----------
        scale_level : int
            0=trunk (largest parent), 1=next level, 2=next-next level, ...
        """
        if self.forest is None:
            return []

        try:
            target_level = max(0, int(scale_level))
        except Exception:
            target_level = 0

        strict = bool(getattr(cfg, 'ANALYSIS_STRUCTURE_SCALE_STRICT', True))
        selected = []
        seen = set()
        skipped_trees = 0

        iterator = self.forest.forest.values() if isinstance(self.forest.forest, dict) else self.forest.forest
        for tree in iterator:
            if hasattr(tree, 'tree_members') and hasattr(tree, 'trunk'):
                trunk = tree.trunk
                members = list(tree.tree_members)
            elif hasattr(tree, 'cluster_members'):
                trunk = tree
                members = [tree]
            else:
                continue

            tree_nodes_with_level = []
            for node in members:
                level = self._cluster_depth_from_trunk(node, trunk)
                tree_nodes_with_level.append((node, level))

            candidates = [node for node, level in tree_nodes_with_level if level == target_level]
            if len(candidates) == 0 and not strict and len(tree_nodes_with_level) > 0:
                max_level = max(level for _, level in tree_nodes_with_level)
                candidates = [node for node, level in tree_nodes_with_level if level == max_level]

            if len(candidates) == 0:
                skipped_trees += 1
                continue

            for node in candidates:
                node_key = id(node)
                if node_key in seen:
                    continue
                seen.add(node_key)
                selected.append(node)

        if skipped_trees > 0:
            print(
                f"Scale {target_level}: skipped {skipped_trees} trees "
                f"without nodes at this hierarchy level."
            )

        return selected

    def get_available_structure_scale_levels(self):
        """
        Return sorted hierarchy levels available in the current structure forest.

        Levels follow trunk-based indexing used by get_structure_nodes_for_scale:
        0=trunk/root, 1=next level, ...
        """
        if self.forest is None:
            return []

        levels = set()
        iterator = self.forest.forest.values() if isinstance(self.forest.forest, dict) else self.forest.forest
        for tree in iterator:
            if hasattr(tree, 'tree_members') and hasattr(tree, 'trunk'):
                trunk = tree.trunk
                members = list(tree.tree_members)
            elif hasattr(tree, 'cluster_members'):
                trunk = tree
                members = [tree]
            else:
                continue

            for node in members:
                levels.add(int(self._cluster_depth_from_trunk(node, trunk)))

        return sorted(levels)

    def _node_is_leaf(self, node):
        """Best-effort check whether a node is a leaf-like terminal structure."""
        if node is None:
            return False

        if hasattr(node, 'leaf_cluster'):
            try:
                return bool(node.leaf_cluster)
            except Exception:
                pass

        descendants = getattr(node, 'descendants', None)
        if descendants is not None:
            try:
                return len(descendants) == 0
            except Exception:
                pass

        leaves = getattr(node, 'leaves', None)
        if leaves is not None:
            try:
                return len(leaves) <= 1
            except Exception:
                pass

        return False

    def get_structure_nodes_for_group(self, group_name):
        """
        Return structure nodes for fixed pseudo-groups.

        Supported group_name:
        - 'leaves_all': all terminal structures in all trees
        - 'roots_nonleaf': each tree trunk, excluding trunks that are themselves leaves
        """
        if self.forest is None:
            return []

        group = str(group_name).strip().lower()
        selected = []
        seen = set()
        iterator = self.forest.forest.values() if isinstance(self.forest.forest, dict) else self.forest.forest

        if group == 'leaves_all':
            for tree in iterator:
                if hasattr(tree, 'leaves') and tree.leaves is not None:
                    nodes = list(tree.leaves)
                elif hasattr(tree, 'cluster_members'):
                    nodes = [tree] if self._node_is_leaf(tree) else []
                else:
                    nodes = []

                for node in nodes:
                    node_key = id(node)
                    if node_key in seen:
                        continue
                    seen.add(node_key)
                    selected.append(node)

            return selected

        if group == 'roots_nonleaf':
            for tree in iterator:
                trunk = None
                if hasattr(tree, 'trunk'):
                    trunk = tree.trunk
                elif hasattr(tree, 'cluster_members'):
                    trunk = tree

                if trunk is None:
                    continue
                if self._node_is_leaf(trunk):
                    continue

                node_key = id(trunk)
                if node_key in seen:
                    continue
                seen.add(node_key)
                selected.append(trunk)

            return selected

        return []

    def _resolution_match_flag(self, wcs_a_celestial, wcs_b_celestial):
        """Return resolution-match flag for CSV observability."""
        flag_default = str(getattr(cfg, 'RESOLUTION_MATCH_FLAG_DEFAULT', 'not_matched'))
        if wcs_a_celestial is None or wcs_b_celestial is None:
            return 'unknown'
        try:
            area_ratio = self._estimate_pixel_area_ratio_ab(wcs_a_celestial, wcs_b_celestial)
            if np.isfinite(area_ratio) and area_ratio > 0:
                return flag_default
        except Exception:
            pass
        return 'unknown'

    def _build_a_intensity_map(self, pts_x, pts_y, pts_i):
        """Build A-pixel intensity map (mean per pixel) from per-point arrays."""
        if pts_i is None:
            return {}

        try:
            xi = np.asarray(pts_x).astype(int)
            yi = np.asarray(pts_y).astype(int)
            ii = np.asarray(pts_i).astype(float)
        except Exception:
            return {}

        n = min(len(xi), len(yi), len(ii))
        if n == 0:
            return {}

        accum = {}
        for x, y, val in zip(xi[:n], yi[:n], ii[:n]):
            if not np.isfinite(val):
                continue
            key = (int(x), int(y))
            if key not in accum:
                accum[key] = [float(val)]
            else:
                accum[key].append(float(val))

        return {k: float(np.mean(v)) for k, v in accum.items() if len(v) > 0}

    def _compute_leaf_spearman_score(
        self,
        leaf_id,
        a_pixel_component_map,
        a_intensity_map,
        min_points=5
    ):
        """
        Compute Spearman rho between A intensity and B component amplitude
        for one candidate B leaf, using only overlap-valid A pixels.
        """
        if not a_intensity_map:
            return {'valid': False, 'rho': np.nan, 'pvalue': np.nan, 'n': 0}

        a_vals = []
        b_vals = []

        for a_key, comps in a_pixel_component_map.items():
            a_val = a_intensity_map.get(a_key, np.nan)
            if not np.isfinite(a_val):
                continue

            b_num = 0.0
            b_den = 0.0
            for c in comps:
                comp_idx = c.get('idx')
                if comp_idx is None:
                    continue
                if self.physics_component_to_leaf.get(comp_idx) != leaf_id:
                    continue

                amp = float(c.get('amp', np.nan))
                frac_a = float(c.get('frac_a', 0.0))
                if (not np.isfinite(amp)) or frac_a <= 0:
                    continue

                b_num += frac_a * amp
                b_den += frac_a

            if b_den <= 0:
                continue

            b_val = b_num / b_den
            if not np.isfinite(b_val):
                continue

            a_vals.append(float(a_val))
            b_vals.append(float(b_val))

        if len(a_vals) < int(min_points):
            return {'valid': False, 'rho': np.nan, 'pvalue': np.nan, 'n': int(len(a_vals))}

        a_arr = np.asarray(a_vals, dtype=float)
        b_arr = np.asarray(b_vals, dtype=float)
        valid = np.isfinite(a_arr) & np.isfinite(b_arr)
        a_arr = a_arr[valid]
        b_arr = b_arr[valid]

        n_use = int(a_arr.size)
        if n_use < int(min_points):
            return {'valid': False, 'rho': np.nan, 'pvalue': np.nan, 'n': n_use}

        # Constant vectors make Spearman undefined.
        if np.allclose(a_arr, a_arr[0]) or np.allclose(b_arr, b_arr[0]):
            return {'valid': False, 'rho': np.nan, 'pvalue': np.nan, 'n': n_use}

        rho, pval = spearmanr(a_arr, b_arr)
        if not np.isfinite(rho):
            return {'valid': False, 'rho': np.nan, 'pvalue': np.nan, 'n': n_use}

        return {
            'valid': True,
            'rho': float(rho),
            'pvalue': float(pval) if np.isfinite(pval) else np.nan,
            'n': n_use
        }

    @utils.trace_error
    def _load_physics_metadata(self):
        """
        Attempts to load beam size and other metadata from the physics FITS file (B) if provided.
        Falls back to estimates or config if not available.
        """
        meta = {
            'beam_area': None,
            'S': None,
            'dx': 1, 'dy': 1
        }

        def _compute_beam_area_and_S_from_deg(bmaj_deg, bmin_deg):
            """Compute beam solid angle from beam major/minor (degrees)."""
            bmaj = float(bmaj_deg) * u.deg
            bmin = float(bmin_deg) * u.deg
            fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
            beam_area = 2 * np.pi * (bmaj * bmin * fwhm_to_sigma**2)
            return beam_area

        # Priority 1: manual B-beam from config in DUAL mode.
        cfg_beam = getattr(cfg, 'DUAL_B_BEAM_SIZE_ARCSEC', None)
        if cfg_beam is not None:
            try:
                if np.isscalar(cfg_beam):
                    bmaj_arcsec = float(cfg_beam)
                    bmin_arcsec = float(cfg_beam)
                elif len(cfg_beam) == 2:
                    bmaj_arcsec = float(cfg_beam[0])
                    bmin_arcsec = float(cfg_beam[1])
                else:
                    raise ValueError(
                        "DUAL_B_BEAM_SIZE_ARCSEC must be scalar or length-2 sequence"
                    )

                if bmaj_arcsec <= 0 or bmin_arcsec <= 0:
                    raise ValueError("beam size must be positive")

                bmaj_deg = bmaj_arcsec / 3600.0
                bmin_deg = bmin_arcsec / 3600.0
                beam_area = _compute_beam_area_and_S_from_deg(bmaj_deg, bmin_deg)
                meta['beam_area'] = beam_area
                print(
                    "  Using manual DUAL-B beam size from config "
                    f"(arcsec): BMAJ={bmaj_arcsec}, BMIN={bmin_arcsec}"
                )
                print(f"  Beam Area (config): {beam_area}")
            except Exception as e:
                print(f"  Warning: Invalid DUAL_B_BEAM_SIZE_ARCSEC in config: {e}")
                print("  Falling back to FITS header for beam metadata.")
        
        if self.physics_fits_path and os.path.exists(self.physics_fits_path):
            print(f"Loading physics metadata from {self.physics_fits_path}...")
            try:
                hdul = fits.open(self.physics_fits_path)
                header = hdul[0].header
                
                # Priority 2: FITS header beam (only if config did not provide beam_area).
                if meta['beam_area'] is None and 'BMAJ' in header and 'BMIN' in header:
                    beam_area = _compute_beam_area_and_S_from_deg(header['BMAJ'], header['BMIN'])
                    meta['beam_area'] = beam_area
                    print(f"  Beam Area found in FITS header: {beam_area}")
                elif meta['beam_area'] is None:
                    print("  Warning: BMAJ/BMIN not found. Cannot calculate beam_area via header.")

                # S is the physical area of one B pixel (cm^2), not beam area.
                # Prefer WCS-based projected pixel area in steradian.
                if meta['S'] is None:
                    try:
                        wcs_b = wcs.WCS(header)
                        try:
                            wcs_b_celestial = wcs_b.celestial
                        except Exception:
                            wcs_b_celestial = wcs_b

                        omega_pix_sr = np.abs(wcs.utils.proj_plane_pixel_area(wcs_b_celestial))
                        if np.isfinite(omega_pix_sr) and omega_pix_sr > 0:
                            meta['S'] = (cfg.DISTANCE**2) * omega_pix_sr
                            print(f"  Pixel area S from WCS: {meta['S']}")
                    except Exception as e:
                        print(f"  Warning: Could not compute S from WCS pixel area: {e}")

                # Fallback for S if WCS-based area is unavailable.
                if meta['S'] is None and 'CDELT1' in header:
                    dx_deg = abs(header['CDELT1'])
                    size_cm = cfg.DISTANCE * np.tan(np.deg2rad(dx_deg))
                    meta['S'] = size_cm**2
                    print(f"  Fallback S calculated from CDELT1: {meta['S']}")
                
                hdul.close()
            except Exception as e:
                print(f"  Warning: Could not read header from physics FITS: {e}")
        
        # Final Fallback check
        if meta['S'] is None:
             print("  Warning: S could not be determined. Mass calculations will be wrong.")
             meta['S'] = 1.0 # Avoid crash
        
        return meta

    @utils.trace_error
    def load_physics_data(self):
        """
        Load ScousePy best fit solution file (physics data 'B').
        Organize it into a dictionary for fast spatial lookup.
        Assumes columns: 1:x, 2:y, 3:amp, 5:velocity, 7:width, 9:rms
        """
        print(f"Loading physics data from {self.physics_dat_path}...")
        self.physics_data_map = {}
        try:
            raw_data = np.loadtxt(self.physics_dat_path, skiprows=1)
        except OSError:
            raise FileNotFoundError(f"Could not load {self.physics_dat_path}")

        raw_data = self._filter_physics_raw_rows(raw_data, context_label="physics_data_map")
        if raw_data.size == 0:
            print("Physics data file is empty.")
            return

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        xs = raw_data[:, 1].astype(int)
        ys = raw_data[:, 2].astype(int)
        amps = raw_data[:, 3]
        vels = raw_data[:, 5]
        widths = raw_data[:, 7] # FWHM usually, check if Scouse outputs sigma
        rms = raw_data[:, 9]

        for i in range(len(xs)):
            key = (xs[i], ys[i])
            comp = {
                'idx': i,
                'amp': amps[i],
                'vel': vels[i],
                'width': widths[i], # This is taken as FWHM based on analyzer.py
                'rms': rms[i]
            }
            if key not in self.physics_data_map:
                self.physics_data_map[key] = []
            self.physics_data_map[key].append(comp)
            
        print(f"Loaded {len(raw_data)} spectral components into map.")

    def _filter_physics_raw_rows(self, raw_data, context_label="physics"):
        """
        Remove clearly invalid Scouse components before ACORNS/mapping.
        Intensity-based filtering is currently disabled by design.
        """
        if raw_data is None:
            return np.array([])

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        if raw_data.size == 0:
            return raw_data

        if raw_data.shape[1] <= 9:
            print(f"Warning: {context_label} raw_data has too few columns: {raw_data.shape}")
            return raw_data

        amp = raw_data[:, 3]
        vel = raw_data[:, 5]
        wid = raw_data[:, 7]
        rms = raw_data[:, 9]

        valid = np.isfinite(amp) & np.isfinite(vel) & np.isfinite(wid) & np.isfinite(rms)
        valid &= (wid > 0) & (rms > 0)

        filtered = raw_data[valid]
        removed = raw_data.shape[0] - filtered.shape[0]
        if removed > 0:
            print(
                f"Filtered {removed}/{raw_data.shape[0]} anomalous components in {context_label} "
                f"(finite + width/rms checks only)."
            )

        return filtered

    def _estimate_pixel_area_ratio_ab(self, wcs_a_celestial, wcs_b_celestial):
        """
        Estimate ratio = area(B pixel) / area(A pixel) on projected plane.
        Used to convert sampled A-area on B pixels into B coverage fraction.
        """
        try:
            area_a = np.abs(wcs.utils.proj_plane_pixel_area(wcs_a_celestial))
            area_b = np.abs(wcs.utils.proj_plane_pixel_area(wcs_b_celestial))
            if np.isfinite(area_a) and np.isfinite(area_b) and area_a > 0 and area_b > 0:
                return float(area_b / area_a)
        except Exception:
            pass
        return 1.0

    def _build_overlap_maps_a_to_b(self, a_pixels, wcs_a_celestial, wcs_b_celestial=None, nx_b=None, ny_b=None):
        """
        Build overlap mappings between A structure pixels and B pixels using subpixel WCS sampling.

        Returns
        -------
        a_to_b_weights : dict
            {(ax, ay): {(bx, by): frac_of_A_pixel_area}}
        b_coverage : dict
            {(bx, by): frac_of_B_pixel_covered_by_A_union}, clipped to [0, 1]
        """
        a_to_b_weights = {}
        b_coverage = {}

        if a_pixels is None:
            return a_to_b_weights, b_coverage

        a_pixels = np.asarray(a_pixels)
        if a_pixels.size == 0:
            return a_to_b_weights, b_coverage

        if a_pixels.ndim == 1:
            a_pixels = a_pixels.reshape(1, -1)

        a_pixels = np.unique(a_pixels[:, :2].astype(int), axis=0)
        if a_pixels.size == 0:
            return a_to_b_weights, b_coverage

        subpix = max(1, int(self.mapping_subpixel_divisions))
        offsets = (np.arange(subpix, dtype=float) + 0.5) / subpix - 0.5
        dx, dy = np.meshgrid(offsets, offsets)
        dx = dx.ravel()
        dy = dy.ravel()
        n_sub = dx.size

        n_pix = a_pixels.shape[0]
        pix_ids = np.repeat(np.arange(n_pix), n_sub)
        sample_x = np.repeat(a_pixels[:, 0].astype(float), n_sub) + np.tile(dx, n_pix)
        sample_y = np.repeat(a_pixels[:, 1].astype(float), n_sub) + np.tile(dy, n_pix)

        try:
            if wcs_b_celestial is not None:
                ra, dec = wcs_a_celestial.wcs_pix2world(sample_x, sample_y, 0)
                bx, by = wcs_b_celestial.wcs_world2pix(ra, dec, 0)
            else:
                bx, by = sample_x, sample_y
        except Exception:
            bx, by = sample_x, sample_y

        bx_i = np.floor(bx + 0.5).astype(int)
        by_i = np.floor(by + 0.5).astype(int)

        valid = np.isfinite(bx) & np.isfinite(by)
        if nx_b is not None and ny_b is not None:
            valid &= (bx_i >= 0) & (bx_i < nx_b) & (by_i >= 0) & (by_i < ny_b)

        if not np.any(valid):
            return a_to_b_weights, b_coverage

        pix_ids_v = pix_ids[valid]
        bx_v = bx_i[valid]
        by_v = by_i[valid]

        triplets = np.column_stack((pix_ids_v, bx_v, by_v))
        uniq_triplets, counts_triplets = np.unique(triplets, axis=0, return_counts=True)

        for row, count in zip(uniq_triplets, counts_triplets):
            pix_id = int(row[0])
            b_key = (int(row[1]), int(row[2]))
            a_key = (int(a_pixels[pix_id, 0]), int(a_pixels[pix_id, 1]))
            if a_key not in a_to_b_weights:
                a_to_b_weights[a_key] = {}
            a_to_b_weights[a_key][b_key] = float(count) / float(n_sub)

        ratio_b_over_a = 1.0
        if wcs_b_celestial is not None:
            ratio_b_over_a = self._estimate_pixel_area_ratio_ab(wcs_a_celestial, wcs_b_celestial)

        b_pairs = np.column_stack((bx_v, by_v))
        uniq_pairs, counts_pairs = np.unique(b_pairs, axis=0, return_counts=True)
        for row, count in zip(uniq_pairs, counts_pairs):
            b_key = (int(row[0]), int(row[1]))
            area_a_units = float(count) / float(n_sub)
            frac_b = area_a_units / ratio_b_over_a if ratio_b_over_a > 0 else area_a_units
            b_coverage[b_key] = b_coverage.get(b_key, 0.0) + frac_b

        for key in list(b_coverage.keys()):
            b_coverage[key] = float(min(1.0, max(0.0, b_coverage[key])))

        return a_to_b_weights, b_coverage

    def _select_physics_components_for_structure(
        self,
        pts_x,
        pts_y,
        pts_z,
        pts_i=None,
        wcs_a_celestial=None,
        wcs_b_celestial=None,
        nx_b=None,
        ny_b=None
    ):
        """
        Select B-side spectral components used for one A structure.
        This mirrors the selection/fallback logic in physical-property calculation.

        Returns
        -------
        dict or None
            Keys include:
            - total_pixels
            - filling_factor
            - selected_components
            - component_weights
            - effective_b_area_pix
            - selected_component_indices
            - selected_b_pixels
            - selected_comp_map
            - a_to_b_weights
            - b_coverage
            - used_fallback
        """
        if pts_x is None or pts_y is None or len(pts_x) == 0:
            return None

        a_pixels = np.unique(np.column_stack((pts_x, pts_y)).astype(int), axis=0)
        if a_pixels.size == 0:
            return None

        a_to_b_weights, b_coverage = self._build_overlap_maps_a_to_b(
            a_pixels,
            wcs_a_celestial=wcs_a_celestial,
            wcs_b_celestial=wcs_b_celestial,
            nx_b=nx_b,
            ny_b=ny_b
        )

        if not a_to_b_weights or not b_coverage:
            return None

        cand_vels = []
        a_pixel_component_map = {}

        for a_key, b_weights in a_to_b_weights.items():
            comps_for_a = []
            for b_key, frac_a in b_weights.items():
                if frac_a < self.mapping_min_overlap:
                    continue
                comps_at_b = self.physics_data_map.get(b_key)
                if not comps_at_b:
                    continue

                for comp in comps_at_b:
                    comp_local = {
                        'idx': comp.get('idx'),
                        'amp': comp['amp'],
                        'vel': comp['vel'],
                        'width': comp['width'],
                        'rms': comp['rms'],
                        'b_key': b_key,
                        'frac_a': float(frac_a),
                        'frac_b': float(b_coverage.get(b_key, 0.0))
                    }
                    comps_for_a.append(comp_local)
                    cand_vels.append(comp_local['vel'])
            if comps_for_a:
                a_pixel_component_map[a_key] = comps_for_a

        if not cand_vels:
            return None

        selected_component_candidates = []
        selected_b_pixels = set()
        total_pixels = len(a_to_b_weights)
        mapped_b_area = float(np.sum(list(b_coverage.values()))) if b_coverage else 0.0

        matching_method_used = 'legacy_overlap_area'
        spearman_rho = np.nan
        spearman_pvalue = np.nan
        spearman_n = 0
        spearman_valid = 0
        selected_leaf_id = None

        spearman_enabled = bool(getattr(cfg, 'SPEARMAN_MATCHING_ENABLED', True))
        spearman_min_points = max(3, int(getattr(cfg, 'SPEARMAN_MIN_POINTS', 5)))
        spearman_tie_tol = float(getattr(cfg, 'SPEARMAN_TIE_TOL', 1.0e-3))

        a_intensity_map = self._build_a_intensity_map(pts_x, pts_y, pts_i)

        def _collect_leaf_components(leaf_id):
            comps_sel = []
            b_pix_sel = set()
            for comps in a_pixel_component_map.values():
                for c in comps:
                    comp_idx = c.get('idx')
                    if comp_idx is None:
                        continue
                    if self.physics_component_to_leaf.get(comp_idx) != leaf_id:
                        continue
                    comps_sel.append(c)
                    if c.get('frac_b', 0.0) > 0:
                        b_pix_sel.add(c['b_key'])
            return comps_sel, b_pix_sel

        def _compute_filling_for_b_pixels(b_pixels):
            covered_area = 0.0
            for _, b_weights in a_to_b_weights.items():
                frac_sum = 0.0
                for b_key, frac_a in b_weights.items():
                    if b_key in b_pixels:
                        frac_sum += frac_a
                covered_area += min(1.0, frac_sum)
            return covered_area / total_pixels if total_pixels > 0 else 0.0

        if self.is_2d:
            leaf_ids = set()
            leaf_b_pixels = {}

            for comps in a_pixel_component_map.values():
                for c in comps:
                    comp_idx = c.get('idx')
                    if comp_idx is None:
                        continue
                    leaf_id = self.physics_component_to_leaf.get(comp_idx)
                    if leaf_id is not None:
                        leaf_ids.add(leaf_id)
                        if leaf_id not in leaf_b_pixels:
                            leaf_b_pixels[leaf_id] = set()
                        leaf_b_pixels[leaf_id].add(c['b_key'])

            leaf_scores = {
                leaf_id: np.sum([b_coverage.get(k, 0.0) for k in b_keys])
                for leaf_id, b_keys in leaf_b_pixels.items()
            }

            # Spearman-first with iterative filling-factor validation.
            # If valid Spearman candidates exist but all fail threshold, direct fallback is used later.
            has_valid_spearman = False
            if spearman_enabled and len(leaf_ids) > 0:
                valid_spearman = {}
                for lid in leaf_ids:
                    score = self._compute_leaf_spearman_score(
                        lid,
                        a_pixel_component_map,
                        a_intensity_map,
                        min_points=spearman_min_points
                    )
                    if score.get('valid', False):
                        valid_spearman[lid] = score

                if len(valid_spearman) > 0:
                    has_valid_spearman = True
                    remaining = set(valid_spearman.keys())
                    ranked_spearman_leaf_ids = []
                    while len(remaining) > 0:
                        best_rho = max(valid_spearman[lid]['rho'] for lid in remaining)
                        rho_group = [
                            lid for lid in remaining
                            if abs(valid_spearman[lid]['rho'] - best_rho) <= spearman_tie_tol
                        ]

                        if mapped_b_area <= 0:
                            rho_group_sorted = sorted(
                                rho_group,
                                key=lambda lid: (
                                    -leaf_scores.get(lid, 0.0),
                                    int(lid)
                                )
                            )
                        else:
                            rho_group_sorted = sorted(
                                rho_group,
                                key=lambda lid: (
                                    -leaf_scores.get(lid, 0.0),
                                    abs((self.physics_leaf_area.get(lid, 0.0) / mapped_b_area) - 1.0),
                                    int(lid)
                                )
                            )

                        ranked_spearman_leaf_ids.extend(rho_group_sorted)
                        remaining -= set(rho_group)

                    for lid in ranked_spearman_leaf_ids:
                        comp_cands, b_pix_cands = _collect_leaf_components(lid)
                        if len(comp_cands) == 0:
                            continue

                        cand_filling = _compute_filling_for_b_pixels(b_pix_cands)
                        if cand_filling < self.mapping_fallback_filling_threshold:
                            continue

                        selected_leaf_id = lid
                        selected_component_candidates = comp_cands
                        selected_b_pixels = b_pix_cands
                        spearman_valid = 1
                        spearman_rho = float(valid_spearman[lid]['rho'])
                        spearman_pvalue = float(valid_spearman[lid]['pvalue']) if np.isfinite(valid_spearman[lid].get('pvalue', np.nan)) else np.nan
                        spearman_n = int(valid_spearman[lid]['n'])
                        matching_method_used = 'spearman'
                        break

            # Legacy selection remains available only when Spearman is disabled
            # or when no valid Spearman candidates exist.
            if selected_leaf_id is None and len(leaf_ids) > 0:
                if (not spearman_enabled) or (not has_valid_spearman):
                    ranked_leaf_ids = sorted(leaf_scores.keys(), key=lambda lid: leaf_scores[lid], reverse=True)
                    if len(ranked_leaf_ids) == 1 or mapped_b_area <= 0:
                        selected_leaf_id = ranked_leaf_ids[0]
                    else:
                        top_score = leaf_scores[ranked_leaf_ids[0]]
                        top_leafs = [
                            lid for lid in ranked_leaf_ids
                            if np.isclose(leaf_scores[lid], top_score, rtol=1e-6, atol=1e-6)
                        ]
                        if len(top_leafs) == 1:
                            selected_leaf_id = top_leafs[0]
                        else:
                            selected_leaf_id = min(
                                top_leafs,
                                key=lambda lid: abs((self.physics_leaf_area.get(lid, 0.0) / mapped_b_area) - 1.0)
                            )

                    selected_component_candidates, selected_b_pixels = _collect_leaf_components(selected_leaf_id)
                    matching_method_used = 'legacy_overlap_area'

            # Spearman had valid candidates but none met the filling threshold:
            # keep empty selection here so fallback_direct_scouse is triggered below.
            if selected_leaf_id is None and spearman_enabled and has_valid_spearman:
                matching_method_used = 'spearman'
        else:
            if pts_z is not None and len(pts_z) > 0:
                mz = np.mean(pts_z)
            else:
                mz = 0

            try:
                cx = np.mean(pts_x)
                cy = np.mean(pts_y)
                cz = np.mean(pts_z) if (pts_z is not None and len(pts_z) > 0) else mz
                world = self.structure_wcs.wcs_pix2world([[cx, cy, cz]], 0)
                mean_v_core = world[0][2]

                try:
                    spec_unit = u.Unit(self.structure_wcs.wcs.cunit[2])
                    if spec_unit == u.m / u.s:
                        mean_v_core /= 1000.0
                except Exception:
                    if abs(mean_v_core) > 10000:
                        mean_v_core /= 1000.0
            except Exception:
                mean_v_core = mz

            v_low, v_high = (mean_v_core - 3.0, mean_v_core + 3.0)
            range_cen = (v_low + v_high) / 2.0
            for _, comps in a_pixel_component_map.items():
                valid = [c for c in comps if v_low <= c['vel'] <= v_high]
                if valid:
                    best = min(
                        valid,
                        key=lambda c: (
                            abs(c['vel'] - range_cen),
                            -c.get('frac_b', 0.0),
                            -c.get('frac_a', 0.0)
                        )
                    )
                    selected_component_candidates.append(best)
                    if best.get('frac_b', 0.0) > 0:
                        selected_b_pixels.add(best['b_key'])

        covered_a_area = 0.0
        for _, b_weights in a_to_b_weights.items():
            frac_sum = 0.0
            for b_key, frac_a in b_weights.items():
                if b_key in selected_b_pixels:
                    frac_sum += frac_a
            covered_a_area += min(1.0, frac_sum)

        filling_factor = covered_a_area / total_pixels if total_pixels > 0 else 0.0

        selected_comp_map = {}
        for c in selected_component_candidates:
            comp_idx = c.get('idx')
            if comp_idx is None:
                continue
            frac_b = float(c.get('frac_b', 0.0))
            if frac_b <= 0:
                continue

            if comp_idx not in selected_comp_map:
                selected_comp_map[comp_idx] = {
                    'comp': c,
                    'weight': min(1.0, frac_b)
                }
            else:
                selected_comp_map[comp_idx]['weight'] = max(
                    selected_comp_map[comp_idx]['weight'],
                    min(1.0, frac_b)
                )

        used_fallback = False
        if (
            len(selected_comp_map) == 0
            or filling_factor < self.mapping_fallback_filling_threshold
        ):
            used_fallback = True
            matching_method_used = 'fallback_direct_scouse'
            selected_comp_map = {}
            selected_b_pixels = set()

            for b_key, frac_b in b_coverage.items():
                if frac_b < self.mapping_min_overlap:
                    continue

                comps_at_b = self.physics_data_map.get(b_key)
                if not comps_at_b:
                    continue

                strongest = max(comps_at_b, key=lambda c: c['amp'])
                comp_idx = strongest.get('idx')
                if comp_idx is None:
                    continue

                selected_comp_map[comp_idx] = {
                    'comp': strongest,
                    'weight': min(1.0, float(frac_b))
                }
                selected_b_pixels.add(b_key)

            covered_a_area = 0.0
            for _, b_weights in a_to_b_weights.items():
                frac_sum = 0.0
                for b_key, frac_a in b_weights.items():
                    if b_key in selected_b_pixels:
                        frac_sum += frac_a
                covered_a_area += min(1.0, frac_sum)

            filling_factor = covered_a_area / total_pixels if total_pixels > 0 else 0.0

        selected_components = [v['comp'] for v in selected_comp_map.values()]
        component_weights = np.array([v['weight'] for v in selected_comp_map.values()], dtype=float)
        effective_b_area_pix = np.sum([b_coverage.get(k, 0.0) for k in selected_b_pixels]) if selected_b_pixels else 0.0
        selected_component_indices = sorted([int(k) for k in selected_comp_map.keys()])
        overlap_frac_a = float(filling_factor)
        overlap_frac_b = float(effective_b_area_pix / mapped_b_area) if mapped_b_area > 0 else 0.0
        resolution_match_flag = self._resolution_match_flag(wcs_a_celestial, wcs_b_celestial)

        return {
            'total_pixels': total_pixels,
            'filling_factor': float(filling_factor),
            'selected_components': selected_components,
            'component_weights': component_weights,
            'effective_b_area_pix': float(effective_b_area_pix),
            'selected_component_indices': selected_component_indices,
            'selected_b_pixels': selected_b_pixels,
            'selected_comp_map': selected_comp_map,
            'a_to_b_weights': a_to_b_weights,
            'b_coverage': b_coverage,
            'used_fallback': used_fallback,
            'selected_leaf_id': selected_leaf_id,
            'matching_method_used': matching_method_used,
            'spearman_rho': float(spearman_rho) if np.isfinite(spearman_rho) else np.nan,
            'spearman_pvalue': float(spearman_pvalue) if np.isfinite(spearman_pvalue) else np.nan,
            'spearman_n': int(spearman_n),
            'spearman_valid': int(spearman_valid),
            'overlap_frac_a': overlap_frac_a,
            'overlap_frac_b': overlap_frac_b,
            'resolution_match_flag': resolution_match_flag
        }

    def _run_scouse_generic(self, fits_path, output_dir, coverage_mask=None):
        """
        Runs ScousePy on a given FITS file.
        If coverage_mask is provided, it applies it to the FITS file before running Scouse.
        Returns the path to the output .dat file.
        """
        try:
            from scousepy import scouse
            from scousepy.io import output_ascii_indiv
        except ImportError:
            raise ImportError("scousepy not installed.")

        # Prepare filenames
        fits_basename = os.path.basename(fits_path)
        fname_no_ext = os.path.splitext(fits_basename)[0]
        fits_dir = output_dir # Scouse looks here
        
        target_fits_path = os.path.join(output_dir, fits_basename)

        # Handle Masking
        if coverage_mask is not None:
             fname_no_ext = f"masked_{fname_no_ext}"
             target_fits_path = os.path.join(output_dir, f"{fname_no_ext}.fits")
             
             print(f"Applying coverage mask to {fits_path}...")
             with fits.open(fits_path) as hdu_list:
                data = hdu_list[0].data
                header = hdu_list[0].header
                
                data_masked = data.copy()
                
                # Broadcast mask
                if data_masked.ndim == 3 and coverage_mask.ndim == 2:
                    mask_expanded = coverage_mask[np.newaxis, :, :]
                    data_masked = np.where(mask_expanded > 0, data_masked, np.nan)
                elif data_masked.ndim == 2:
                    data_masked = np.where(coverage_mask > 0, data_masked, np.nan)
                else:
                    # Generic multiply attempt
                    try: data_masked = data_masked * coverage_mask
                    except: print("Warning: Direct mask broadcast failed.")
                
                fits.writeto(target_fits_path, data_masked, header=header, overwrite=True)
                print(f"Masked FITS saved to {target_fits_path}")
        else:
             # If no mask, copy original to intermediate so Scouse finds it easily
             if not os.path.exists(target_fits_path):
                 shutil.copy(fits_path, target_fits_path)

        # Run Scouse Steps
        print(f"Initializing ScousePy setup for {fname_no_ext}...")
        config_file = scouse.run_setup(fname_no_ext, fits_dir, outputdir=output_dir)
        
        print("Running ScousePy fitting stages...")
        s = scouse.stage_1(config=config_file, interactive=False)
        s = scouse.stage_2(config=config_file)
        s = scouse.stage_3(config=config_file)
        s = scouse.stage_4(config=config_file, bitesize=True)
        
        print("Exporting ScousePy results...")
        output_ascii_indiv(s, output_dir)
        
        # Locate Output DAT
        generated_dat = None
        for f in os.listdir(output_dir):
             if f.endswith('.dat'):
                 generated_dat = os.path.join(output_dir, f)
                 break
                 
        if generated_dat:
             print(f"ScousePy solution found: {generated_dat}")
             return generated_dat
        else:
             raise FileNotFoundError("ScousePy failed to generate a .dat solution file.")

    def _build_acorns_from_scouse(self, dat_path, save_name="physics_B.acorn"):
        """
        Build an ACORNS forest from a ScousePy .dat file and index leaf areas/components.
        Stores results in physics_* attributes.
        """
        if not os.path.exists(dat_path):
            raise FileNotFoundError(f"ScousePy .dat file not found: {dat_path}")

        print(f"Building ACORNS forest for physics data from {dat_path}...")
        try:
            raw_data = np.loadtxt(dat_path, skiprows=1)
        except Exception:
            raw_data = np.loadtxt(dat_path)

        raw_data = self._filter_physics_raw_rows(raw_data, context_label="physics_acorns")
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        if raw_data.size == 0:
            raise ValueError(f"No valid Scouse components left after filtering: {dat_path}")

        if raw_data.shape[1] <= 9:
            raise ValueError(f"ScousePy output format unexpected: {raw_data.shape}")

        # Standard Scouse columns: 1:x, 2:y, 3:amp, 4:amp_err, 5:vel, 7:width, 9:rms
        try:
            data_subset = raw_data[:, [1, 2, 3, 4, 5, 7, 9]]
        except IndexError:
            if raw_data.shape[1] >= 7:
                data_subset = raw_data[:, :7]
            else:
                raise ValueError(f"ScousePy output format unexpected: {raw_data.shape}")

        dataarr_T = data_subset.T
        dataarr_acorns = dataarr_T[[0, 1, 2, 3, 4, 5], :]

        try:
            rms_noise = np.nanmedian(raw_data[:, 9])
        except Exception:
            rms_noise = np.nanmedian(raw_data[:, -1])

        min_height = self.physics_min_height_multiple * rms_noise
        cluster_criteria = np.array([self.physics_min_radius_pix, self.physics_velo_link, self.physics_dv_link])

        save_path = os.path.join(self.intermediate_dir, save_name)
        meta_path = f"{save_path}.meta.json"

        # Include selection-rule signature in cache metadata so stale forests
        # are rebuilt when filtering logic changes.
        filter_signature = "no_intensity_filter__finite_vel_amp_wid_rms__wid_rms_pos_v1"

        current_meta = {
            'dat_path': os.path.abspath(dat_path),
            'dat_mtime': os.path.getmtime(dat_path),
            'dat_size': os.path.getsize(dat_path),
            'physics_min_radius_pix': float(self.physics_min_radius_pix),
            'physics_min_height_multiple': float(self.physics_min_height_multiple),
            'physics_velo_link': float(self.physics_velo_link),
            'physics_dv_link': float(self.physics_dv_link),
            'physics_pixel_size': float(self.physics_pixel_size),
            'physics_stop': float(self.physics_stop),
            'physics_relax': [float(v) for v in np.asarray(self.physics_relax).flatten().tolist()],
            'physics_max_component_snr': float(self.physics_max_component_snr),
            'physics_filter_signature': filter_signature
        }

        reuse_existing = False
        if os.path.exists(save_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    saved_meta = json.load(f)
                if saved_meta == current_meta:
                    reuse_existing = True
            except Exception:
                reuse_existing = False

        if reuse_existing:
            print(f"Loading existing physics ACORNS forest: {save_path}")
            forest = Acorns.load_from(save_path)
        else:
            if os.path.exists(save_path):
                print("Existing physics ACORNS cache is stale or incompatible. Rebuilding...")
            forest = Acorns.process(
                dataarr_acorns,
                cluster_criteria,
                method="PPV",
                min_height=min_height,
                pixel_size=self.physics_pixel_size,
                relax=self.physics_relax,
                stop=self.physics_stop,
                verbose=True
            )
            forest.save_to(save_path)
            print(f"Physics ACORNS forest saved to {save_path}")
            try:
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(current_meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save ACORNS cache metadata: {e}")

        self.physics_forest = forest
        self.physics_dataarr_acorns = dataarr_acorns
        self.physics_dataarr_raw = raw_data

        self._index_physics_leaves()

    def _index_physics_leaves(self):
        """
        Index physics forest leaves for fast component->leaf lookup and leaf area.
        """
        self.physics_component_to_leaf = {}
        self.physics_leaf_area = {}

        if self.physics_forest is None or self.physics_dataarr_acorns is None:
            return

        all_leaves = []
        iterator = self.physics_forest.forest.values() if isinstance(self.physics_forest.forest, dict) else self.physics_forest.forest
        for tree in iterator:
            if hasattr(tree, 'leaves'):
                all_leaves.extend(tree.leaves)
            elif hasattr(tree, 'cluster_members'):
                all_leaves.append(tree)

        for leaf_id, node in enumerate(all_leaves):
            indices = None
            if hasattr(node, 'cluster_members'):
                indices = node.cluster_members
            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                indices = node.trunk.cluster_members

            if indices is None:
                continue

            for idx in indices:
                self.physics_component_to_leaf[idx] = leaf_id

            pts_x = self.physics_dataarr_acorns[0, indices].astype(int)
            pts_y = self.physics_dataarr_acorns[1, indices].astype(int)
            area_pix = len(set(zip(pts_x, pts_y)))
            self.physics_leaf_area[leaf_id] = area_pix

    @utils.trace_error
    def run_structure_analysis(self, method='PP',  save_name='structure.acorn'):
        intermediate_dir = os.path.join(self.intermediate_dir, save_name)
        pixel_size = self.structure_pixel_size
        min_radius_pix = self.structure_min_radius_pix
        min_height_multiple = self.structure_min_height_multiple
        velo_link = self.structure_velo_link
        dv_link = self.structure_dv_link
        relax = self.structure_relax
        stop = self.structure_stop
        
        # Load header info
        hdul = fits.open(self.structure_path)
        hdu = hdul[0]
        header = hdu.header
        data = np.squeeze(hdu.data)
        self.structure_wcs = wcs.WCS(header)
        self.structure_shape = data.shape
        hdul.close()

        print(f"Analyzing structure from {self.structure_path} using {method}...")

        data_clean = np.nan_to_num(data)
        
        # --- Construct Data Array (Before Loading/Running) ---
        if method == 'PP':
            # Calculate RMS first
            ind = data_clean != 0
            if np.sum(ind) > 0:
                # Use sigma clipping for robust RMS estimation (better than simple MAD)
                mean, median, std = sigma_clipped_stats(data_clean[ind], sigma=3.0, maxiters=5)
                rms_noise = std
            else:
                rms_noise = 1.0 
                
            print(f"Estimated RMS noise for structure: {rms_noise}")
        
            self.is_2d = True
            if data_clean.ndim > 2:
                 data_clean_2d = np.max(data_clean, axis=0)
            else:
                 data_clean_2d = data_clean
            
            ny, nx = data_clean_2d.shape
            x = np.arange(nx)
            y = np.arange(ny)
            xx, yy = np.meshgrid(x, y)
            xx = xx.flatten(order='F')
            yy = yy.flatten(order='F')
            data_flat = data_clean_2d.flatten(order='F')
            noise_arr = np.ones_like(data_flat) * rms_noise
            
            self.dataarr_acorns = np.array([xx, yy, data_flat, noise_arr])
            cluster_criteria = np.array([min_radius_pix])
            relax = np.array([0])
            
        elif method == 'PPV':
            self.is_2d = False
            
            # Step 1: Run ScousePy on dataset A
            print("Method is PPV. Running ScousePy on Structure FITS first (SINGLE mode)...")
            dat_name_A = "scouse_structure_A.dat"
            dat_path_A = os.path.join(self.intermediate_dir, dat_name_A)
            
            if os.path.exists(dat_path_A):
                 print(f"Found existing ScousePy A solution: {dat_path_A}")
            else:
                 gen_dat = self._run_scouse_generic(self.structure_path, self.intermediate_dir, coverage_mask=None)
                 shutil.copy(gen_dat, dat_path_A)
                 
            # Step 2: Create dataarr_acorns from .dat
            print(f"Building ACORNS data array from {dat_path_A}...")
            try:
                raw_data = np.loadtxt(dat_path_A, skiprows=1) # Assuming standard Scouse output
            except Exception as e:
                # Try skiprows=0?
                raw_data = np.loadtxt(dat_path_A)

            if raw_data.ndim == 1: raw_data = raw_data.reshape(1, -1)
            
            rms_noise = np.nanmedian(raw_data[:, 9])
            
            # Scouse Columns (File): 1:x, 2:y, 3:amp, 4:amp_err, 5:vel, 6:vel_err, 7:width, 8:width_err, 9:rms
            # User Requested Mapping: x, y, peak, err, vel, width, rms
            # User indices in conceptual array (if file had these cols in order): 0,1,2,3,4,6,8
            # Mapping Scouse File Cols: 1, 2, 3, 4, 5, 7, 9
            
            # Extract relevant columns from Scouse output
            # Cols: 1(x), 2(y), 3(amp), 4(err), 5(vel), 7(width), 9(rms)
            try:
                # Select only the needed columns from raw data
                # We need columns [1, 2, 3, 4, 5, 7, 9] (0-based index from raw_data if skiprow removed header)
                data_subset = raw_data[:, [1, 2, 3, 4, 5, 7, 9]]
                
                            
            except IndexError:
                print("Warning: Standard ScousePy columns not found. Attempting fallback or using available columns.")
                # Fallback: assume user provided a clean file? Use as is if 7 cols?
                if raw_data.shape[1] >= 7:
                    data_subset = raw_data[:, :7]
                else: 
                     raise ValueError(f"ScousePy output format unexpected: {raw_data.shape}")

            # User instruction: 
            # dataarr = np.array(dataarr[:,np.array([0,1,2,3,4,6,8])]).T  <-- This assumes data_subset above is correct order
            # Actually, we constructed data_subset to be [x, y, amp, err, vel, width, rms]
            # So its indices 0..6 correspond accurately.
            # User wants dataarr_acorns from [0, 1, 2, 3, 4, 5] of this subset.
            # So we drop RMS (index 6).
            
            dataarr_T = data_subset.T 
            # construct dataarr_acorns: rows 0(x), 1(y), 2(amp), 3(err), 4(vel), 5(width)
            self.dataarr_acorns = dataarr_T[[0, 1, 2, 3, 4, 5], :]
            
            # Determine criteria
            # Spatial: min_radius_pix
            # Spectral: Scouse Velocity Width? Or fixed?
            # Default "1.0" in previous code was for pixel channels.
            # Here "1.0" km/s might be too large/small depending on source.
            # Let's approximate velocity resolution from Header (CDELT3) if possible.
            try:
                dv = abs(self.structure_wcs.wcs.cdelt[2]) 
                # Check units? Assuming header is km/s or m/s.
                if dv > 100: dv /= 1000.0 # Convert m/s -> km/s guess
            except:
                dv = 0.5 # Default guess 0.5 km/s
            
            print(f"Using spectral clustering radius: {dv} km/s (Derived/Guess)")
            cluster_criteria = np.array([min_radius_pix, velo_link, dv_link]) 

        else:
            raise ValueError("Method must be 'PP' or 'PPV'.")

        # Check if existing result can be used
        if os.path.exists(intermediate_dir):
             print(f"Found existing structure analysis: {intermediate_dir}")
             print("Loading forest from file (Skipping re-analysis)...")
             try:
                 self.forest = Acorns.load_from(intermediate_dir)
                 
                 # Force inferred is_2d from method logic, verified by leaf check if needed
                 # But we already set self.is_2d based on 'method' above.
                 print(f"Loaded structure forest. (is_2d={self.is_2d})")
                 
                 return self.forest
             except Exception as e:
                 print(f"Failed to load existing file: {e}. Proceeding with new analysis.")

        min_height = min_height_multiple * rms_noise
        
        print(f"Running Acorns process with {self.dataarr_acorns.shape[1]} points...")
        self.forest = Acorns.process(self.dataarr_acorns, cluster_criteria, method=method, min_height=min_height, pixel_size=pixel_size, relax=relax, stop=stop, verbose=True)
        
        self.forest.save_to(intermediate_dir)
        print(f"Structure analysis saved to {intermediate_dir}")
        
        # Auto-plot
        try:
             self.plot_structures()
        except Exception as e:
             print(f"Warning: Failed to plot structures: {e}")
             
        return self.forest
    
    # [Old run_structure_analysis is replaced by above]
    
    @utils.trace_error
    def run_scouse_on_structures(self, coverage_name='structure_coverage.fits', dat_name='scouse_solution.dat'):
        """
        Generates a coverage mask from identified structures and runs ScousePy on dataset B.
        Returns the path to the generated .dat file.
        """
        # Check existing result
        target_dat_path = os.path.join(self.output_dir, dat_name)
        physics_acorn_path = os.path.join(self.intermediate_dir, "physics_B.acorn")

        if self.is_2d and os.path.exists(target_dat_path) and os.path.exists(physics_acorn_path):
            print(f"Found existing physics ACORNS forest: {physics_acorn_path}")
            print("Skipping ScousePy and ACORNS for B...")
            self.physics_dat_path = target_dat_path
            if self.physics_forest is None:
                self._build_acorns_from_scouse(target_dat_path, save_name="physics_B.acorn")
            return self.physics_dat_path

        if os.path.exists(target_dat_path):
             print(f"Found existing ScousePy solution: {target_dat_path}")
             print("Skipping ScousePy fitting...")
             self.physics_dat_path = target_dat_path
             if self.is_2d and self.physics_forest is None:
                 # Build physics ACORNS forest from existing .dat
                 self._build_acorns_from_scouse(target_dat_path)
             return self.physics_dat_path

        if self.forest is None:
            raise RuntimeError("Run run_structure_analysis() first.")
        if not self.physics_fits_path:
            raise ValueError("physics_fits_path is required to run ScousePy.")

        # If structure is 2D, do NOT mask B. Run full ScousePy + ACORNS on B.
        if self.is_2d:
            print("Structure is 2D. Running ScousePy on full B (no mask) and building 3D structures...")
            gen_dat = self._run_scouse_generic(self.physics_fits_path, self.intermediate_dir, coverage_mask=None)
            shutil.copy(gen_dat, target_dat_path)
            self.physics_dat_path = target_dat_path
            self._build_acorns_from_scouse(target_dat_path)
            return self.physics_dat_path

        print("Generating coverage mask for ScousePy on B...")
        
        # Open B to get shape and WCS
        if not os.path.exists(self.physics_fits_path):
             raise FileNotFoundError(f"Physics fits file not found: {self.physics_fits_path}")

        with fits.open(self.physics_fits_path) as hdul_b:
            header_b = hdul_b[0].header
            wcs_b = wcs.WCS(header_b)
            shape_b = hdul_b[0].shape
        
        try: wcs_b_celestial = wcs_b.celestial
        except: wcs_b_celestial = wcs_b

        # Determine spatial dimensions of B
        nx = header_b.get('NAXIS1')
        ny = header_b.get('NAXIS2')
        if nx is None or ny is None:
             if len(shape_b) >= 2:
                 nx = shape_b[-1]
                 ny = shape_b[-2]
             else:
                 raise ValueError("Cannot determine spatial dimensions of B.")
        
        mask = np.zeros((ny, nx), dtype=int)
        
        # Get A's spatial WCS
        try: wcs_a_celestial = self.structure_wcs.celestial
        except: wcs_a_celestial = self.structure_wcs

        # Iterate all leaves in A
        all_leaves = []
        is_dict = isinstance(self.forest.forest, dict)
        iterator = self.forest.forest.values() if is_dict else self.forest.forest
        
        for tree in iterator:
            if hasattr(tree, 'leaves'): 
                # If tree object has leaves, we take them
                all_leaves.extend(tree.leaves)
            elif hasattr(tree, 'cluster_members'): 
                # If tree object is itself a cluster/leaf
                all_leaves.append(tree)

        if not all_leaves: print("Warning: No leaves found in forest structure.")

        count_valid_pixels = 0.0
        overlap_threshold = max(0.0, float(self.mapping_min_overlap))
        
        for i, node in enumerate(all_leaves):
            pts_x, pts_y = None, None
            indices = None

            if hasattr(node, 'cluster_members'): indices = node.cluster_members
            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'): indices = node.trunk.cluster_members
            
            if indices is not None and hasattr(self, 'dataarr_acorns') and self.dataarr_acorns is not None:
                 pts_x = self.dataarr_acorns[0, indices].astype(int)
                 pts_y = self.dataarr_acorns[1, indices].astype(int)
            
            # Removed usage of .data access as Acorns objects rely on indices + dataarr

            
            if pts_x is None or len(pts_x) == 0: continue

            a_pixels = np.unique(np.column_stack((pts_x, pts_y)).astype(int), axis=0)
            _, b_coverage = self._build_overlap_maps_a_to_b(
                a_pixels,
                wcs_a_celestial=wcs_a_celestial,
                wcs_b_celestial=wcs_b_celestial,
                nx_b=nx,
                ny_b=ny
            )

            for (bx, by), frac_b in b_coverage.items():
                if frac_b >= overlap_threshold:
                    mask[by, bx] = 1
                    count_valid_pixels += frac_b
        
        print(f"DEBUG: Total weighted B coverage mapped from A: {count_valid_pixels:.2f}")
        print(f"DEBUG: Total B pixels kept in coverage mask: {int(np.sum(mask))}")

        if np.sum(mask) == 0:
            raise RuntimeError("ScousePy coverage mask is empty! No structures mapped to the physics FITS field.")
                
        # Save coverage mask
        cov_path = os.path.join(self.intermediate_dir, coverage_name)
        header_mask = wcs_b_celestial.to_header()
        fits.writeto(cov_path, mask, header=header_mask, overwrite=True)
        print(f"Coverage mask saved to {cov_path}")
        
        # Run Scouse Generic on B with Mask
        gen_dat = self._run_scouse_generic(self.physics_fits_path, self.intermediate_dir, coverage_mask=mask)
        
        # Rename to desired output
        target = os.path.join(self.intermediate_dir, dat_name)
        shutil.copy(gen_dat, target)
        self.physics_dat_path = target
        
        return self.physics_dat_path
    
    @utils.trace_error
    def map_and_analyze(self, selected_nodes=None, scale_level=0):
        if self.forest is None:
            raise RuntimeError("Run run_structure_analysis() first.")

        if self.physics_dat_path is None:
             raise ValueError("physics_dat_path is None. Please provide it in init or run run_scouse_on_structures() first.")

        print(f"Mapping structures and calculating properties (scale={scale_level})...")
        
        # Load Physics Data (Now that we definitely have the path)
        self.load_physics_data()

        if self.is_2d and self.physics_forest is None:
            raise RuntimeError("Physics ACORNS forest missing in 2D mode. Run run_scouse_on_structures() first.")
        
        # --- Setup WCS mapping A -> B ---
        wcs_b_celestial = None
        nx_b, ny_b = None, None
        if self.physics_fits_path and os.path.exists(self.physics_fits_path):
            try:
                with fits.open(self.physics_fits_path) as hdul_b:
                    hdu_b = hdul_b[0]
                    wcs_b = wcs.WCS(hdu_b.header)
                    nx_b = hdu_b.header.get('NAXIS1')
                    ny_b = hdu_b.header.get('NAXIS2')
                    if (nx_b is None or ny_b is None) and hdu_b.data is not None and hdu_b.data.ndim >= 2:
                        ny_b = hdu_b.data.shape[-2]
                        nx_b = hdu_b.data.shape[-1]
                try: wcs_b_celestial = wcs_b.celestial
                except: wcs_b_celestial = wcs_b
            except:
                print("Warning: Could not load WCS from physics FITS. Using pixel-pixel match.")
        
        try: wcs_a_celestial = self.structure_wcs.celestial
        except: wcs_a_celestial = self.structure_wcs

        # Physics constants
        S = self.meta_data_B['S']
        beam_area = self.meta_data_B['beam_area']
        
        # Check if we can compute Mass (needs beam_area)
        can_compute_mass = beam_area is not None
        if not can_compute_mass:
            print("Warning: 'beam_area' not found (Physics FITS not provided or missing header). Mass/Density calculations will be skipped or limited.")

        # Constants from analyzer.py
        Qrot = utils.get_partition_function(cfg.TEX)
        c0 = 8*np.pi / cfg.LAMDA**3 / cfg.A_UL
        c1 = cfg.GL / cfg.GU
        c2 = 1 / (utils.Jp(cfg.TEX) - utils.Jp(cfg.TBG))
        c3 = 1 / (1 - np.exp(-cfg.HP*cfg.FREQ/(cfg.KB*cfg.TEX)))
        c4 = Qrot / (cfg.GL)
        # rts = (1.074+5.371+3.222)/1.074
        rts = cfg.LINE_STRENGTH_RATIO
        factor_col_density = 1e5 * c0 * c1 * c2 * c3 * c4 * rts / cfg.X_MOL

        results = []
        direct_scouse_fallback_count = 0
        
        # Collect structure nodes to analyze at the requested scale.
        if selected_nodes is None:
            all_leaves = self.get_structure_nodes_for_scale(scale_level)
        else:
            all_leaves = list(selected_nodes)
        print(f"Total structures to analyze: {len(all_leaves)}")
        
        for i, node in enumerate(all_leaves):
            pts_x, pts_y = None, None
            pts_z = None # For 3D
            pts_i = None
            
            # --- UPDATED DATA ACCESS ---
            indices = None
            if hasattr(node, 'cluster_members'):
                    indices = node.cluster_members
            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                    indices = node.trunk.cluster_members
            
            if indices is not None and hasattr(self, 'dataarr_acorns') and self.dataarr_acorns is not None:
                    pts_x = self.dataarr_acorns[0, indices].astype(int)
                    pts_y = self.dataarr_acorns[1, indices].astype(int)
                    if self.dataarr_acorns.shape[0] >= 3:
                        pts_i = self.dataarr_acorns[2, indices].astype(float)
                    if self.dataarr_acorns.shape[0] >= 5:
                        pts_z = self.dataarr_acorns[4, indices]
            
            if pts_x is None or len(pts_x) == 0:
                continue

            selection = self._select_physics_components_for_structure(
                pts_x=pts_x,
                pts_y=pts_y,
                pts_z=pts_z,
                pts_i=pts_i,
                wcs_a_celestial=wcs_a_celestial,
                wcs_b_celestial=wcs_b_celestial,
                nx_b=nx_b,
                ny_b=ny_b
            )
            if selection is None:
                continue

            if selection['used_fallback']:
                direct_scouse_fallback_count += 1

            total_pixels = selection['total_pixels']
            filling_factor = selection['filling_factor']
            selected_components = selection['selected_components']
            component_weights = selection['component_weights']
            effective_b_area_pix = selection['effective_b_area_pix']
            
            # --- Physics Calculation ---
            if filling_factor > 0 and len(selected_components) > 0 and np.sum(component_weights) > 0:
                widths = np.array([c['width'] for c in selected_components]) # FWHM
                amps_raw = np.array([c['amp'] for c in selected_components])
                amps = amps_raw * component_weights
                vels = np.array([c['vel'] for c in selected_components])
                rms_arr = np.array([c['rms'] for c in selected_components])
                
                # 1. Geometry: Ellipse Fitting for Eccentricity (f_obs)
                # Create mask from structure pixels
                padding = 5
                min_x, max_x = min(pts_x), max(pts_x)
                min_y, max_y = min(pts_y), max(pts_y)
                w = max_x - min_x + 2*padding
                h = max_y - min_y + 2*padding
                mask = np.zeros((h, w), dtype=np.uint8)
                for px, py in zip(pts_x, pts_y):
                    mask[py - min_y + padding, px - min_x + padding] = 1
                
                edge_pts = np.argwhere(mask > 0) # y, x
                # Swap to x, y for cv2
                pts_for_fit = np.array([ [p[1], p[0]] for p in edge_pts ]).astype(np.float32) # x, y local
                
                try:
                    if len(pts_for_fit) >= 5:
                        (xc, yc), (a_fit, b_fit), theta = cv2.fitEllipse(pts_for_fit)
                        a_axis, b_axis = max(a_fit, b_fit), min(a_fit, b_fit)
                    else:
                        # Circular approx
                        r_approx = np.sqrt(total_pixels/np.pi)
                        a_axis, b_axis = r_approx*2, r_approx*2
                except:
                    r_approx = np.sqrt(total_pixels/np.pi)
                    a_axis, b_axis = r_approx*2, r_approx*2

                f_obs = b_axis / a_axis if a_axis > 0 else 1.0
                eccentricity = np.sqrt(1 - f_obs**2) if f_obs <= 1 else 0

                # 2. Velocity Dispersion
                # Flux weighted Width (FWHM)
                total_amp = np.sum(amps)
                w_mean_fwhm = np.average(widths, weights=amps) if total_amp > 0 else np.mean(widths)
                v_mean = np.average(vels, weights=amps) if total_amp > 0 else np.mean(vels)
                
                # Convert FWHM to sigma (km/s)
                factor_sig = 1.0 / (2 * np.sqrt(2 * np.log(2)))
                sigma_v_1d = w_mean_fwhm * factor_sig
                
                # Effective Sound Speed
                cs_eff = np.sqrt(sigma_v_1d**2 + 0.92*(cfg.CS**2))
                
                # 3. Radius & Area
                area_phys = max(effective_b_area_pix, 0.0) * S # cm^2, based on mapped B coverage
                radius_cm = np.sqrt(area_phys / np.pi)
                radius_pc = radius_cm / cfg.PC
                radius_au = radius_cm / cfg.AU

                area_a_pix = float(max(total_pixels, 0.0))
                area_b_pix_eff = float(max(effective_b_area_pix, 0.0))
                max_area = max(area_a_pix, area_b_pix_eff)
                size_similarity = (min(area_a_pix, area_b_pix_eff) / max_area) if max_area > 0 else 0.0
                
                # 4. Mass (Mc) & Density (rho)
                Mc = 0
                rho = 0
                if can_compute_mass:
                    equiv = u.brightness_temperature(cfg.FREQ*u.Hz)
                    # We do this per component
                    fluxes = amps * u.Jy / beam_area.to(u.sr)
                    T_brs = fluxes.to(u.K, equivalencies=equiv).value
                    
                    # Integrated Intensity I_tot = T_br * sigma_v * sqrt(2pi) / ri
                    sigmas = widths * factor_sig
                    I_tots = T_brs * sigmas * np.sqrt(2*np.pi) / cfg.RI # K km/s
                    
                    # Column Density N = I_tot * factor
                    N_tots = I_tots * factor_col_density
                    
                    # Mass sum
                    mass_g = cfg.MU * cfg.MH * np.sum(N_tots) * S
                    Mc = mass_g / cfg.MSUN
                    
                    # Density rho
                    vol = (4/3) * np.pi * radius_cm**3
                    rho = mass_g / vol if vol > 0 else 0
                    
                # 5. Virial Mass
                # M_vir with shape correction
                sigma_cgs = sigma_v_1d * 1e5
                m_vir_g = utils.calculate_virial_mass(radius_cm, f_obs, sigma_cgs)
                M_vir = m_vir_g / cfg.MSUN
                
                alpha_vir = (M_vir / Mc) if Mc > 0 else 0
                
                # 6. Potential Energy
                E_pot = 0
                if can_compute_mass:
                    E_pot = utils.calculate_potential_energy(radius_cm, f_obs, mass_g)
                    
                # 7. Jeans Mass & Bonnor-Ebert Mass
                M_J, M_BE = 0, 0
                L_J, L_J_cy = 0, 0
                if rho > 0:
                    M_J_val = np.pi**(5/2) * (cs_eff*1e5)**3 / (6 * cfg.GR**(3/2) * rho**(1/2))
                    M_BE_val = 1.18 * (cs_eff*1e5)**4 / (cfg.P_IC**(1/2) * cfg.GR**(3/2))
                    M_J = M_J_val / cfg.MSUN
                    M_BE = M_BE_val / cfg.MSUN
                    
                    # Jeans Lengths
                    L_J_val = np.sqrt(np.pi * (cs_eff*1e5)**2 / (cfg.GR * rho))
                    L_J_cy_val = 20 * cs_eff * 1e5 / np.sqrt(4 * np.pi * cfg.GR * rho)
                    L_J = L_J_val / cfg.PC
                    L_J_cy = L_J_cy_val / cfg.PC

                # Store results
                res = {
                    'id': i,
                    'analysis_scale_level': int(scale_level),
                    'ra': 0, 'dec': 0,
                    'x': np.mean(pts_x),
                    'y': np.mean(pts_y),
                    'v_lsr': v_mean,
                    'width': w_mean_fwhm, # FWHM
                    'sigma_v': sigma_v_1d, # dispersion
                    'cs_eff': cs_eff,
                    'filling_factor': filling_factor,
                    'matching_method_used': selection.get('matching_method_used', 'legacy_overlap_area'),
                    'spearman_rho': selection.get('spearman_rho', np.nan),
                    'spearman_pvalue': selection.get('spearman_pvalue', np.nan),
                    'spearman_n': selection.get('spearman_n', 0),
                    'spearman_valid': selection.get('spearman_valid', 0),
                    'overlap_frac_A': selection.get('overlap_frac_a', np.nan),
                    'overlap_frac_B': selection.get('overlap_frac_b', np.nan),
                    'resolution_match_flag': selection.get('resolution_match_flag', 'unknown'),
                    'area_pix': total_pixels,
                    'area_pix_b_eff': effective_b_area_pix,
                    'size_similarity': size_similarity,
                    'area': area_phys,
                    'R_eff': radius_cm,
                    'R_eff_pc': radius_pc,
                    'R_eff_au': radius_au,
                    'eccentricity': eccentricity,
                    'Mc': Mc,
                    'rho': rho,
                    'M_vir': M_vir,
                    'alpha_vir': alpha_vir,
                    'E_pot': E_pot,
                    'M_J': M_J,
                    'M_BE': M_BE,
                    'L_J_pc': L_J,
                    'L_J_cy_pc': L_J_cy
                }
                
                cx, cy = np.mean(pts_x), np.mean(pts_y)
                try:
                    if self.is_2d:
                        wc = self.structure_wcs.wcs_pix2world([[cx, cy]], 0)
                        res['ra'], res['dec'] = wc[0][0], wc[0][1]
                    else:
                        mz = 0
                        if pts_z is not None:
                            mz = np.mean(pts_z)
                        
                        wc = self.structure_wcs.wcs_pix2world([[cx, cy, mz]], 0)
                        res['ra'], res['dec'] = wc[0][0], wc[0][1]
                except:
                    pass
                
                results.append(res)
        
        df = pd.DataFrame(results)

        # Drop fields that are not requested in final CSV output.
        drop_cols = [
            'analysis_scale_level',
            'spearman_n',
            'spearman_valid',
            'overlap_frac_A',
            'overlap_frac_B',
            'resolution_match_flag',
            'area_pix',
            'area_pix_b_eff',
            'size_similarity'
        ]
        keep_cols = [c for c in df.columns if c not in drop_cols]
        df = df[keep_cols]

        # Add units to exported headers without changing internal field names.
        header_units = {
            'analysis_scale_level': '1',
            'ra': 'deg',
            'dec': 'deg',
            'x': 'pix',
            'y': 'pix',
            'v_lsr': 'km/s',
            'width': 'km/s',
            'sigma_v': 'km/s',
            'cs_eff': 'km/s',
            'filling_factor': '1',
            'matching_method_used': '1',
            'spearman_rho': '1',
            'spearman_pvalue': '1',
            'spearman_n': '1',
            'spearman_valid': '1',
            'overlap_frac_A': '1',
            'overlap_frac_B': '1',
            'resolution_match_flag': '1',
            'area_pix': 'pix',
            'area_pix_b_eff': 'pix',
            'size_similarity': '1',
            'area': 'cm^2',
            'R_eff': 'cm',
            'R_eff_pc': 'pc',
            'R_eff_au': 'au',
            'eccentricity': '1',
            'Mc': 'Msun',
            'rho': 'g/cm^3',
            'M_vir': 'Msun',
            'alpha_vir': '1',
            'E_pot': 'erg',
            'M_J': 'Msun',
            'M_BE': 'Msun',
            'L_J_pc': 'pc',
            'L_J_cy_pc': 'pc'
        }
        rename_map = {
            col: f"{col}({header_units[col]})"
            for col in df.columns
            if col in header_units
        }
        if rename_map:
            df = df.rename(columns=rename_map)

        save_csv = os.path.join(self.output_dir, 'multi_wave_properties.csv')
        df.to_csv(save_csv, index=False)
        print("structures using direct-Scouse fallback:", direct_scouse_fallback_count)
        print("total structures accepted and analyzed:", len(df))
        print(f"Properties saved to {save_csv}")
        return df

    def plot_structures(self, selected_nodes=None, scale_level=0, run_plot6=True):
        """
        Plots structures on both Dataset A (Structure) and Dataset B (Physics) backgrounds.
        Requires forest to be loaded and physics data available.
        """
        if self.forest is None:
            print("Cannot plot structures: Forest is None.")
            return

        def _fit_ellipse_from_points(x_vals, y_vals):
            if len(x_vals) < 5 or len(y_vals) < 5:
                return None
            if cv2 is None:
                raise RuntimeError("cv2 is required for ellipse fitting.")
            pts = np.vstack([x_vals, y_vals]).T.astype(np.float32)
            (xc, yc), (a_fit, b_fit), theta = cv2.fitEllipse(pts)
            #a_axis, b_axis = max(a_fit, b_fit), min(a_fit, b_fit)
            a_axis, b_axis = a_fit, b_fit
            return xc, yc, a_axis, b_axis, theta

        def _imshow_with_log_colorbar(ax, data, label):
            data = np.asarray(data, dtype=float)
            positive = np.isfinite(data) & (data > 0)

            if np.any(positive):
                positive_vals = data[positive]

                lower_pct = 25.0
                upper_pct = 99.8
                vmin = np.nanpercentile(positive_vals, lower_pct)
                vmax = np.nanpercentile(positive_vals, upper_pct)

                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin = np.nanmin(positive_vals)
                    vmax = np.nanmax(positive_vals)
                if vmax <= vmin:
                    vmax = vmin * 10.0

                data_for_plot = np.ma.masked_where(~positive, data)
                im = ax.imshow(
                    data_for_plot,
                    origin='lower',
                    cmap='gray_r',
                    interpolation='nearest',
                    norm=colors.LogNorm(vmin=vmin, vmax=vmax)
                )
                plt.colorbar(im, ax=ax, label=label)
            else:
                im = ax.imshow(data, origin='lower', cmap='gray_r', interpolation='nearest')
                plt.colorbar(im, ax=ax, label=label)

            return im

        def _annotate_structure_index(ax, x_vals, y_vals, idx, color='yellow'):
            """Place a compact index label near a structure contour."""
            if x_vals is None or y_vals is None:
                return
            if len(x_vals) == 0 or len(y_vals) == 0:
                return

            x0 = float(np.nanmean(x_vals))
            y0 = float(np.nanmean(y_vals))
            if not np.isfinite(x0) or not np.isfinite(y0):
                return

            ax.text(
                x0 + 1.0,
                y0 + 1.0,
                str(int(idx)),
                color=color,
                fontsize=6,
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.35, edgecolor='none')
            )

        def _cluster_depth_from_trunk(node, trunk):
            """
            Compute hierarchy depth for a cluster node relative to a tree trunk.
            Depth 0 means trunk, larger values are lower levels in the dendrogram.
            """
            if node is trunk:
                return 0

            depth = 0
            current = node
            visited = set()

            while current is not None and current is not trunk:
                if id(current) in visited:
                    break
                visited.add(id(current))

                parent = getattr(current, 'antecedent', None)
                if parent is None:
                    break

                depth += 1
                current = parent

            return depth

        # --- Plot 1: Structures on Dataset A (Structure) ---
        print(f"Plotting structures on Dataset A (scale={scale_level})...")
        hdul_a = fits.open(self.structure_path)
        data_a = np.squeeze(hdul_a[0].data)
        # Flatten if 3D to 2D image for background (Moment 0 or Peak)
        if data_a.ndim > 2:
            data_bg_a = np.nanmax(data_a, axis=0) # Peak intensity map
        else:
            data_bg_a = data_a
        
        # WCS A
        wcs_a = wcs.WCS(hdul_a[0].header).celestial
        hdul_a.close()

        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111, projection=wcs_a)
        
        # Background
        im1 = _imshow_with_log_colorbar(ax1, data_bg_a, 'Intensity (A)')
        
        # Build node-level list for plotting.
        # If selected_nodes is provided, plot only those nodes.
        tree_nodes_with_level = []
        seen_nodes = set()

        if selected_nodes is None:
            all_leaves = self.get_structure_nodes_for_scale(scale_level)
        else:
            all_leaves = list(selected_nodes)

        if selected_nodes is None:
            iterator = self.forest.forest.values() if isinstance(self.forest.forest, dict) else self.forest.forest
            for tree in iterator:
                if hasattr(tree, 'tree_members') and hasattr(tree, 'trunk'):
                    trunk = tree.trunk
                    members = tree.tree_members
                elif hasattr(tree, 'cluster_members'):
                    trunk = tree
                    members = [tree]
                else:
                    continue

                for node in members:
                    node_key = id(node)
                    if node_key in seen_nodes:
                        continue
                    seen_nodes.add(node_key)
                    level = _cluster_depth_from_trunk(node, trunk)
                    tree_nodes_with_level.append((node, level))
        else:
            for node in all_leaves:
                node_key = id(node)
                if node_key in seen_nodes:
                    continue
                seen_nodes.add(node_key)

                # Approximate level from root when plotting selected scale nodes.
                level = 0
                current = node
                visited = set()
                while current is not None:
                    if id(current) in visited:
                        break
                    visited.add(id(current))
                    parent = getattr(current, 'antecedent', None)
                    if parent is None:
                        break
                    level += 1
                    current = parent
                tree_nodes_with_level.append((node, level))

        ny_a, nx_a = data_bg_a.shape
        if len(tree_nodes_with_level) > 0:
            max_level = max(level for _, level in tree_nodes_with_level)
        else:
            max_level = 0

        cmap_levels = cm.get_cmap('turbo', max_level + 1 if max_level >= 1 else 2)
        level_colors = {lvl: cmap_levels(lvl) for lvl in range(max_level + 1)}

        # Draw from upper to deeper levels; deeper contours are drawn last.
        tree_nodes_with_level.sort(key=lambda item: item[1])

        for idx_plot, (node, level) in enumerate(tree_nodes_with_level):
            indices = None
            if hasattr(node, 'cluster_members'):
                indices = node.cluster_members
            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                indices = node.trunk.cluster_members

            if indices is None or self.dataarr_acorns is None:
                continue

            pts_x = self.dataarr_acorns[0, indices].astype(int)
            pts_y = self.dataarr_acorns[1, indices].astype(int)
            valid = (pts_x >= 0) & (pts_x < nx_a) & (pts_y >= 0) & (pts_y < ny_a)
            if np.sum(valid) == 0:
                continue

            mask = np.zeros((ny_a, nx_a), dtype=int)
            mask[pts_y[valid], pts_x[valid]] = 1
            x1, y1 = utils.edge_fun(mask)
            if len(x1) == 0:
                continue

            ax1.scatter(x1, y1, color=level_colors[level], marker='.', s=1)
            _annotate_structure_index(ax1, pts_x[valid], pts_y[valid], idx_plot, color='cyan')

        # Add a compact level-color legend via colorbar.
        level_norm = colors.Normalize(vmin=0, vmax=max_level if max_level > 0 else 1)
        level_sm = cm.ScalarMappable(norm=level_norm, cmap=cmap_levels)
        level_sm.set_array([])
        level_cbar = plt.colorbar(level_sm, ax=ax1, pad=0.02)
        level_cbar.set_label('Hierarchy Level (0=trunk)')

        # Keep legacy palette for downstream plots (1b/2/2b etc.).
        cmap = cm.get_cmap('hsv')

        ax1.set_title("Structures on Dataset A (All Hierarchy Levels)")
        ax1.set_xlabel("RA")
        ax1.set_ylabel("Dec")
        
        savename_a = os.path.join(self.output_dir, "map_structures_on_A.pdf")
        fig1.savefig(savename_a, dpi=300)
        plt.close(fig1)
        print(f"Saved plot: {savename_a}")

        # --- Plot 1 (Legacy): leaf-only outlines with per-structure colors ---
        print("Plotting legacy Plot1 on Dataset A...")
        fig1_legacy = plt.figure(figsize=(10, 10))
        ax1_legacy = fig1_legacy.add_subplot(111, projection=wcs_a)
        _imshow_with_log_colorbar(ax1_legacy, data_bg_a, 'Intensity (A)')

        for i, node in enumerate(all_leaves):
            indices = None
            if hasattr(node, 'cluster_members'):
                indices = node.cluster_members
            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                indices = node.trunk.cluster_members

            if indices is None or self.dataarr_acorns is None:
                continue

            pts_x = self.dataarr_acorns[0, indices].astype(int)
            pts_y = self.dataarr_acorns[1, indices].astype(int)
            if pts_x.size == 0:
                continue

            # Keep the previous local-contour rendering style for compatibility.
            pad = 5
            min_x, max_x = np.min(pts_x), np.max(pts_x)
            min_y, max_y = np.min(pts_y), np.max(pts_y)
            w = max_x - min_x + 2 * pad
            h = max_y - min_y + 2 * pad

            mask_local = np.zeros((h, w), dtype=int)
            mask_local[pts_y - min_y + pad, pts_x - min_x + pad] = 1

            color = cmap((i * 50) % 255 / 255.0)
            try:
                ax1_legacy.contour(
                    mask_local,
                    levels=[0.5],
                    colors=[color],
                    linewidths=1,
                    extent=[min_x - pad - 0.5, max_x + pad + 0.5, min_y - pad - 0.5, max_y + pad + 0.5],
                    transform=ax1_legacy.get_transform(wcs_a)
                )
            except Exception:
                ax1_legacy.contour(
                    mask_local,
                    levels=[0.5],
                    colors=[color],
                    linewidths=1,
                    extent=[min_x - pad - 0.5, max_x + pad + 0.5, min_y - pad - 0.5, max_y + pad + 0.5]
                )

            _annotate_structure_index(ax1_legacy, pts_x, pts_y, i)

        ax1_legacy.set_title("Structures on Dataset A (hierarchical ends)")
        ax1_legacy.set_xlabel("RA")
        ax1_legacy.set_ylabel("Dec")

        savename_a_legacy = os.path.join(self.output_dir, "map_structures_on_A_legacy.pdf")
        fig1_legacy.savefig(savename_a_legacy, dpi=300)
        plt.close(fig1_legacy)
        print(f"Saved plot: {savename_a_legacy}")

        # --- Plot 1b: Ellipse outlines on Dataset A ---
        print("Plotting ellipse outlines on Dataset A...")
        fig1b = plt.figure(figsize=(10, 10))
        ax1b = fig1b.add_subplot(111, projection=wcs_a)
        im1b = _imshow_with_log_colorbar(ax1b, data_bg_a, 'Intensity (A)')

        for i, node in enumerate(all_leaves):
            indices = None
            if hasattr(node, 'cluster_members'):
                indices = node.cluster_members
            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                indices = node.trunk.cluster_members

            pts_x, pts_y = None, None
            if indices is not None and hasattr(self, 'dataarr_acorns') and self.dataarr_acorns is not None:
                pts_x = self.dataarr_acorns[0, indices].astype(int)
                pts_y = self.dataarr_acorns[1, indices].astype(int)

            if pts_x is None or len(pts_x) == 0:
                continue

            fit = _fit_ellipse_from_points(pts_x, pts_y)
            if fit is None:
                continue

            xc, yc, a_axis, b_axis, theta = fit
            color = cmap((i * 50) % 255 / 255.0)

            try:
                e = Ellipse(
                    (xc, yc),
                    width=a_axis,
                    height=b_axis,
                    angle=theta,
                    fill=False,
                    edgecolor=color,
                    linewidth=1,
                    transform=ax1b.get_transform(wcs_a)
                )
                ax1b.add_patch(e)
            except Exception:
                e = Ellipse(
                    (xc, yc),
                    width=a_axis,
                    height=b_axis,
                    angle=theta,
                    fill=False,
                    edgecolor=color,
                    linewidth=1
                )
                ax1b.add_patch(e)

        ax1b.set_title("Ellipse Outlines on Dataset A")
        ax1b.set_xlabel("RA")
        ax1b.set_ylabel("Dec")

        savename_a_ell = os.path.join(self.output_dir, "map_ellipses_on_A.pdf")
        fig1b.savefig(savename_a_ell, dpi=300)
        plt.close(fig1b)
        print(f"Saved plot: {savename_a_ell}")

        # --- Plot 1c: ACORNS dendrogram for structure forest ---
        print("Plotting ACORNS dendrogram for structure forest...")
        try:
            fig_den = plt.figure(figsize=(8, 8))
            ax_den = fig_den.add_subplot(111)

            # Match example style while keeping axis limits data-driven.
            ax_den.set_xlabel('Structure index')
            ax_den.set_ylabel('Cluster value')

            if hasattr(self, 'dataarr_acorns') and self.dataarr_acorns is not None and self.dataarr_acorns.shape[1] > 0:
                baseline = float(np.nanmean(self.dataarr_acorns[2, :]))
                if not np.isfinite(baseline):
                    baseline = 0.0
            else:
                baseline = 0.0

            if isinstance(self.forest.forest, dict):
                tree_ids = list(self.forest.forest.keys())
                get_tree = lambda tid: self.forest.forest[tid]
            else:
                tree_ids = list(range(len(self.forest.forest)))
                get_tree = lambda tid: self.forest.forest[tid]

            # Build a global level colormap for dendrogram segments.
            max_level_den = 0
            for tree_id in tree_ids:
                tree = get_tree(tree_id)
                if not hasattr(tree, 'tree_members') or not hasattr(tree, 'trunk'):
                    continue
                for member in tree.tree_members:
                    max_level_den = max(max_level_den, _cluster_depth_from_trunk(member, tree.trunk))

            cmap_levels_den = cm.get_cmap('turbo', max_level_den + 1 if max_level_den >= 1 else 2)
            level_colors_den = {lvl: cmap_levels_den(lvl) for lvl in range(max_level_den + 1)}

            count = 0.0

            for tree_id in tree_ids:
                tree = get_tree(tree_id)

                if not hasattr(tree, 'tree_members') or not hasattr(tree, 'cluster_vertices') or not hasattr(tree, 'horizontals'):
                    continue

                for j in range(len(tree.tree_members)):
                    member = tree.tree_members[j]
                    level = _cluster_depth_from_trunk(member, tree.trunk) if hasattr(tree, 'trunk') else 0
                    color_level = level_colors_den.get(level, cmap_levels_den(0))

                    if hasattr(tree, 'trunk') and tree.tree_members[j] == tree.trunk:
                        ax_den.plot(
                            tree.cluster_vertices[0][j] + count,
                            np.array([baseline, tree.cluster_vertices[1][j][0]]),
                            'k:'
                        )

                    ax_den.plot(tree.cluster_vertices[0][j] + count, tree.cluster_vertices[1][j], c=color_level)
                    ax_den.plot(tree.horizontals[0][j] + count, tree.horizontals[1][j], c=color_level)

                if hasattr(tree, 'leaves'):
                    count += len(tree.leaves)

            level_norm_den = colors.Normalize(vmin=0, vmax=max_level_den if max_level_den > 0 else 1)
            level_sm_den = cm.ScalarMappable(norm=level_norm_den, cmap=cmap_levels_den)
            level_sm_den.set_array([])
            cbar_den = plt.colorbar(level_sm_den, ax=ax_den, pad=0.02)
            cbar_den.set_label('Hierarchy Level (0=trunk)')

            ax_den.set_title('ACORNS Dendrogram (Structure Forest)')
            savename_den = os.path.join(self.output_dir, 'dendrogram_structure.pdf')
            fig_den.savefig(savename_den, dpi=300)
            plt.close(fig_den)
            print(f"Saved plot: {savename_den}")
        except Exception as e:
            print(f"Warning: failed to plot structure dendrogram: {e}")


        # --- Plot 2: Structures on Dataset B (Physics) ---
        if self.physics_fits_path and os.path.exists(self.physics_fits_path):
            print("Plotting structures on Dataset B...")
            hdul_b = fits.open(self.physics_fits_path)
            header_b = hdul_b[0].header
            data_b = np.squeeze(hdul_b[0].data)
            wcs_b = wcs.WCS(header_b).celestial
             # Flatten B if 3D
            if data_b.ndim > 2:
                data_bg_b = np.nanmax(data_b, axis=0)
            else:
                data_bg_b = data_b
            hdul_b.close()
            
            nx_b = header_b.get('NAXIS1')
            ny_b = header_b.get('NAXIS2')

            fig2 = plt.figure(figsize=(10, 10))
            ax2 = fig2.add_subplot(111, projection=wcs_b)
            
            im2 = _imshow_with_log_colorbar(ax2, data_bg_b, 'Intensity (B)')
            
            # For B, we must transform coordinates from A -> B
            # We already have robust A->B mapping logic in loop
            
            # WCS objects
            wcs_a = self.structure_wcs # Full WCS A
            try:
                wcs_a_celestial = wcs_a.celestial
            except Exception:
                wcs_a_celestial = wcs_a
            
            for i, node in enumerate(all_leaves):
                indices = None
                if hasattr(node, 'cluster_members'): indices = node.cluster_members
                elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'): indices = node.trunk.cluster_members
                
                pts_x, pts_y = None, None
                if indices is not None and hasattr(self, 'dataarr_acorns') and self.dataarr_acorns is not None:
                     pts_x = self.dataarr_acorns[0, indices].astype(int)
                     pts_y = self.dataarr_acorns[1, indices].astype(int)
                
                if pts_x is None: continue

                a_pixels = np.unique(np.column_stack((pts_x, pts_y)).astype(int), axis=0)
                _, b_coverage = self._build_overlap_maps_a_to_b(
                    a_pixels,
                    wcs_a_celestial=wcs_a_celestial,
                    wcs_b_celestial=wcs_b,
                    nx_b=nx_b,
                    ny_b=ny_b
                )

                b_pixels = np.array(
                    [k for k, frac_b in b_coverage.items() if frac_b >= self.mapping_min_overlap],
                    dtype=int
                )
                if b_pixels.size == 0:
                    continue

                mask_b = np.zeros((ny_b, nx_b), dtype=int)
                mask_b[b_pixels[:, 1], b_pixels[:, 0]] = 1

                x1, y1 = utils.edge_fun(mask_b)
                if len(x1) == 0:
                    continue
                
                color = cmap((i * 50) % 255 / 255.0)
                ax2.scatter(x1, y1, color=color, marker='.', s=1)
                _annotate_structure_index(ax2, b_pixels[:, 0], b_pixels[:, 1], i)
            
            ax2.set_title("Structures Mapped on Dataset B")
            ax2.set_xlabel("RA")
            ax2.set_ylabel("Dec")
            
            savename_b = os.path.join(self.output_dir, "map_structures_on_B.pdf")
            fig2.savefig(savename_b, dpi=300)
            plt.close(fig2)
            print(f"Saved plot: {savename_b}")

            # --- Plot 2b: Ellipse outlines on Dataset B ---
            print("Plotting ellipse outlines on Dataset B...")
            fig2b = plt.figure(figsize=(10, 10))
            ax2b = fig2b.add_subplot(111, projection=wcs_b)
            im2b = _imshow_with_log_colorbar(ax2b, data_bg_b, 'Intensity (B)')

            for i, node in enumerate(all_leaves):
                indices = None
                if hasattr(node, 'cluster_members'):
                    indices = node.cluster_members
                elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                    indices = node.trunk.cluster_members

                pts_x, pts_y = None, None
                if indices is not None and hasattr(self, 'dataarr_acorns') and self.dataarr_acorns is not None:
                    pts_x = self.dataarr_acorns[0, indices].astype(int)
                    pts_y = self.dataarr_acorns[1, indices].astype(int)

                if pts_x is None or len(pts_x) == 0:
                    continue

                a_pixels = np.unique(np.column_stack((pts_x, pts_y)).astype(int), axis=0)
                _, b_coverage = self._build_overlap_maps_a_to_b(
                    a_pixels,
                    wcs_a_celestial=wcs_a_celestial,
                    wcs_b_celestial=wcs_b,
                    nx_b=nx_b,
                    ny_b=ny_b
                )

                b_pixels = np.array(
                    [k for k, frac_b in b_coverage.items() if frac_b >= self.mapping_min_overlap],
                    dtype=int
                )
                if b_pixels.size == 0:
                    continue

                bx = b_pixels[:, 0]
                by = b_pixels[:, 1]

                fit = _fit_ellipse_from_points(bx, by)
                if fit is None:
                    continue

                xc, yc, a_axis, b_axis, theta = fit
                color = cmap((i * 50) % 255 / 255.0)

                try:
                    e = Ellipse(
                        (xc, yc),
                        width=a_axis,
                        height=b_axis,
                        angle=theta,
                        fill=False,
                        edgecolor=color,
                        linewidth=1
                    )
                    ax2b.add_patch(e)
                except Exception:
                    e = Ellipse(
                        (xc, yc),
                        width=a_axis,
                        height=b_axis,
                        angle=theta,
                        fill=False,
                        edgecolor=color,
                        linewidth=1
                    )
                    ax2b.add_patch(e)

            ax2b.set_title("Ellipse Outlines on Dataset B")
            ax2b.set_xlabel("RA")
            ax2b.set_ylabel("Dec")

            savename_b_ell = os.path.join(self.output_dir, "map_ellipses_on_B.pdf")
            fig2b.savefig(savename_b_ell, dpi=300)
            plt.close(fig2b)
            print(f"Saved plot: {savename_b_ell}")

            # --- Plot 3: Physics structures on Dataset B ---
            if self.physics_forest is not None and self.physics_dataarr_acorns is not None:
                print("Plotting physics structures on Dataset B...")
                fig3 = plt.figure(figsize=(10, 10))
                ax3 = fig3.add_subplot(111, projection=wcs_b)
                im3 = _imshow_with_log_colorbar(ax3, data_bg_b, 'Intensity (B)')

                # Filter bright trees
                bright_trees_b = []
                iterator_b = self.physics_forest.forest.values() if isinstance(self.physics_forest.forest, dict) else self.physics_forest.forest
                tree_list_b = list(iterator_b)
                for tree in tree_list_b:
                    if hasattr(tree, 'cluster_vertices') and np.max(tree.cluster_vertices[1]) > 0.1:
                        bright_trees_b.append(tree)

                if len(bright_trees_b) == 0:
                    print("No bright B trees found with threshold 0.24; plotting all B trees for diagnostics.")
                    bright_trees_b = tree_list_b

                colour_bright_b = iter(cm.rainbow(np.linspace(0, 1, len(bright_trees_b) if len(bright_trees_b) > 0 else 1)))
                
                for tree in bright_trees_b:
                    co = next(colour_bright_b)
                    leaves = tree.leaves if hasattr(tree, 'leaves') else [tree]
                    for blade in leaves:
                        indices = None
                        if hasattr(blade, 'cluster_members'):
                            indices = blade.cluster_members
                        elif hasattr(blade, 'trunk') and hasattr(blade.trunk, 'cluster_members'):
                            indices = blade.trunk.cluster_members
                        
                        if indices is None or len(indices) == 0:
                            continue
                            
                        pts_x = self.physics_dataarr_acorns[0, indices].astype(int)
                        pts_y = self.physics_dataarr_acorns[1, indices].astype(int)
                        
                        # Create mask on B
                        mask = np.zeros((ny_b, nx_b), dtype=int)
                        valid = (pts_x >= 0) & (pts_x < nx_b) & (pts_y >= 0) & (pts_y < ny_b)
                        mask[pts_y[valid], pts_x[valid]] = 1
                        #mask[np.rint(pts_y[valid]/2).astype(int)-1, np.rint(pts_x[valid]/2).astype(int)-1] = 1
                        
                        x1, y1 = utils.edge_fun(mask)
                        ax3.scatter(x1, y1, color=co, marker='.', s=1)

                ax3.set_title("Physics Structures on Dataset B")
                ax3.set_xlabel("RA")
                ax3.set_ylabel("Dec")

                savename_b_phys = os.path.join(self.output_dir, "map_physics_structures_on_B.pdf")
                fig3.savefig(savename_b_phys, dpi=300)
                plt.close(fig3)
                print(f"Saved plot: {savename_b_phys}")

            # --- Plot 4: Comparison of A and B structures on Dataset A ---
            if self.physics_forest is not None and self.physics_dataarr_acorns is not None:
                print("Plotting comparison of A and B structures on Dataset A...")
                fig4 = plt.figure(figsize=(10, 10))
                ax4 = fig4.add_subplot(111, projection=wcs_a.celestial)
                im4 = _imshow_with_log_colorbar(ax4, data_bg_a, 'Intensity (A)')

                ny_a, nx_a = data_bg_a.shape

                # Plot A structures (Red)
                bright_trees_a = []
                iterator_a = self.forest.forest.values() if isinstance(self.forest.forest, dict) else self.forest.forest
                for tree in iterator_a:
                    if hasattr(tree, 'cluster_vertices') and np.max(tree.cluster_vertices[1]) > 0.24:
                        bright_trees_a.append(tree)

                for tree in bright_trees_a:
                    leaves = tree.leaves if hasattr(tree, 'leaves') else [tree]
                    for blade in leaves:
                        indices = None
                        if hasattr(blade, 'cluster_members'):
                            indices = blade.cluster_members
                        elif hasattr(blade, 'trunk') and hasattr(blade.trunk, 'cluster_members'):
                            indices = blade.trunk.cluster_members
                        
                        if indices is None or len(indices) == 0:
                            continue
                            
                        pts_x = self.dataarr_acorns[0, indices].astype(int)
                        pts_y = self.dataarr_acorns[1, indices].astype(int)
                        
                        mask = np.zeros((ny_a, nx_a), dtype=int)
                        valid = (pts_x >= 0) & (pts_x < nx_a) & (pts_y >= 0) & (pts_y < ny_a)
                        mask[pts_y[valid], pts_x[valid]] = 1
                        
                        x1, y1 = utils.edge_fun(mask)
                        ax4.scatter(x1, y1, color='red', marker='.', s=1)

                # Plot B structures (Blue)
                for tree in bright_trees_b:
                    leaves = tree.leaves if hasattr(tree, 'leaves') else [tree]
                    for blade in leaves:
                        indices = None
                        if hasattr(blade, 'cluster_members'):
                            indices = blade.cluster_members
                        elif hasattr(blade, 'trunk') and hasattr(blade.trunk, 'cluster_members'):
                            indices = blade.trunk.cluster_members
                        
                        if indices is None or len(indices) == 0:
                            continue
                            
                        pts_x = self.physics_dataarr_acorns[0, indices].astype(int)
                        pts_y = self.physics_dataarr_acorns[1, indices].astype(int)

                        # 1) Get contour points on B first
                        mask_b = np.zeros((ny_b, nx_b), dtype=int)
                        valid_b = (pts_x >= 0) & (pts_x < nx_b) & (pts_y >= 0) & (pts_y < ny_b)
                        if np.sum(valid_b) == 0:
                            continue
                        mask_b[pts_y[valid_b], pts_x[valid_b]] = 1

                        edge_x_b, edge_y_b = utils.edge_fun(mask_b)
                        if len(edge_x_b) == 0:
                            continue

                        # 2) Convert contour points B -> A
                        ra, dec = wcs_b.wcs_pix2world(edge_x_b, edge_y_b, 0)
                        ax_edge, ay_edge = wcs_a.celestial.wcs_world2pix(ra, dec, 0)
                        ax_edge = np.round(ax_edge).astype(int)
                        ay_edge = np.round(ay_edge).astype(int)

                        # 3) Draw only converted contour points on A
                        valid_a = (ax_edge >= 0) & (ax_edge < nx_a) & (ay_edge >= 0) & (ay_edge < ny_a)
                        if np.sum(valid_a) == 0:
                            continue

                        ax4.scatter(ax_edge[valid_a], ay_edge[valid_a], color='blue', marker='.', s=1)

                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Structure FITS (A)', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Physics FITS (B)', markersize=8)
                ]
                ax4.legend(handles=legend_elements, loc='upper right')

                ax4.set_title("Comparison of A and B Structures on Dataset A")
                ax4.set_xlabel("RA")
                ax4.set_ylabel("Dec")

                savename_comp = os.path.join(self.output_dir, "map_comparison_A_and_B_on_A.pdf")
                fig4.savefig(savename_comp, dpi=300)
                plt.close(fig4)
                print(f"Saved plot: {savename_comp}")

            # --- Plot 7: Comparison on A using only non-fallback A<->B selections ---
            if self.physics_forest is not None and self.physics_dataarr_acorns is not None:
                print("Plotting Plot 7 (non-fallback A/B structures only) on Dataset A...")
                fig7 = plt.figure(figsize=(10, 10))
                ax7 = fig7.add_subplot(111, projection=wcs_a.celestial)
                _imshow_with_log_colorbar(ax7, data_bg_a, 'Intensity (A)')

                ny_a, nx_a = data_bg_a.shape
                used_count = 0
                b_leaf_color_cache = {}
                plotted_b_leaf_ids = set()
                plot7_loop_dir = os.path.join(self.output_dir, 'plot7_loops')
                os.makedirs(plot7_loop_dir, exist_ok=True)
                saved_loop_count = 0
                involved_b_leaf_ids = set()
                b_leaf_a_masks = {}
                b_leaf_b_points_on_a = {}
                b_leaf_velocity_text = {}

                # Build leaf_id -> B leaf node mapping using the same ordering as _index_physics_leaves.
                physics_leaf_nodes = {}
                leaf_id_counter = 0
                iterator_b7 = self.physics_forest.forest.values() if isinstance(self.physics_forest.forest, dict) else self.physics_forest.forest
                for tree_b in iterator_b7:
                    if hasattr(tree_b, 'leaves'):
                        nodes_b = tree_b.leaves
                    elif hasattr(tree_b, 'cluster_members'):
                        nodes_b = [tree_b]
                    else:
                        nodes_b = []
                    for leaf_node in nodes_b:
                        physics_leaf_nodes[leaf_id_counter] = leaf_node
                        leaf_id_counter += 1

                # Build count-based color mapping to maximize color separation
                # between nearby leaf IDs (instead of using raw leaf_id values).
                ordered_leaf_ids = sorted(physics_leaf_nodes.keys())
                n_leaf_colors = max(1, len(ordered_leaf_ids))
                for i_color, leaf_id in enumerate(ordered_leaf_ids):
                    if n_leaf_colors == 1:
                        color_value = 0.6
                    else:
                        color_value = i_color / float(n_leaf_colors - 1)
                    b_leaf_color_cache[leaf_id] = cm.viridis(color_value)

                for node_idx, node in enumerate(all_leaves):
                    indices = None
                    if hasattr(node, 'cluster_members'):
                        indices = node.cluster_members
                    elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                        indices = node.trunk.cluster_members

                    if indices is None or len(indices) == 0 or self.dataarr_acorns is None:
                        continue

                    ax_pts = self.dataarr_acorns[0, indices].astype(int)
                    ay_pts = self.dataarr_acorns[1, indices].astype(int)
                    if len(ax_pts) == 0:
                        continue

                    az_pts = None
                    if self.dataarr_acorns.shape[0] >= 5:
                        az_pts = self.dataarr_acorns[4, indices]

                    selection = self._select_physics_components_for_structure(
                        pts_x=ax_pts,
                        pts_y=ay_pts,
                        pts_z=az_pts,
                        wcs_a_celestial=wcs_a_celestial,
                        wcs_b_celestial=wcs_b,
                        nx_b=nx_b,
                        ny_b=ny_b
                    )
                    if selection is None:
                        continue
                    if bool(selection.get('used_fallback', False)):
                        continue

                    selected_component_indices = selection.get('selected_component_indices', [])
                    if selected_component_indices is None or len(selected_component_indices) == 0:
                        continue

                    selected_leaf_ids = {
                        self.physics_component_to_leaf[idx]
                        for idx in selected_component_indices
                        if idx in self.physics_component_to_leaf
                    }
                    if len(selected_leaf_ids) == 0:
                        continue

                    # Build A mask once for this loop and draw A contour only.
                    mask_a = np.zeros((ny_a, nx_a), dtype=bool)
                    valid_a = (ax_pts >= 0) & (ax_pts < nx_a) & (ay_pts >= 0) & (ay_pts < ny_a)
                    if np.sum(valid_a) == 0:
                        continue
                    mask_a[ay_pts[valid_a], ax_pts[valid_a]] = True

                    edge_x_a, edge_y_a = utils.edge_fun(mask_a.astype(int))
                    if len(edge_x_a) > 0:
                        ax7.scatter(edge_x_a, edge_y_a, color='red', marker='.', s=1, alpha=1)

                    # Draw all B leaf pixels (mapped onto A) with leaf-dependent colors.
                    plotted_b_any = False
                    for leaf_id in selected_leaf_ids:
                        involved_b_leaf_ids.add(leaf_id)
                        if leaf_id not in b_leaf_a_masks:
                            b_leaf_a_masks[leaf_id] = np.zeros((ny_a, nx_a), dtype=bool)
                        b_leaf_a_masks[leaf_id] |= mask_a

                        # Skip if this B leaf structure has already been drawn in previous loops.
                        if leaf_id in plotted_b_leaf_ids:
                            continue

                        b_leaf_color = b_leaf_color_cache.get(leaf_id, cm.viridis(0.6))

                        leaf_node = physics_leaf_nodes.get(leaf_id)
                        if leaf_node is None:
                            continue

                        b_indices = None
                        if hasattr(leaf_node, 'cluster_members'):
                            b_indices = leaf_node.cluster_members
                        elif hasattr(leaf_node, 'trunk') and hasattr(leaf_node.trunk, 'cluster_members'):
                            b_indices = leaf_node.trunk.cluster_members

                        if b_indices is None or len(b_indices) == 0:
                            continue

                        bx_full = self.physics_dataarr_acorns[0, b_indices].astype(int)
                        by_full = self.physics_dataarr_acorns[1, b_indices].astype(int)
                        if len(bx_full) == 0:
                            continue

                        mask_b = np.zeros((ny_b, nx_b), dtype=int)
                        valid_b = (bx_full >= 0) & (bx_full < nx_b) & (by_full >= 0) & (by_full < ny_b)
                        if np.sum(valid_b) == 0:
                            continue
                        mask_b[by_full[valid_b], bx_full[valid_b]] = 1

                        b_y_all, b_x_all = np.where(mask_b == 1)
                        if len(b_x_all) == 0:
                            continue

                        ra, dec = wcs_b.wcs_pix2world(b_x_all, b_y_all, 0)
                        ax_edge, ay_edge = wcs_a.celestial.wcs_world2pix(ra, dec, 0)
                        ax_edge = np.round(ax_edge).astype(int)
                        ay_edge = np.round(ay_edge).astype(int)
                        valid_a2 = (ax_edge >= 0) & (ax_edge < nx_a) & (ay_edge >= 0) & (ay_edge < ny_a)
                        if np.sum(valid_a2) == 0:
                            continue

                        ax7.scatter(ax_edge[valid_a2], ay_edge[valid_a2], color=b_leaf_color, marker='s', s=3, alpha=0.4)
                        b_leaf_b_points_on_a[leaf_id] = (ax_edge[valid_a2], ay_edge[valid_a2], b_leaf_color)
                        plotted_b_any = True
                        plotted_b_leaf_ids.add(leaf_id)

                        if self.physics_dataarr_acorns.shape[0] >= 5:
                            v_vals = self.physics_dataarr_acorns[4, b_indices].astype(float)
                            v_vals = v_vals[np.isfinite(v_vals)]
                            if v_vals.size > 0:
                                b_leaf_velocity_text[leaf_id] = (
                                    f"leaf {leaf_id}: {np.min(v_vals):.2f} to {np.max(v_vals):.2f} km/s"
                                )

                    if plotted_b_any:
                        used_count += 1

                # Save one image per involved B structure (leaf), with all associated A contours overlaid.
                for leaf_id in sorted(involved_b_leaf_ids):
                    fig7_loop = plt.figure(figsize=(10, 10))
                    ax7_loop = fig7_loop.add_subplot(111, projection=wcs_a.celestial)
                    _imshow_with_log_colorbar(ax7_loop, data_bg_a, 'Intensity (A)')

                    if leaf_id in b_leaf_b_points_on_a:
                        px, py, pcolor = b_leaf_b_points_on_a[leaf_id]
                        ax7_loop.scatter(px, py, color=pcolor, marker='s', s=3, alpha=0.4)

                    a_mask_leaf = b_leaf_a_masks.get(leaf_id)
                    if a_mask_leaf is not None:
                        edge_x_a, edge_y_a = utils.edge_fun(a_mask_leaf.astype(int))
                        if len(edge_x_a) > 0:
                            ax7_loop.scatter(edge_x_a, edge_y_a, color='red', marker='s', s=3, alpha=1)

                    range_text = b_leaf_velocity_text.get(leaf_id, f"leaf {leaf_id}: velocity n/a")
                    ax7_loop.text(
                        0.02,
                        0.98,
                        "B velocity range\n" + range_text,
                        transform=ax7_loop.transAxes,
                        va='top',
                        ha='left',
                        fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75)
                    )

                    ax7_loop.set_title(f"Plot 7 B Leaf {leaf_id} (Non-fallback)")
                    ax7_loop.set_xlabel("RA")
                    ax7_loop.set_ylabel("Dec")
                    loop_save = os.path.join(plot7_loop_dir, f"plot7_leaf_{leaf_id:04d}.pdf")
                    fig7_loop.savefig(loop_save, dpi=300)
                    plt.close(fig7_loop)
                    saved_loop_count += 1

                from matplotlib.lines import Line2D
                legend_elements7 = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='A (non-fallback)', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=cm.viridis(0.6), label='B mapped by leaf color (non-fallback)', markersize=8)
                ]
                ax7.legend(handles=legend_elements7, loc='upper right')

                ax7.set_title("Comparison of A and B Structures on A (Non-fallback Only)")
                ax7.set_xlabel("RA")
                ax7.set_ylabel("Dec")

                savename_comp_nf = os.path.join(self.output_dir, "map_comparison_A_and_B_on_A_non_fallback.pdf")
                fig7.savefig(savename_comp_nf, dpi=300)
                plt.close(fig7)
                print(f"Saved plot: {savename_comp_nf} (structures plotted: {used_count})")
                print(f"Saved per-loop plot7 images: {saved_loop_count} in {plot7_loop_dir}")

            # --- Plot 8: B-anchored panels with matched A overlays ---
            if bool(getattr(cfg, 'B_ANCHORED_PLOTS_ENABLED', True)) and self.physics_forest is not None:
                print("Plotting B-anchored subplot pages...")

                ny_a, nx_a = data_bg_a.shape

                def _draw_a_structure_outline_on_b(ax, pts_x, pts_y, color='red', s=1, alpha=1.0):
                    if pts_x is None or pts_y is None or len(pts_x) == 0:
                        return
                    mask_a_local = np.zeros((ny_a, nx_a), dtype=int)
                    valid_a_local = (pts_x >= 0) & (pts_x < nx_a) & (pts_y >= 0) & (pts_y < ny_a)
                    if np.sum(valid_a_local) == 0:
                        return
                    mask_a_local[pts_y[valid_a_local], pts_x[valid_a_local]] = 1
                    ex_a, ey_a = utils.edge_fun(mask_a_local)
                    if len(ex_a) == 0:
                        return

                    try:
                        ra, dec = wcs_a_celestial.wcs_pix2world(ex_a, ey_a, 0)
                        bx, by = wcs_b.wcs_world2pix(ra, dec, 0)
                        bx = np.round(bx).astype(int)
                        by = np.round(by).astype(int)
                        valid_b_local = (bx >= 0) & (bx < nx_b) & (by >= 0) & (by < ny_b)
                        if np.sum(valid_b_local) == 0:
                            return
                        ax.scatter(bx[valid_b_local], by[valid_b_local], color=color, marker='.', s=s, alpha=alpha)
                    except Exception:
                        pass

                # Build transformed A-global contour grid once.
                contour_levels = []
                try:
                    perc_cfg = list(getattr(cfg, 'B_ANCHORED_A_CONTOUR_PERCENTILES', [70, 85, 95]))
                    vals = np.asarray(data_bg_a, dtype=float)
                    vals_fin = vals[np.isfinite(vals)]
                    if vals_fin.size > 0:
                        contour_levels = sorted({
                            float(np.nanpercentile(vals_fin, float(p)))
                            for p in perc_cfg
                            if np.isfinite(p)
                        })
                        contour_levels = [lv for lv in contour_levels if np.isfinite(lv)]
                except Exception:
                    contour_levels = []

                yy_a_grid, xx_a_grid = np.indices(data_bg_a.shape)
                bx_grid, by_grid = None, None
                if len(contour_levels) > 0:
                    try:
                        ra_grid, dec_grid = wcs_a_celestial.wcs_pix2world(xx_a_grid, yy_a_grid, 0)
                        bx_grid, by_grid = wcs_b.wcs_world2pix(ra_grid, dec_grid, 0)
                    except Exception:
                        bx_grid, by_grid = None, None

                def _draw_a_global_contours_on_b(ax):
                    if bx_grid is None or by_grid is None or len(contour_levels) == 0:
                        return
                    try:
                        ax.contour(
                            bx_grid,
                            by_grid,
                            data_bg_a,
                            levels=contour_levels,
                            colors='cyan',
                            linewidths=0.6,
                            alpha=0.5
                        )
                    except Exception:
                        pass

                from matplotlib.lines import Line2D

                def _build_plot8_legend_handles(n_matched, n_unmatched, include_global_contour=True):
                    handles = [
                        Line2D([0], [0], marker='o', linestyle='None', color='red', markersize=5,
                               label=f'A matched (N={int(n_matched)})'),
                        Line2D([0], [0], marker='o', linestyle='None', color='yellow', markersize=5,
                               label=f'A unmatched (N={int(n_unmatched)})')
                    ]
                    if include_global_contour:
                        handles.append(
                            Line2D([0], [0], color='cyan', linewidth=1.0, alpha=0.7, label='A global contour')
                        )
                    return handles

                a_node_points = {}
                matched_a_node_indices = set()

                def _draw_a_nodes_on_b(ax, node_indices, color, s=1.0, alpha=0.95):
                    drawn = 0
                    for node_idx in node_indices:
                        pts = a_node_points.get(int(node_idx))
                        if pts is None:
                            continue
                        _draw_a_structure_outline_on_b(
                            ax,
                            pts['pts_x'],
                            pts['pts_y'],
                            color=color,
                            s=s,
                            alpha=alpha
                        )
                        drawn += 1
                    return drawn

                def _draw_overview_panel(ax, draw_split=False, add_legend=False):
                    ax.imshow(data_bg_b, origin='lower', cmap='gray_r', interpolation='nearest')
                    _draw_a_global_contours_on_b(ax)

                    if draw_split:
                        matched_sorted = sorted(matched_a_node_indices)
                        unmatched_sorted = sorted(
                            idx for idx in a_node_points.keys() if idx not in matched_a_node_indices
                        )

                        n_match_drawn = _draw_a_nodes_on_b(ax, matched_sorted, color='red', s=1.0, alpha=0.95)
                        n_unmatch_drawn = _draw_a_nodes_on_b(ax, unmatched_sorted, color='yellow', s=1.0, alpha=0.95)
                        ax.set_title(
                            f'Overview: B background + A contour + A matched/unmatched '
                            f'| matched={n_match_drawn} | unmatched={n_unmatch_drawn}'
                        )

                        if add_legend:
                            ax.legend(
                                handles=_build_plot8_legend_handles(n_match_drawn, n_unmatch_drawn, include_global_contour=True),
                                loc='upper right',
                                framealpha=0.8,
                                fontsize=8
                            )
                    else:
                        for node in all_leaves:
                            indices = None
                            if hasattr(node, 'cluster_members'):
                                indices = node.cluster_members
                            elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                                indices = node.trunk.cluster_members
                            if indices is None or len(indices) == 0:
                                continue
                            pts_x_all = self.dataarr_acorns[0, indices].astype(int)
                            pts_y_all = self.dataarr_acorns[1, indices].astype(int)
                            _draw_a_structure_outline_on_b(
                                ax,
                                pts_x_all,
                                pts_y_all,
                                color='orange',
                                s=1,
                                alpha=0.95
                            )
                        ax.set_title('Overview: B background + A contour + all A structures')

                    ax.set_xlim(0, nx_b - 1)
                    ax.set_ylim(0, ny_b - 1)

                # Build leaf_id -> B leaf node mapping (same ordering logic as _index_physics_leaves).
                physics_leaf_nodes = {}
                leaf_id_counter = 0
                iterator_b8 = self.physics_forest.forest.values() if isinstance(self.physics_forest.forest, dict) else self.physics_forest.forest
                for tree_b in iterator_b8:
                    if hasattr(tree_b, 'leaves'):
                        nodes_b = tree_b.leaves
                    elif hasattr(tree_b, 'cluster_members'):
                        nodes_b = [tree_b]
                    else:
                        nodes_b = []
                    for leaf_node in nodes_b:
                        physics_leaf_nodes[leaf_id_counter] = leaf_node
                        leaf_id_counter += 1

                # Build B leaf -> matched A structures.
                leaf_to_a_entries = {}
                for node_idx, node in enumerate(all_leaves):
                    indices = None
                    if hasattr(node, 'cluster_members'):
                        indices = node.cluster_members
                    elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                        indices = node.trunk.cluster_members

                    if indices is None or len(indices) == 0 or self.dataarr_acorns is None:
                        continue

                    ax_pts = self.dataarr_acorns[0, indices].astype(int)
                    ay_pts = self.dataarr_acorns[1, indices].astype(int)
                    if len(ax_pts) == 0:
                        continue

                    a_node_points[int(node_idx)] = {
                        'pts_x': ax_pts,
                        'pts_y': ay_pts
                    }

                    az_pts = None
                    ai_pts = None
                    if self.dataarr_acorns.shape[0] >= 3:
                        ai_pts = self.dataarr_acorns[2, indices].astype(float)
                    if self.dataarr_acorns.shape[0] >= 5:
                        az_pts = self.dataarr_acorns[4, indices]

                    selection = self._select_physics_components_for_structure(
                        pts_x=ax_pts,
                        pts_y=ay_pts,
                        pts_z=az_pts,
                        pts_i=ai_pts,
                        wcs_a_celestial=wcs_a_celestial,
                        wcs_b_celestial=wcs_b,
                        nx_b=nx_b,
                        ny_b=ny_b
                    )
                    if selection is None:
                        continue

                    # For per-B subplots, keep only non-fallback A<->B matches.
                    if bool(selection.get('used_fallback', False)):
                        continue

                    comp_indices = selection.get('selected_component_indices', [])
                    if comp_indices is None or len(comp_indices) == 0:
                        continue

                    selected_leaf_ids = {
                        self.physics_component_to_leaf[idx]
                        for idx in comp_indices
                        if idx in self.physics_component_to_leaf
                    }
                    if len(selected_leaf_ids) == 0:
                        continue

                    matched_a_node_indices.add(int(node_idx))

                    for leaf_id in selected_leaf_ids:
                        if leaf_id not in leaf_to_a_entries:
                            leaf_to_a_entries[leaf_id] = []
                        leaf_to_a_entries[leaf_id].append({
                            'node_idx': int(node_idx),
                            'pts_x': ax_pts,
                            'pts_y': ay_pts,
                            'spearman_rho': selection.get('spearman_rho', np.nan)
                        })

                matched_leaf_ids = [
                    lid for lid in sorted(leaf_to_a_entries.keys())
                    if lid in physics_leaf_nodes and len(leaf_to_a_entries.get(lid, [])) > 0
                ]

                unmatched_a_node_indices = sorted(
                    idx for idx in a_node_points.keys() if idx not in matched_a_node_indices
                )

                if len(matched_leaf_ids) == 0:
                    print("Skip Plot 8: no B leaf has matched A structures.")
                else:
                    # Extra figure 1: standalone overview-only figure.
                    fig_over = plt.figure(figsize=(10, 10))
                    ax_over_only = fig_over.add_subplot(111, projection=wcs_b)
                    _draw_overview_panel(ax_over_only, draw_split=True, add_legend=True)
                    over_only_name = os.path.join(self.output_dir, "b_anchor_overview_only.pdf")
                    fig_over.savefig(over_only_name, dpi=300)
                    plt.close(fig_over)
                    print(f"Saved plot: {over_only_name}")

                    # Extra figure 2: merge all non-overview panels into a single axis.
                    fig_merge = plt.figure(figsize=(10, 10))
                    ax_merge = fig_merge.add_subplot(111, projection=wcs_b)

                    merged_bg = np.full((ny_b, nx_b), np.nan, dtype=float)
                    for leaf_id in matched_leaf_ids:
                        leaf_node = physics_leaf_nodes.get(leaf_id)
                        if leaf_node is None:
                            continue

                        b_indices = None
                        if hasattr(leaf_node, 'cluster_members'):
                            b_indices = leaf_node.cluster_members
                        elif hasattr(leaf_node, 'trunk') and hasattr(leaf_node.trunk, 'cluster_members'):
                            b_indices = leaf_node.trunk.cluster_members

                        if b_indices is None or len(b_indices) == 0:
                            continue

                        bx = self.physics_dataarr_acorns[0, b_indices].astype(int)
                        by = self.physics_dataarr_acorns[1, b_indices].astype(int)
                        valid_b = (bx >= 0) & (bx < nx_b) & (by >= 0) & (by < ny_b)
                        if np.sum(valid_b) == 0:
                            continue

                        merged_bg[by[valid_b], bx[valid_b]] = np.asarray(data_bg_b, dtype=float)[by[valid_b], bx[valid_b]]

                    ax_merge.imshow(merged_bg, origin='lower', cmap='gray_r', interpolation='nearest')
                    _draw_a_global_contours_on_b(ax_merge)

                    merged_a_count = 0
                    for leaf_id in matched_leaf_ids:
                        entries = leaf_to_a_entries.get(leaf_id, [])
                        for ent in entries:
                            _draw_a_structure_outline_on_b(
                                ax_merge,
                                ent['pts_x'],
                                ent['pts_y'],
                                color='red',
                                s=1.0,
                                alpha=0.95
                            )
                            merged_a_count += 1

                    unmatched_drawn_merge = _draw_a_nodes_on_b(
                        ax_merge,
                        unmatched_a_node_indices,
                        color='yellow',
                        s=1.0,
                        alpha=0.95
                    )

                    unique_matched_a = len(matched_a_node_indices)
                    ax_merge.legend(
                        handles=_build_plot8_legend_handles(unique_matched_a, unmatched_drawn_merge, include_global_contour=True),
                        loc='upper right',
                        framealpha=0.8,
                        fontsize=8
                    )

                    ax_merge.set_xlim(0, nx_b - 1)
                    ax_merge.set_ylim(0, ny_b - 1)
                    ax_merge.set_title(
                        f"Merged non-overview overlays | B leaves={len(matched_leaf_ids)} | "
                        f"A matched overlays={merged_a_count} | A unmatched={unmatched_drawn_merge}"
                    )
                    merged_name = os.path.join(self.output_dir, "b_anchor_overlay_non_overview.pdf")
                    fig_merge.savefig(merged_name, dpi=300)
                    plt.close(fig_merge)
                    print(f"Saved plot: {merged_name}")

                    rows = max(1, int(getattr(cfg, 'B_ANCHORED_SUBPLOT_ROWS', 3)))
                    cols = max(1, int(getattr(cfg, 'B_ANCHORED_SUBPLOT_COLS', 3)))
                    panels_per_page = rows * cols

                    page_idx = 0
                    cursor = 0
                    while True:
                        fig8, axs8 = plt.subplots(
                            rows,
                            cols,
                            figsize=(4.2 * cols, 4.2 * rows),
                            subplot_kw={'projection': wcs_b}
                        )
                        axs_flat = np.asarray(axs8).reshape(-1)

                        slot = 0
                        # First panel: global overview as requested.
                        if page_idx == 0 and panels_per_page > 0:
                            ax_over = axs_flat[0]
                            _draw_overview_panel(ax_over)
                            slot = 1

                        while slot < panels_per_page and cursor < len(matched_leaf_ids):
                            leaf_id = matched_leaf_ids[cursor]
                            leaf_node = physics_leaf_nodes.get(leaf_id)
                            axp = axs_flat[slot]

                            b_indices = None
                            if hasattr(leaf_node, 'cluster_members'):
                                b_indices = leaf_node.cluster_members
                            elif hasattr(leaf_node, 'trunk') and hasattr(leaf_node.trunk, 'cluster_members'):
                                b_indices = leaf_node.trunk.cluster_members

                            mask_leaf = np.zeros((ny_b, nx_b), dtype=bool)
                            if b_indices is not None and len(b_indices) > 0:
                                bx = self.physics_dataarr_acorns[0, b_indices].astype(int)
                                by = self.physics_dataarr_acorns[1, b_indices].astype(int)
                                valid_b = (bx >= 0) & (bx < nx_b) & (by >= 0) & (by < ny_b)
                                if np.sum(valid_b) > 0:
                                    mask_leaf[by[valid_b], bx[valid_b]] = True

                            bg_masked = np.full((ny_b, nx_b), np.nan, dtype=float)
                            bg_masked[mask_leaf] = np.asarray(data_bg_b, dtype=float)[mask_leaf]
                            axp.imshow(bg_masked, origin='lower', cmap='gray_r', interpolation='nearest')

                            _draw_a_global_contours_on_b(axp)

                            entries = leaf_to_a_entries.get(leaf_id, [])
                            for ent in entries:
                                _draw_a_structure_outline_on_b(
                                    axp,
                                    ent['pts_x'],
                                    ent['pts_y'],
                                    color='red',
                                    s=1.0,
                                    alpha=0.95
                                )

                            rho_vals = [ent.get('spearman_rho', np.nan) for ent in entries]
                            rho_vals = [rv for rv in rho_vals if np.isfinite(rv)]
                            if len(rho_vals) > 0:
                                rho_text = f"rho_max={np.max(rho_vals):.3f}"
                            else:
                                rho_text = "rho_max=n/a"

                            axp.set_xlim(0, nx_b - 1)
                            axp.set_ylim(0, ny_b - 1)
                            axp.set_title(f"B leaf {leaf_id} | A={len(entries)} | {rho_text}")

                            slot += 1
                            cursor += 1

                        for k in range(slot, panels_per_page):
                            axs_flat[k].axis('off')

                        if page_idx > 0 and slot == 0:
                            plt.close(fig8)
                            break

                        fig8.tight_layout()
                        page_name = os.path.join(self.output_dir, f"b_anchor_panels_page_{page_idx:02d}.pdf")
                        fig8.savefig(page_name, dpi=300)
                        plt.close(fig8)
                        print(f"Saved plot: {page_name}")

                        if cursor >= len(matched_leaf_ids):
                            break
                        page_idx += 1

            # --- Plot 5: Interactive 3D scatter (x, y, v) with A-plane and B highlights ---
            if self.physics_forest is not None and self.physics_dataarr_acorns is not None:
                print("Plotting interactive 3D B-scatter with A-plane contours...")

                if self.physics_dataarr_acorns.shape[0] < 2:
                    print("Skip Plot 5: physics_dataarr_acorns does not contain x/y axes.")
                else:
                    bx_all = self.physics_dataarr_acorns[0, :].astype(float)
                    by_all = self.physics_dataarr_acorns[1, :].astype(float)
                    if self.physics_dataarr_acorns.shape[0] >= 5:
                        bv_all = self.physics_dataarr_acorns[4, :].astype(float)
                    else:
                        bv_all = np.zeros_like(bx_all)

                    valid_all = np.isfinite(bx_all) & np.isfinite(by_all) & np.isfinite(bv_all)
                    if np.sum(valid_all) == 0:
                        print("Skip Plot 5: no valid B points for 3D scatter.")
                    else:
                        original_valid_indices = np.where(valid_all)[0]
                        original_to_filtered_idx = {
                            int(orig_idx): int(idx_filtered)
                            for idx_filtered, orig_idx in enumerate(original_valid_indices)
                        }

                        bx_all = bx_all[valid_all]
                        by_all = by_all[valid_all]
                        bv_all = bv_all[valid_all]

                        # Use 95% percentile (2.5% - 97.5%) to avoid extreme data points stretching the range
                        v_min_robust = np.nanpercentile(bv_all, 2.5)
                        v_max_robust = np.nanpercentile(bv_all, 97.5)
                        if not np.isfinite(v_min_robust) or not np.isfinite(v_max_robust) or v_min_robust >= v_max_robust:
                            v_min_robust = np.nanmin(bv_all)
                            v_max_robust = np.nanmax(bv_all)

                        v_span = v_max_robust - v_min_robust
                        z_plane = v_min_robust
                        z_min_plot = z_plane
                        z_max_plot = v_max_robust

                        # Shared background preparation for Plot 5 / Plot 6
                        ny_a, nx_a = data_bg_a.shape
                        bg = np.asarray(data_bg_a, dtype=float)
                        finite_bg = np.isfinite(bg)
                        if np.any(finite_bg):
                            bg_vals = bg[finite_bg]
                            p5 = np.nanpercentile(bg_vals, 5)
                            p95 = np.nanpercentile(bg_vals, 95)
                            if not np.isfinite(p5) or not np.isfinite(p95) or p95 <= p5:
                                p5 = np.nanmin(bg_vals)
                                p95 = np.nanmax(bg_vals)
                            if p95 <= p5:
                                p95 = p5 + 1.0
                            bg_norm = np.clip((np.nan_to_num(bg, nan=p5) - p5) / (p95 - p5), 0.0, 1.0)
                        else:
                            bg_norm = np.zeros_like(bg)

                        enable_plot5 = False
                        if enable_plot5:
                            fig5 = plt.figure(figsize=(12, 10))
                            ax5 = fig5.add_subplot(111, projection='3d')

                       
                            # Fallback for matplotlib versions without 3D imshow support
                            x_ds = np.arange(0, nx_a, 1)
                            y_ds = np.arange(0, ny_a, 1)
                            xg, yg = np.meshgrid(x_ds, y_ds)
                            face_colors = cm.get_cmap('gray_r')(bg_norm)
                            z_plane_arr = np.full_like(bg_norm, z_plane, dtype=float)
                            ax5.plot_surface(
                                xg,
                                yg,
                                z_plane_arr,
                                rstride=1,
                                cstride=1,
                                facecolors=face_colors,
                                shade=False,
                                antialiased=False,
                                linewidth=0,
                                alpha=0.35
                            )

                            # A structure contours on the x-y plane (uniform red)
                            for node in all_leaves:
                                indices = None
                                if hasattr(node, 'cluster_members'):
                                    indices = node.cluster_members
                                elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                                    indices = node.trunk.cluster_members

                                if indices is None or self.dataarr_acorns is None:
                                    continue

                                ax_pts = self.dataarr_acorns[0, indices].astype(int)
                                ay_pts = self.dataarr_acorns[1, indices].astype(int)
                                if len(ax_pts) == 0:
                                    continue

                                pad = 5
                                min_x, max_x = int(np.min(ax_pts)), int(np.max(ax_pts))
                                min_y, max_y = int(np.min(ay_pts)), int(np.max(ay_pts))
                                w = max_x - min_x + 2 * pad
                                h = max_y - min_y + 2 * pad
                                if w <= 0 or h <= 0:
                                    continue

                                mask_local = np.zeros((h, w), dtype=int)
                                mask_local[ay_pts - min_y + pad, ax_pts - min_x + pad] = 1

                                x_local = np.linspace(min_x - pad - 0.5, max_x + pad + 0.5, w)
                                y_local = np.linspace(min_y - pad - 0.5, max_y + pad + 0.5, h)
                                x_local_grid, y_local_grid = np.meshgrid(x_local, y_local)

                                ax5.contour(
                                    x_local_grid,
                                    y_local_grid,
                                    mask_local,
                                    levels=[0.5],
                                    colors=['red'],
                                    linewidths=0.9,
                                    zdir='z',
                                    offset=z_plane + 0.001
                                )

                        # B ACORNS scatter by tree (different colors, similar to examples/plot.py style)
                        iterator_b = self.physics_forest.forest.values() if isinstance(self.physics_forest.forest, dict) else self.physics_forest.forest
                        tree_list_b = list(iterator_b)
                        colour_b = iter(cm.rainbow(np.linspace(0, 1, len(tree_list_b) if len(tree_list_b) > 0 else 1)))

                        if enable_plot5:
                            for tree in tree_list_b:
                                c_tree = next(colour_b)
                                members = set()

                                if hasattr(tree, 'leaves') and len(tree.leaves) > 0:
                                    for leaf in tree.leaves:
                                        if hasattr(leaf, 'cluster_members'):
                                            members.update(list(leaf.cluster_members))
                                elif hasattr(tree, 'trunk') and hasattr(tree.trunk, 'cluster_members'):
                                    members.update(list(tree.trunk.cluster_members))
                                elif hasattr(tree, 'cluster_members'):
                                    members.update(list(tree.cluster_members))

                                if len(members) == 0:
                                    continue

                                idx_arr = np.array(sorted(members), dtype=int)
                                valid_idx = (
                                    (idx_arr >= 0) & (idx_arr < self.physics_dataarr_acorns.shape[1])
                                )
                                idx_arr = idx_arr[valid_idx]
                                if len(idx_arr) == 0:
                                    continue

                                bx = self.physics_dataarr_acorns[0, idx_arr].astype(float)
                                by = self.physics_dataarr_acorns[1, idx_arr].astype(float)
                                if self.physics_dataarr_acorns.shape[0] >= 5:
                                    bv = self.physics_dataarr_acorns[4, idx_arr].astype(float)
                                else:
                                    bv = np.zeros_like(bx)

                                valid_tree = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bv)
                                if np.sum(valid_tree) == 0:
                                    continue

                                bx = bx[valid_tree]
                                by = by[valid_tree]
                                bv = bv[valid_tree]

                                ax5.scatter(
                                    bx,
                                    by,
                                    bv,
                                    marker='.',
                                    s=1,
                                    c='black',
                                    linewidths=0.1,
                                    alpha=0.1,
                                    depthshade=False
                                )
                            # ax5.scatter(
                            #     bx,
                            #     by,
                            #     bv,
                            #     marker='o',
                            #     s=12,
                            #     facecolors='none',
                            #     edgecolors=[c_tree],
                            #     linewidths=0.8,
                            #     alpha=0.9,
                            #     depthshade=False
                            # )

                        # Highlight selected B points participating in each A-structure calculation
                        if enable_plot5 and self.dataarr_acorns is not None:
                            highlight_cmap = cm.get_cmap('viridis', max(1, len(all_leaves)))
                            for i, node in enumerate(all_leaves):
                                indices = None
                                if hasattr(node, 'cluster_members'):
                                    indices = node.cluster_members
                                elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                                    indices = node.trunk.cluster_members

                                if indices is None or len(indices) == 0:
                                    continue

                                ax_pts = self.dataarr_acorns[0, indices].astype(int)
                                ay_pts = self.dataarr_acorns[1, indices].astype(int)
                                if len(ax_pts) == 0:
                                    continue

                                az_pts = None
                                if self.dataarr_acorns.shape[0] >= 5:
                                    az_pts = self.dataarr_acorns[4, indices]

                                selection = self._select_physics_components_for_structure(
                                    pts_x=ax_pts,
                                    pts_y=ay_pts,
                                    pts_z=az_pts,
                                    wcs_a_celestial=wcs_a_celestial,
                                    wcs_b_celestial=wcs_b,
                                    nx_b=nx_b,
                                    ny_b=ny_b
                                )
                                if selection is None:
                                    continue

                                selected_component_indices = selection['selected_component_indices']
                                used_fallback = bool(selection.get('used_fallback', False))

                                highlight_idx = [
                                    original_to_filtered_idx[idx]
                                    for idx in selected_component_indices
                                    if idx in original_to_filtered_idx
                                ]
                                if len(highlight_idx) == 0:
                                    continue

                                highlight_idx = np.array(sorted(set(highlight_idx)), dtype=int)
                                c_high = 'red' if used_fallback else highlight_cmap(i)
                                
                                ax5.scatter(
                                    bx_all[highlight_idx],
                                    by_all[highlight_idx],
                                    bv_all[highlight_idx],
                                    marker='.',
                                    s=5,
                                    facecolors='none',
                                    edgecolors=[c_high],
                                    linewidths=1,
                                    alpha=0.3,
                                    depthshade=False
                                )

                        if enable_plot5:
                            ax5.set_xlabel('X')
                            ax5.set_ylabel('Y')
                            ax5.set_zlabel('V')
                            #ax5.set_zlim([z_plane, v_max + 0.08 * v_span])
                            ax5.set_zlim([z_min_plot, z_max_plot]) #temp
                            ax5.set_title('3D B Scatter (x, y, v) with A-plane and Selected B Highlights')
                            ax5.grid(True, alpha=0.2)

                            savename_3d = os.path.join(self.output_dir, "interactive_3d_structures_A_plane_B_scatter.png")
                            plt.show()
                            #fig5.savefig(savename_3d, dpi=300)
                            plt.close(fig5)
                            print(f"Saved plot: {savename_3d}")
                        else:
                            print("Skip Plot 5: disabled by code setting (enable_plot5=False).")

                        # --- Plot 6: PyVista version of Plot 5 ---
                        if run_plot6:
                            if pv is None:
                                print("Skip Plot 6: pyvista is not installed.")
                            else:
                                print("Plotting Plot 6 with pyvista...")
                                z_stretch = max(nx_a, ny_a) / max(v_span, 1e-6)
                                z_plane_plot = z_plane * z_stretch
                                z_min_plot_stretched = z_min_plot * z_stretch
                                z_max_plot_stretched = z_max_plot * z_stretch

                                plotter = pv.Plotter()

                                # A image projected as a textured plane at z = z_plane
                                if np.any(finite_bg):
                                    bg_255 = np.clip(bg_norm * 255.0, 0.0, 255.0).astype(np.uint8)
                                else:
                                    bg_255 = np.zeros((ny_a, nx_a), dtype=np.uint8)
                                texture_rgb = np.dstack([bg_255, bg_255, bg_255])
                                texture = pv.numpy_to_texture(texture_rgb)

                                x_c = (nx_a - 1) / 2.0
                                y_c = (ny_a - 1) / 2.0
                                plane = pv.Plane(
                                    center=(x_c, y_c, z_plane_plot),
                                    direction=(0.0, 0.0, 1.0),
                                    i_size=float(max(nx_a - 1, 1)),
                                    j_size=float(max(ny_a - 1, 1)),
                                    i_resolution=max(nx_a - 1, 1),
                                    j_resolution=max(ny_a - 1, 1)
                                )
                                plotter.add_mesh(plane, texture=texture, opacity=0.35, lighting=False, name='a_plane')

                                # A structure contours on z=z_plane
                                for node in all_leaves:
                                    indices = None
                                    if hasattr(node, 'cluster_members'):
                                        indices = node.cluster_members
                                    elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                                        indices = node.trunk.cluster_members

                                    if indices is None or self.dataarr_acorns is None:
                                        continue

                                    ax_pts = self.dataarr_acorns[0, indices].astype(int)
                                    ay_pts = self.dataarr_acorns[1, indices].astype(int)
                                    if len(ax_pts) == 0:
                                        continue

                                    pad = 5
                                    min_x, max_x = int(np.min(ax_pts)), int(np.max(ax_pts))
                                    min_y, max_y = int(np.min(ay_pts)), int(np.max(ay_pts))
                                    w = max_x - min_x + 2 * pad
                                    h = max_y - min_y + 2 * pad
                                    if w <= 0 or h <= 0:
                                        continue

                                    mask_local = np.zeros((h, w), dtype=np.uint8)
                                    ix = ax_pts - min_x + pad
                                    iy = ay_pts - min_y + pad
                                    valid_xy = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                                    if np.sum(valid_xy) == 0:
                                        continue
                                    mask_local[iy[valid_xy], ix[valid_xy]] = 1

                                    x_local = np.linspace(min_x - pad - 0.5, max_x + pad + 0.5, w)
                                    y_local = np.linspace(min_y - pad - 0.5, max_y + pad + 0.5, h)
                                    xx, yy = np.meshgrid(x_local, y_local)
                                    zz = np.full_like(xx, z_plane_plot + 0.001 * z_stretch, dtype=float)

                                    grid = pv.StructuredGrid(xx, yy, zz)
                                    grid['mask'] = mask_local.ravel(order='F').astype(float)
                                    contour = grid.contour(isosurfaces=[0.5], scalars='mask')
                                    if contour.n_points > 0:
                                        plotter.add_mesh(
                                            contour,
                                            color='red',
                                            line_width=2.0,
                                            name=f'a_contour_{id(node)}'
                                        )

                                # B base scatter by tree
                                colour_b = iter(cm.rainbow(np.linspace(0, 1, len(tree_list_b) if len(tree_list_b) > 0 else 1)))
                                for tree in tree_list_b:
                                    _ = next(colour_b)
                                    members = set()

                                    if hasattr(tree, 'leaves') and len(tree.leaves) > 0:
                                        for leaf in tree.leaves:
                                            if hasattr(leaf, 'cluster_members'):
                                                members.update(list(leaf.cluster_members))
                                    elif hasattr(tree, 'trunk') and hasattr(tree.trunk, 'cluster_members'):
                                        members.update(list(tree.trunk.cluster_members))
                                    elif hasattr(tree, 'cluster_members'):
                                        members.update(list(tree.cluster_members))

                                    if len(members) == 0:
                                        continue

                                    idx_arr = np.array(sorted(members), dtype=int)
                                    valid_idx = (idx_arr >= 0) & (idx_arr < self.physics_dataarr_acorns.shape[1])
                                    idx_arr = idx_arr[valid_idx]
                                    if len(idx_arr) == 0:
                                        continue

                                    bx = self.physics_dataarr_acorns[0, idx_arr].astype(float)
                                    by = self.physics_dataarr_acorns[1, idx_arr].astype(float)
                                    if self.physics_dataarr_acorns.shape[0] >= 5:
                                        bv = self.physics_dataarr_acorns[4, idx_arr].astype(float)
                                    else:
                                        bv = np.zeros_like(bx)

                                    # Keep all finite points; do not hide out-of-axis-range points.
                                    valid_tree = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bv)
                                    if np.sum(valid_tree) == 0:
                                        continue

                                    pts_tree = np.column_stack((bx[valid_tree], by[valid_tree], bv[valid_tree] * z_stretch))
                                    cloud_tree = pv.PolyData(pts_tree)
                                    plotter.add_mesh(
                                        cloud_tree,
                                        color='black',
                                        point_size=3,
                                        opacity=0.10,
                                        render_points_as_spheres=True
                                    )

                                # Highlight selected B points participating in each A-structure calculation
                                highlight_cmap = cm.get_cmap('viridis', max(1, len(all_leaves)))
                                for i, node in enumerate(all_leaves):
                                    indices = None
                                    if hasattr(node, 'cluster_members'):
                                        indices = node.cluster_members
                                    elif hasattr(node, 'trunk') and hasattr(node.trunk, 'cluster_members'):
                                        indices = node.trunk.cluster_members

                                    if indices is None or len(indices) == 0:
                                        continue

                                    ax_pts = self.dataarr_acorns[0, indices].astype(int)
                                    ay_pts = self.dataarr_acorns[1, indices].astype(int)
                                    if len(ax_pts) == 0:
                                        continue

                                    az_pts = None
                                    if self.dataarr_acorns.shape[0] >= 5:
                                        az_pts = self.dataarr_acorns[4, indices]

                                    selection = self._select_physics_components_for_structure(
                                        pts_x=ax_pts,
                                        pts_y=ay_pts,
                                        pts_z=az_pts,
                                        wcs_a_celestial=wcs_a_celestial,
                                        wcs_b_celestial=wcs_b,
                                        nx_b=nx_b,
                                        ny_b=ny_b
                                    )
                                    if selection is None:
                                        continue

                                    selected_component_indices = selection['selected_component_indices']
                                    used_fallback = bool(selection.get('used_fallback', False))

                                    highlight_idx = [
                                        original_to_filtered_idx[idx]
                                        for idx in selected_component_indices
                                        if idx in original_to_filtered_idx
                                    ]
                                    if len(highlight_idx) == 0:
                                        continue

                                    highlight_idx = np.array(sorted(set(highlight_idx)), dtype=int)
                                    c_high = 'red' if used_fallback else highlight_cmap(i)
                                    rgb = colors.to_rgb(c_high)

                                    pts_high = np.column_stack((
                                        bx_all[highlight_idx],
                                        by_all[highlight_idx],
                                        bv_all[highlight_idx] * z_stretch
                                    ))
                                    # Keep highlighted points even when outside displayed axis bounds.
                                    if pts_high.shape[0] == 0:
                                        continue
                                    cloud_high = pv.PolyData(pts_high)
                                    plotter.add_mesh(
                                        cloud_high,
                                        color=rgb,
                                        point_size=8,
                                        opacity=0.9,
                                        render_points_as_spheres=True
                                    )

                                plotter.set_background('white')
                                plotter.add_axes()
                                plotter.show_grid(
                                    xlabel='X',
                                    ylabel='Y',
                                    zlabel=f'V (x{z_stretch:g} visual stretch)',
                                    bounds=(0.0, float(nx_a - 1), 0.0, float(ny_a - 1), float(z_min_plot_stretched), float(z_max_plot_stretched))
                                )

                                # Use an initial camera close to matplotlib defaults.
                                z_mid = 0.5 * (z_min_plot_stretched + z_max_plot_stretched)
                                plotter.camera_position = [
                                    (x_c, y_c - 1.8 * max(nx_a, ny_a), z_mid),
                                    (x_c, y_c, z_mid),
                                    (0.0, 0.0, 1.0)
                                ]

                                savename_pyvista = os.path.join(
                                    self.output_dir,
                                    'interactive_3d_structures_A_plane_B_scatter_pyvista.png'
                                )
                                try:
                                    plotter.show(title='Plot 6: PyVista 3D B Scatter with A-plane', auto_close=False)
                                except Exception as exc:
                                    print(f"Plot 6 warning: viewer exited with exception ({exc}). Continuing...")

                                try:
                                    if getattr(plotter, 'closed', False):
                                        print("Plot 6 info: plotter was closed by user; skip screenshot.")
                                    else:
                                        plotter.screenshot(savename_pyvista)
                                        print(f"Saved plot: {savename_pyvista}")
                                except Exception as exc:
                                    print(f"Plot 6 warning: failed to capture screenshot ({exc}).")
                                finally:
                                    if not getattr(plotter, 'closed', True):
                                        plotter.close()



