# src/loader.py
import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from spectral_cube import SpectralCube as sc
import os
import config as cfg
from src import utils

@utils.trace_error
def load_data():
    """
    Load FITS data, compute RMS, extract beam info and physical size parameters.
    
    Returns:
        scu1: Processed SpectralCube object (includes mask and unit)
        intensity: 2D peak intensity map (Moment Max)
        grid: 2D WCS object (used for plot projection)
        meta_data: Dictionary containing global variables: dx, dy, xmin, beam_area, S, etc.
    """
    filepath = os.path.join(cfg.INPUT_DIR, cfg.FITS_DATA)
    print(f"Loading FITS data from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    # 1. Read base FITS
    hdu = fits.open(filepath)[0]
    wh = wcs.WCS(hdu.header)

    def _parse_beam_from_config(beam_cfg, label):
        """Parse manual beam config (arcsec) into (bmaj_deg, bmin_deg)."""
        if beam_cfg is None:
            return None
        if np.isscalar(beam_cfg):
            bmaj_arcsec = float(beam_cfg)
            bmin_arcsec = float(beam_cfg)
        elif len(beam_cfg) == 2:
            bmaj_arcsec = float(beam_cfg[0])
            bmin_arcsec = float(beam_cfg[1])
        else:
            raise ValueError(f"{label} must be scalar or length-2 sequence")

        if bmaj_arcsec <= 0 or bmin_arcsec <= 0:
            raise ValueError(f"{label} must be positive")

        return bmaj_arcsec / 3600.0, bmin_arcsec / 3600.0
    
    # Load SpectralCube
    scu0 = sc.read(hdu)
    # Handle potential NaN
    scu0_data = np.squeeze(scu0.hdu.data)
    scu0 = scu0.with_spectral_unit(u.km / u.s)

    # 3. Compute Beam and pixel physical size
    # Priority: SINGLE_BEAM_SIZE_ARCSEC > FITS Header(BMAJ/BMIN)
    bmaj_deg = None
    bmin_deg = None

    try:
        parsed = _parse_beam_from_config(getattr(cfg, 'SINGLE_BEAM_SIZE_ARCSEC', None), 'SINGLE_BEAM_SIZE_ARCSEC')
        if parsed is not None:
            bmaj_deg, bmin_deg = parsed
            print(
                "Using manual SINGLE beam from config "
                f"(arcsec): BMAJ={bmaj_deg * 3600.0}, BMIN={bmin_deg * 3600.0}"
            )
    except Exception as e:
        print(f"Warning: invalid SINGLE_BEAM_SIZE_ARCSEC: {e}. Falling back to FITS header.")

    if bmaj_deg is None or bmin_deg is None:
        if 'BMAJ' in hdu.header and 'BMIN' in hdu.header:
            bmaj_deg = float(hdu.header['BMAJ'])
            bmin_deg = float(hdu.header['BMIN'])
            print(f"Using SINGLE beam from FITS header (deg): BMAJ={bmaj_deg}, BMIN={bmin_deg}")
        else:
            raise KeyError(
                "No beam info found for SINGLE mode. "
                "Please set SINGLE_BEAM_SIZE_ARCSEC in config or provide BMAJ/BMIN in FITS header."
            )

    bmaj = bmaj_deg * u.deg
    bmin = bmin_deg * u.deg
    beam_deg = bmaj_deg * bmin_deg  # keep consistent with original S formula

    # Helper conversion factor
    fwhm_to_sigma = 1. / (8 * np.log(2))**0.5

    # Beam Area (for flux-to-temperature conversion)
    # Original formula: 2*pi*(bmaj*bmin*fwhm_to_sigma**2)
    beam_area = 2 * np.pi * (bmaj * bmin * fwhm_to_sigma**2)

    # Pixel physical size S (cm^2)
    # Original formula: (distance * sin( sqrt(beam_deg * fwhm_to_sigma) * 2pi/360 ))^2
    # Note: beam_deg here is BMAJ*BMIN
    # The geometric logic maps beam size to physical scale as a proxy for "pixel size"
    # Strictly reproduce original logic:
    term = np.sqrt(beam_deg * fwhm_to_sigma)  # beam_deg is float here
    sin_val = np.sin(term * 2 * np.pi / 360)  # convert degrees to radians
    S = (cfg.DISTANCE * sin_val)**2
    
# 4. Data preprocessing and RMS calculation (restoring original lines 164-180)
    scu0_data1 = np.nan_to_num(scu0.hdu.data)

    # Create a new Cube object for processing
    # use_dask=True is kept from original code
    scu1 = sc(data=scu0_data1, wcs=scu0.wcs)
    scu1 = scu1.with_beam(scu0.beam)
    scu1 = scu1.with_spectral_unit(u.km / u.s)

    # Calculate RMS noise
    print("Calculating RMS noise...")
    max_flux = scu1.max()
    # Mask out signal regions (above 20% peak)
    rms_mask = scu1 < 0.2 * max_flux
    rms_noise = scu1.with_mask(rms_mask).std()
    print(f"RMS Noise: {rms_noise.value:.4e} {rms_noise.unit}")

    # 5. Generate 2D intensity map (for plotting)
    scu1.allow_huge_operations = True
    intensity = scu1.max(axis=0)  # Peak intensity map

    # Extract WCS Grid (for 2D plot projection)
    grid = wcs.WCS(scu1.header[:], naxis=2)

    # 6. Extract coordinate metadata (dx, dy, xmin, ymin...)
    # Restoring original lines 102-108
    velo, dec, ra = scu0.world[:]
    
    xmax = ra.max().value
    xmin = ra.min().value
    xlen = scu0.shape[2]
    # Avoid division by zero
    if xlen > 1:
        dx = (xmax - xmin) / (xlen - 1)
    else:
        dx = 1.0 # Fallback
        
    ymax = dec.max().value
    ymin = dec.min().value
    ylen = scu0.shape[1]
    if ylen > 1:
        dy = (ymax - ymin) / (ylen - 1)
    else:
        dy = 1.0

    # 7. Pack metadata
    meta_data = {
        'beam_area': beam_area,
        'S': S,
        'rms_noise': rms_noise,
        'velo_axis': scu1.spectral_axis,
        'dx': dx,
        'dy': dy,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'scu0_shape': scu0.shape
    }

    return scu1, intensity, grid, meta_data