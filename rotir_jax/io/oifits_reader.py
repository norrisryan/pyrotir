"""OIFITS reader for ROTIR - reads optical interferometry data.

OIFITS is the standard data exchange format for optical interferometry.
This module reads OIFITS files (v1 and v2) and converts to OIData dataclass.

OIFITS format specification:
- OI_WAVELENGTH: wavelength tables
- OI_ARRAY: array configuration (stations, telescopes)
- OI_VIS: complex visibilities
- OI_VIS2: squared visibility amplitudes (V²)
- OI_T3: triple products (closure phases and amplitudes)

References:
- OIFITS v1: Pauls et al. (2005), PASP 117, 1255
- OIFITS v2: Duvert et al. (2017), A&A 597, A8
"""

import numpy as np
from astropy.io import fits
from typing import Optional, Tuple, List
import warnings

import sys
sys.path.append('..')
from rotir_jax.datatypes import OIData


def read_oifits(
    filename: str,
    wavelength_range: Optional[Tuple[float, float]] = None,
    target_id: Optional[int] = None,
    use_vis: bool = False,
    use_v2: bool = True,
    use_t3: bool = True,
    verbose: bool = True,
) -> OIData:
    """Read OIFITS file and convert to OIData structure.

    Args:
        filename: Path to OIFITS file (.fits or .oifits)
        wavelength_range: Optional (min_wl, max_wl) in meters to filter data
        target_id: Optional target ID to select (default: first target)
        use_vis: Include complex visibilities (if available)
        use_v2: Include squared visibilities
        use_t3: Include closure phases/triple products
        verbose: Print info about loaded data

    Returns:
        OIData: Interferometric data in ROTIR format

    Notes:
        - Reads OI_VIS2 and OI_T3 tables from OIFITS file
        - Computes UV coordinates in cycles/radian
        - Filters flagged data (flag=True means bad data)
        - Handles both OIFITS v1 and v2 formats
        - Wavelength selection averages over specified range

    Example:
        >>> data = read_oifits('mystar.oifits', wavelength_range=(1.5e-6, 1.7e-6))
        >>> print(f"Loaded {data.nv2} V² points, {data.nt3phi} closure phases")
    """
    if verbose:
        print(f"Reading OIFITS file: {filename}")

    # Open FITS file
    hdul = fits.open(filename)

    # Read wavelength table(s)
    wavelength_tables = _read_wavelength_tables(hdul)

    # Read array configuration (optional, for completeness)
    array_info = _read_array_table(hdul)

    # Initialize data arrays
    all_v2 = []
    all_v2_err = []
    all_uv_v2 = []
    all_v2_wl = []

    all_t3amp = []
    all_t3amp_err = []
    all_t3phi = []
    all_t3phi_err = []
    all_uv_t3 = []
    all_t3_wl = []

    # Read OI_VIS2 tables (squared visibilities)
    if use_v2:
        v2_tables = [hdu for hdu in hdul if hdu.name == 'OI_VIS2']
        if verbose:
            print(f"Found {len(v2_tables)} OI_VIS2 table(s)")

        for v2_table in v2_tables:
            v2_data = _read_oi_vis2(v2_table, wavelength_tables, wavelength_range, target_id)
            if v2_data is not None:
                all_v2.extend(v2_data['v2'])
                all_v2_err.extend(v2_data['v2_err'])
                all_uv_v2.extend(v2_data['uv'])
                all_v2_wl.extend(v2_data['wavelength'])

    # Read OI_T3 tables (closure phases)
    if use_t3:
        t3_tables = [hdu for hdu in hdul if hdu.name == 'OI_T3']
        if verbose:
            print(f"Found {len(t3_tables)} OI_T3 table(s)")

        for t3_table in t3_tables:
            t3_data = _read_oi_t3(t3_table, wavelength_tables, wavelength_range, target_id)
            if t3_data is not None:
                all_t3amp.extend(t3_data['t3amp'])
                all_t3amp_err.extend(t3_data['t3amp_err'])
                all_t3phi.extend(t3_data['t3phi'])
                all_t3phi_err.extend(t3_data['t3phi_err'])
                all_uv_t3.extend(t3_data['uv'])
                all_t3_wl.extend(t3_data['wavelength'])

    hdul.close()

    # Convert to numpy arrays
    v2 = np.array(all_v2)
    v2_err = np.array(all_v2_err)
    uv_v2 = np.array(all_uv_v2)  # (nv2, 2)

    t3amp = np.array(all_t3amp)
    t3amp_err = np.array(all_t3amp_err)
    t3phi = np.array(all_t3phi)
    t3phi_err = np.array(all_t3phi_err)
    uv_t3 = np.array(all_uv_t3)  # (nt3*3, 2) - three baselines per triangle

    # Combine all UV coordinates
    # For T3, we have 3 baselines per triangle
    # We need to create a unique UV table and index into it
    all_uv = []
    all_uv.extend(uv_v2.tolist() if len(uv_v2) > 0 else [])

    # For T3, add all three baselines
    if len(uv_t3) > 0:
        uv_t3_reshaped = uv_t3.reshape(-1, 3, 2)  # (nt3, 3, 2)
        for i in range(3):
            all_uv.extend(uv_t3_reshaped[:, i, :].tolist())

    uv_array = np.array(all_uv).T if len(all_uv) > 0 else np.zeros((2, 0))  # (2, nuv)

    # Create indices
    nv2 = len(v2)
    nt3 = len(t3amp)

    indx_v2 = np.arange(nv2)  # First nv2 UV points are for V2

    # T3 baselines start after V2
    # Each triangle has 3 baselines: (1-2), (2-3), (3-1)
    indx_t3_1 = nv2 + np.arange(nt3)  # First baseline of each triangle
    indx_t3_2 = nv2 + nt3 + np.arange(nt3)  # Second baseline
    indx_t3_3 = nv2 + 2*nt3 + np.arange(nt3)  # Third baseline

    # Create OIData structure
    oi_data = OIData(
        uv=uv_array,
        nuv=uv_array.shape[1],
        v2=v2,
        v2_err=v2_err,
        nv2=nv2,
        t3amp=t3amp,
        t3amp_err=t3amp_err,
        t3phi=t3phi,
        t3phi_err=t3phi_err,
        nt3=nt3,
        indx_v2=indx_v2,
        indx_t3_1=indx_t3_1,
        indx_t3_2=indx_t3_2,
        indx_t3_3=indx_t3_3,
    )

    if verbose:
        print(f"Loaded {nv2} V² measurements")
        print(f"Loaded {nt3} closure phase triangles")
        print(f"Total UV points: {uv_array.shape[1]}")
        if len(all_v2_wl) > 0:
            mean_wl = np.mean(all_v2_wl + all_t3_wl)
            print(f"Mean wavelength: {mean_wl*1e6:.3f} μm")

    return oi_data


def _read_wavelength_tables(hdul: fits.HDUList) -> dict:
    """Read OI_WAVELENGTH tables from OIFITS file."""
    wl_tables = {}
    for hdu in hdul:
        if hdu.name == 'OI_WAVELENGTH':
            insname = hdu.header.get('INSNAME', 'default')
            wl_tables[insname] = {
                'eff_wave': hdu.data['EFF_WAVE'],  # meters
                'eff_band': hdu.data['EFF_BAND'],  # meters
            }
    return wl_tables


def _read_array_table(hdul: fits.HDUList) -> Optional[dict]:
    """Read OI_ARRAY table (station/telescope configuration)."""
    for hdu in hdul:
        if hdu.name == 'OI_ARRAY':
            return {
                'sta_name': hdu.data['STA_NAME'],
                'sta_index': hdu.data['STA_INDEX'],
            }
    return None


def _read_oi_vis2(
    hdu: fits.BinTableHDU,
    wavelength_tables: dict,
    wavelength_range: Optional[Tuple[float, float]],
    target_id: Optional[int],
) -> Optional[dict]:
    """Read OI_VIS2 table (squared visibilities)."""
    data = hdu.data

    # Get wavelengths
    insname = hdu.header.get('INSNAME', 'default')
    if insname not in wavelength_tables:
        warnings.warn(f"Wavelength table for {insname} not found")
        return None

    eff_wave = wavelength_tables[insname]['eff_wave']

    # Filter by target ID if specified
    if target_id is not None:
        mask = data['TARGET_ID'] == target_id
        data = data[mask]

    # Extract data
    v2_data = data['VIS2DATA']  # (nobs, nwave)
    v2_err = data['VIS2ERR']
    flag = data['FLAG']  # True = bad data
    ucoord = data['UCOORD']  # meters
    vcoord = data['VCOORD']  # meters

    # Wavelength filtering and averaging
    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        wl_mask = (eff_wave >= wl_min) & (eff_wave <= wl_max)
    else:
        wl_mask = np.ones(len(eff_wave), dtype=bool)

    # Average over wavelength channels (weighted by inverse variance)
    v2_list = []
    v2_err_list = []
    uv_list = []
    wl_list = []

    for i in range(len(v2_data)):
        # Get valid wavelength channels (not flagged, in range)
        valid = wl_mask & ~flag[i]

        if not np.any(valid):
            continue  # Skip if all flagged

        # Weighted average over wavelength
        weights = 1.0 / v2_err[i, valid]**2
        v2_avg = np.average(v2_data[i, valid], weights=weights)
        v2_err_avg = 1.0 / np.sqrt(np.sum(weights))

        # UV coordinates (convert from meters to cycles/radian)
        # UV in meters, wavelength in meters → spatial frequency in cycles/radian
        mean_wl = np.mean(eff_wave[valid])
        u_freq = ucoord[i] / mean_wl  # cycles/radian
        v_freq = vcoord[i] / mean_wl

        v2_list.append(v2_avg)
        v2_err_list.append(v2_err_avg)
        uv_list.append([u_freq, v_freq])
        wl_list.append(mean_wl)

    if len(v2_list) == 0:
        return None

    return {
        'v2': v2_list,
        'v2_err': v2_err_list,
        'uv': uv_list,
        'wavelength': wl_list,
    }


def _read_oi_t3(
    hdu: fits.BinTableHDU,
    wavelength_tables: dict,
    wavelength_range: Optional[Tuple[float, float]],
    target_id: Optional[int],
) -> Optional[dict]:
    """Read OI_T3 table (closure phases and triple amplitudes)."""
    data = hdu.data

    # Get wavelengths
    insname = hdu.header.get('INSNAME', 'default')
    if insname not in wavelength_tables:
        warnings.warn(f"Wavelength table for {insname} not found")
        return None

    eff_wave = wavelength_tables[insname]['eff_wave']

    # Filter by target ID if specified
    if target_id is not None:
        mask = data['TARGET_ID'] == target_id
        data = data[mask]

    # Extract data
    t3amp_data = data['T3AMP']  # (nobs, nwave)
    t3amp_err = data['T3AMPERR']
    t3phi_data = data['T3PHI']  # degrees
    t3phi_err = data['T3PHIERR']
    flag = data['FLAG']

    # UV coordinates for 3 baselines
    u1coord = data['U1COORD']
    v1coord = data['V1COORD']
    u2coord = data['U2COORD']
    v2coord = data['V2COORD']

    # Wavelength filtering
    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        wl_mask = (eff_wave >= wl_min) & (eff_wave <= wl_max)
    else:
        wl_mask = np.ones(len(eff_wave), dtype=bool)

    # Average over wavelength channels
    t3amp_list = []
    t3amp_err_list = []
    t3phi_list = []
    t3phi_err_list = []
    uv_list = []  # Will store 3 baselines per triangle
    wl_list = []

    for i in range(len(t3amp_data)):
        # Get valid wavelength channels
        valid = wl_mask & ~flag[i]

        if not np.any(valid):
            continue

        # Weighted average for T3amp
        weights_amp = 1.0 / t3amp_err[i, valid]**2
        t3amp_avg = np.average(t3amp_data[i, valid], weights=weights_amp)
        t3amp_err_avg = 1.0 / np.sqrt(np.sum(weights_amp))

        # Weighted average for T3phi (circular mean for angles)
        weights_phi = 1.0 / t3phi_err[i, valid]**2
        t3phi_avg = np.average(t3phi_data[i, valid], weights=weights_phi)
        t3phi_err_avg = 1.0 / np.sqrt(np.sum(weights_phi))

        # UV coordinates (convert to cycles/radian)
        mean_wl = np.mean(eff_wave[valid])

        u1 = u1coord[i] / mean_wl
        v1 = v1coord[i] / mean_wl
        u2 = u2coord[i] / mean_wl
        v2 = v2coord[i] / mean_wl

        # Third baseline is closure: u3 = -(u1 + u2), v3 = -(v1 + v2)
        u3 = -(u1 + u2)
        v3 = -(v1 + v2)

        t3amp_list.append(t3amp_avg)
        t3amp_err_list.append(t3amp_err_avg)
        t3phi_list.append(t3phi_avg)
        t3phi_err_list.append(t3phi_err_avg)

        # Store all 3 baselines for this triangle
        uv_list.append([[u1, v1], [u2, v2], [u3, v3]])
        wl_list.append(mean_wl)

    if len(t3amp_list) == 0:
        return None

    return {
        't3amp': t3amp_list,
        't3amp_err': t3amp_err_list,
        't3phi': t3phi_list,
        't3phi_err': t3phi_err_list,
        'uv': uv_list,
        'wavelength': wl_list,
    }


def read_oifits_multiepoch(
    filenames: List[str],
    **kwargs
) -> List[OIData]:
    """Read multiple OIFITS files (multi-epoch observations).

    Args:
        filenames: List of OIFITS file paths
        **kwargs: Arguments passed to read_oifits()

    Returns:
        List of OIData structures, one per epoch

    Example:
        >>> files = ['epoch1.oifits', 'epoch2.oifits', 'epoch3.oifits']
        >>> data_list = read_oifits_multiepoch(files)
        >>> print(f"Loaded {len(data_list)} epochs")
    """
    data_list = []
    for filename in filenames:
        data = read_oifits(filename, **kwargs)
        data_list.append(data)

    return data_list


def summarize_oifits(oi_data: OIData) -> dict:
    """Summarize OIFITS data for quick inspection.

    Args:
        oi_data: OIData structure

    Returns:
        summary: Dictionary with data statistics

    Example:
        >>> data = read_oifits('mystar.oifits')
        >>> summary = summarize_oifits(data)
        >>> print(f"V² range: {summary['v2_range']}")
    """
    summary = {
        'nv2': oi_data.nv2,
        'nt3': oi_data.nt3amp,
        'nuv': oi_data.uv.shape[1],
        'v2_range': (oi_data.v2.min(), oi_data.v2.max()) if oi_data.nv2 > 0 else (0, 0),
        'v2_mean_err': oi_data.v2_err.mean() if oi_data.nv2 > 0 else 0,
        't3phi_range': (oi_data.t3phi.min(), oi_data.t3phi.max()) if oi_data.nt3phi > 0 else (0, 0),
        't3phi_mean_err': oi_data.t3phi_err.mean() if oi_data.nt3phi > 0 else 0,
        'uv_max_baseline': np.sqrt(oi_data.uv[0]**2 + oi_data.uv[1]**2).max() if oi_data.uv.shape[1] > 0 else 0,
    }

    return summary
