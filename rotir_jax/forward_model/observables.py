"""Observables module for ROTIR - complete forward model.

This module ties together geometry, polygon FT, and chi-squared calculations
to create a complete forward model for interferometric observations.

It converts:
  Image (pixel intensities) → Complex visibilities → V², T3amp, T3phi → χ²
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict

import sys
sys.path.append('..')
from rotir_jax.datatypes import StellarGeometry, OIData
from rotir_jax.forward_model.polyft import (
    polygon_fourier_transform,
    setup_polyft_matrix,
    polygon_area,
    cvis_to_v2,
    cvis_to_t3,
    mod360,
)


def compute_observables(
    image: jnp.ndarray,
    geom: StellarGeometry,
    oi_data: OIData,
    polyft_matrix: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute interferometric observables from stellar surface image.

    This is the complete forward model:
    Image → Complex visibilities → V², T3amp, T3phi

    Args:
        image: (npix,) surface brightness/intensity map
        geom: StellarGeometry with sky projection and visibility mask
        oi_data: OIData with UV coordinates and baseline indices
        polyft_matrix: Optional precomputed (nuv, npix) polygon FT matrix

    Returns:
        v2_model: (nv2,) squared visibility amplitudes
        t3amp_model: (nt3,) closure phase amplitudes
        t3phi_model: (nt3,) closure phases in degrees

    Notes:
        - Matches observables() in oichi2_spheroid.jl lines 129-135
        - Only visible pixels contribute to observables
        - Flux normalization is handled automatically
    """
    # Extract visible pixels only
    visible_mask = geom.visible_mask
    image_visible = image[visible_mask]

    # Get sky plane coordinates for visible pixels (first 4 vertices)
    x_sky_visible = geom.x_sky[visible_mask, :4]  # (nvis, 4)
    y_sky_visible = geom.y_sky[visible_mask, :4]  # (nvis, 4)

    # Compute pixel fluxes (area * intensity)
    # Use Shoelace formula for projected area
    areas = polygon_area(x_sky_visible, y_sky_visible)  # (nvis,)
    flux_visible = areas * image_visible  # (nvis,)

    # Compute complex visibilities
    if polyft_matrix is not None:
        # Use precomputed matrix (faster)
        # polyft_matrix is (nuv, npix), but we only use visible pixels
        polyft_visible = polyft_matrix[:, visible_mask]
        total_flux = jnp.sum(flux_visible)
        cvis_model = jnp.dot(polyft_visible, flux_visible) / total_flux
    else:
        # Compute on-the-fly
        cvis_model = polygon_fourier_transform(
            x_sky_visible, y_sky_visible,
            oi_data.uv[0, :], oi_data.uv[1, :],
            flux_visible
        )

    # Convert to observables
    # V² at specified baselines
    v2_model = cvis_to_v2(cvis_model)[oi_data.indx_v2]

    # Closure phases at specified triangles
    _, t3amp_model, t3phi_model = cvis_to_t3(
        cvis_model,
        oi_data.indx_t3_1,
        oi_data.indx_t3_2,
        oi_data.indx_t3_3
    )

    return v2_model, t3amp_model, t3phi_model


def compute_chi2(
    image: jnp.ndarray,
    geom: StellarGeometry,
    oi_data: OIData,
    polyft_matrix: Optional[jnp.ndarray] = None,
    return_components: bool = False,
    verbose: bool = False,
) -> jnp.ndarray:
    """Compute total chi-squared for image reconstruction.

    χ² = χ²_V² + χ²_T3amp + χ²_T3phi

    where:
    χ²_V² = Σ((V²_model - V²_data) / σ_V²)²
    χ²_T3amp = Σ((T3amp_model - T3amp_data) / σ_T3amp)²
    χ²_T3phi = Σ(mod360(T3phi_model - T3phi_data) / σ_T3phi)²

    Args:
        image: (npix,) surface brightness map
        geom: StellarGeometry
        oi_data: OIData with measurements and errors
        polyft_matrix: Optional precomputed FT matrix
        return_components: If True, return (chi2_total, chi2_v2, chi2_t3amp, chi2_t3phi)
        verbose: Print chi-squared components

    Returns:
        chi2_total: Total chi-squared (scalar)
        OR (chi2_total, chi2_v2, chi2_t3amp, chi2_t3phi) if return_components=True

    Notes:
        - Matches spheroid_chi2_f in oichi2_spheroid.jl lines 159-168
        - Uses mod360 for closure phase residuals (phase wrapping)
        - Weighted by measurement uncertainties
    """
    # Compute model observables
    v2_model, t3amp_model, t3phi_model = compute_observables(
        image, geom, oi_data, polyft_matrix
    )

    # Compute chi-squared components
    # V² chi-squared
    v2_residual = (v2_model - oi_data.v2) / oi_data.v2_err
    chi2_v2 = jnp.sum(v2_residual ** 2)

    # T3 amplitude chi-squared
    t3amp_residual = (t3amp_model - oi_data.t3amp) / oi_data.t3amp_err
    chi2_t3amp = jnp.sum(t3amp_residual ** 2)

    # T3 phase chi-squared (with phase wrapping)
    t3phi_residual = mod360(t3phi_model - oi_data.t3phi) / oi_data.t3phi_err
    chi2_t3phi = jnp.sum(t3phi_residual ** 2)

    # Total chi-squared
    chi2_total = chi2_v2 + chi2_t3amp + chi2_t3phi

    if verbose:
        # Reduced chi-squared
        chi2_v2_reduced = chi2_v2 / oi_data.nv2 if oi_data.nv2 > 0 else 0.0
        chi2_t3amp_reduced = chi2_t3amp / oi_data.nt3amp if oi_data.nt3amp > 0 else 0.0
        chi2_t3phi_reduced = chi2_t3phi / oi_data.nt3phi if oi_data.nt3phi > 0 else 0.0

        print(f"V²: {chi2_v2_reduced:.4f}  "
              f"T3amp: {chi2_t3amp_reduced:.4f}  "
              f"T3phi: {chi2_t3phi_reduced:.4f}")

    if return_components:
        return chi2_total, chi2_v2, chi2_t3amp, chi2_t3phi
    else:
        return chi2_total


def compute_chi2_gradient(
    image: jnp.ndarray,
    geom: StellarGeometry,
    oi_data: OIData,
    polyft_matrix: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute chi-squared and its gradient with respect to image.

    Uses JAX auto-differentiation for gradient computation.

    Args:
        image: (npix,) surface brightness map
        geom: StellarGeometry
        oi_data: OIData
        polyft_matrix: (nuv, npix) precomputed FT matrix

    Returns:
        chi2: Total chi-squared (scalar)
        gradient: (npix,) gradient of chi2 with respect to image

    Notes:
        - Alternative to manual gradient in spheroid_chi2_fg
        - JAX computes exact gradient via auto-differentiation
        - More maintainable and less error-prone
    """
    # Define chi2 function for auto-differentiation
    def chi2_fn(img):
        return compute_chi2(img, geom, oi_data, polyft_matrix)

    # Compute value and gradient
    chi2, grad = jax.value_and_grad(chi2_fn)(image)

    return chi2, grad


def create_forward_model(
    geom: StellarGeometry,
    oi_data: OIData,
) -> Dict:
    """Create forward model with precomputed matrices.

    This sets up all the infrastructure needed for fast forward model
    evaluation during optimization.

    Args:
        geom: StellarGeometry with sky projection
        oi_data: OIData with UV coordinates

    Returns:
        forward_model: Dictionary with:
            - polyft_matrix: (nuv, npix) precomputed polygon FT matrix
            - geom: StellarGeometry
            - oi_data: OIData
            - compute_chi2: Function image → chi2
            - compute_observables: Function image → (v2, t3amp, t3phi)

    Notes:
        - Precomputing polyft_matrix speeds up optimization significantly
        - Forward model can be reused for multiple reconstructions
    """
    # Extract visible pixels
    visible_mask = geom.visible_mask
    x_sky_visible = geom.x_sky[visible_mask, :4]
    y_sky_visible = geom.y_sky[visible_mask, :4]

    # Precompute polygon FT matrix
    polyft_matrix = setup_polyft_matrix(
        x_sky_visible, y_sky_visible,
        oi_data.uv[0, :], oi_data.uv[1, :]
    )

    # Create wrapper functions
    def chi2_fn(image):
        return compute_chi2(image, geom, oi_data, polyft_matrix)

    def chi2_grad_fn(image):
        return compute_chi2_gradient(image, geom, oi_data, polyft_matrix)

    def observables_fn(image):
        return compute_observables(image, geom, oi_data, polyft_matrix)

    return {
        'polyft_matrix': polyft_matrix,
        'geom': geom,
        'oi_data': oi_data,
        'compute_chi2': chi2_fn,
        'compute_chi2_gradient': chi2_grad_fn,
        'compute_observables': observables_fn,
    }


def compute_residuals(
    image: jnp.ndarray,
    geom: StellarGeometry,
    oi_data: OIData,
    polyft_matrix: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Compute normalized residuals for all observables.

    Useful for diagnostics and quality assessment.

    Args:
        image: (npix,) surface brightness map
        geom: StellarGeometry
        oi_data: OIData
        polyft_matrix: Optional precomputed FT matrix

    Returns:
        residuals: Dictionary with:
            - v2_residual: (nv2,) normalized V² residuals
            - t3amp_residual: (nt3,) normalized T3amp residuals
            - t3phi_residual: (nt3,) normalized T3phi residuals (wrapped)
            - v2_model: (nv2,) model V²
            - t3amp_model: (nt3,) model T3amp
            - t3phi_model: (nt3,) model T3phi
    """
    # Compute model observables
    v2_model, t3amp_model, t3phi_model = compute_observables(
        image, geom, oi_data, polyft_matrix
    )

    # Compute normalized residuals
    v2_residual = (v2_model - oi_data.v2) / oi_data.v2_err
    t3amp_residual = (t3amp_model - oi_data.t3amp) / oi_data.t3amp_err
    t3phi_residual = mod360(t3phi_model - oi_data.t3phi) / oi_data.t3phi_err

    return {
        'v2_residual': v2_residual,
        't3amp_residual': t3amp_residual,
        't3phi_residual': t3phi_residual,
        'v2_model': v2_model,
        't3amp_model': t3amp_model,
        't3phi_model': t3phi_model,
    }


def compute_reduced_chi2(
    chi2: float,
    nv2: int,
    nt3amp: int,
    nt3phi: int,
    n_params: int,
) -> float:
    """Compute reduced chi-squared.

    χ²_reduced = χ² / (n_data - n_params)

    Args:
        chi2: Total chi-squared
        nv2: Number of V² measurements
        nt3amp: Number of T3 amplitude measurements
        nt3phi: Number of T3 phase measurements
        n_params: Number of free parameters (e.g., number of visible pixels)

    Returns:
        chi2_reduced: Reduced chi-squared

    Notes:
        - Good fit: χ²_reduced ≈ 1
        - χ²_reduced >> 1: Model doesn't fit data or errors underestimated
        - χ²_reduced << 1: Overfitting or errors overestimated
    """
    n_data = nv2 + nt3amp + nt3phi
    n_dof = n_data - n_params

    if n_dof <= 0:
        return float('inf')

    return chi2 / n_dof


# Multi-epoch support (for Step 12, but define interface now)
def compute_chi2_multiepoch(
    images: list,
    geoms: list,
    oi_datas: list,
    polyft_matrices: Optional[list] = None,
    epoch_weights: Optional[jnp.ndarray] = None,
    verbose: bool = False,
) -> jnp.ndarray:
    """Compute total chi-squared across multiple epochs.

    Args:
        images: List of (npix,) brightness maps for each epoch
        geoms: List of StellarGeometry for each epoch
        oi_datas: List of OIData for each epoch
        polyft_matrices: Optional list of precomputed FT matrices
        epoch_weights: Optional (nepochs,) weights for each epoch
        verbose: Print per-epoch chi-squared

    Returns:
        chi2_total: Total weighted chi-squared

    Notes:
        - Matches spheroid_chi2_allepochs_f in oichi2_spheroid.jl lines 192-206
        - For Step 12 (multi-epoch support)
        - Currently assumes same image for all epochs (time-independent)
    """
    nepochs = len(images)

    if polyft_matrices is None:
        polyft_matrices = [None] * nepochs

    if epoch_weights is None:
        epoch_weights = jnp.ones(nepochs)

    # Compute chi2 for each epoch
    chi2_epochs = []
    for i in range(nepochs):
        chi2_i = compute_chi2(
            images[i], geoms[i], oi_datas[i],
            polyft_matrices[i], verbose=verbose
        )
        chi2_epochs.append(chi2_i)

    # Total weighted chi-squared
    chi2_total = jnp.sum(jnp.array(chi2_epochs) * epoch_weights)

    if verbose:
        print(f"Multi-epoch total chi²: {chi2_total:.2f}")

    return chi2_total
