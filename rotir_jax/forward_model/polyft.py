"""Polygon Fourier Transform for ROTIR.

This module implements the analytical Fourier transform of polygonal surfaces,
which is the core of the interferometric forward model. Unlike FFT-based methods,
this computes exact Fourier transforms of quadrilateral pixels.

The algorithm is based on Green's theorem, converting the 2D area integral into
a 1D line integral around the polygon boundary.

Reference: oichi2_spheroid.jl lines 1-120
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

import sys
sys.path.append('..')
from rotir_jax.datatypes import StarGeometry


def polygon_area(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute projected area of quadrilateral pixels using Shoelace formula.

    For a quadrilateral with vertices (x1,y1), (x2,y2), (x3,y3), (x4,y4):
    Area = 0.5 * |x1(y2-y4) + x2(y3-y1) + x3(y4-y2) + x4(y1-y3)|

    Args:
        x: (npix, 4) x-coordinates of 4 vertices
        y: (npix, 4) y-coordinates of 4 vertices

    Returns:
        areas: (npix,) projected areas

    Notes:
        - This is the Shoelace formula / surveyor's formula
        - Matches setup_polyflux_single in oichi2_spheroid.jl lines 47-62
        - Only uses first 4 vertices (not center at index 4)
    """
    # Shoelace formula for quadrilateral
    # Area = 0.5 * sum of (x[i] * y[i+1] - x[i+1] * y[i])
    area = 0.5 * (
        x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0] +
        x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1] +
        x[:, 2] * y[:, 3] - x[:, 3] * y[:, 2] +
        x[:, 3] * y[:, 0] - x[:, 0] * y[:, 3]
    )

    # Take absolute value (area should be positive)
    return jnp.abs(area)


def edge_fourier_contribution(
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Fourier transform contribution from a polygon edge.

    For an edge from (x1, y1) to (x2, y2), the FT contribution is:
    sinc(kx*Δx + ky*Δy) * exp(-iπ(kx*(x1+x2) + ky*(y1+y2))) * (ky*Δx - kx*Δy)

    Args:
        x1, y1: (npix,) starting vertex coordinates
        x2, y2: (npix,) ending vertex coordinates
        kx, ky: (nuv,) spatial frequencies

    Returns:
        edge_ft: (nuv, npix) complex Fourier transform contribution

    Notes:
        - Matches stcis function in oichi2_spheroid.jl lines 65-67
        - sinc(x) = sin(πx) / (πx) in Julia convention
        - cis(x) = exp(ix) = cos(x) + i*sin(x)
    """
    # Compute edge vectors
    dx = x2 - x1  # (npix,)
    dy = y2 - y1  # (npix,)

    # Compute midpoints
    x_mid = x1 + x2  # (npix,)
    y_mid = y1 + y2  # (npix,)

    # Broadcast to (nuv, npix)
    kx = kx[:, None]  # (nuv, 1)
    ky = ky[:, None]  # (nuv, 1)

    # Compute sinc argument: kx*Δx + ky*Δy
    sinc_arg = kx * dx + ky * dy  # (nuv, npix)

    # Compute sinc(arg) = sin(π*arg) / (π*arg)
    # Handle sinc(0) = 1 case
    sinc_val = jnp.sinc(sinc_arg)  # JAX sinc already uses sinc(x) = sin(πx)/(πx)

    # Compute phase: -π*(kx*(x1+x2) + ky*(y1+y2))
    phase = -jnp.pi * (kx * x_mid + ky * y_mid)  # (nuv, npix)

    # Compute complex exponential: exp(i*phase)
    cis_val = jnp.exp(1j * phase)  # (nuv, npix)

    # Compute edge weight: ky*Δx - kx*Δy
    edge_weight = ky * dx - kx * dy  # (nuv, npix)

    # Combine all terms
    edge_ft = sinc_val * cis_val * edge_weight  # (nuv, npix)

    return edge_ft


def polygon_fourier_transform(
    x: jnp.ndarray,
    y: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    flux: jnp.ndarray,
) -> jnp.ndarray:
    """Compute analytical Fourier transform of quadrilateral polygons.

    This is the core of the forward model: it converts a pixelated stellar
    surface into complex visibilities at the observed spatial frequencies.

    Args:
        x: (npix, 4) x-coordinates of 4 vertices (in angular units)
        y: (npix, 4) y-coordinates of 4 vertices (in angular units)
        u: (nuv,) u spatial frequencies (cycles per angular unit)
        v: (nuv,) v spatial frequencies (cycles per angular unit)
        flux: (npix,) brightness of each pixel

    Returns:
        cvis: (nuv,) complex visibilities (normalized by total flux)

    Notes:
        - Matches setup_polyft_single in oichi2_spheroid.jl lines 95-104
        - Uses Green's theorem: area integral → line integral
        - Exact analytical FT, not FFT approximation
        - UV coordinates convention: matches OIFITS standard
    """
    npix = x.shape[0]
    nuv = u.shape[0]

    # Convert u,v from OIFITS units (cycles/mas) to radians
    # OIFITS: u,v in cycles per milliarcsecond
    # Conversion: mas → radians = π/(180*3600*1000)
    mas_to_rad = jnp.pi / (180.0 * 3600.0 * 1000.0)
    kx = u * (-mas_to_rad)  # (nuv,) - note sign convention
    ky = v * mas_to_rad     # (nuv,)

    # Compute FT contribution from each of the 4 edges
    # Edge 1: vertex 0 → vertex 1
    edge1 = edge_fourier_contribution(
        x[:, 0], y[:, 0], x[:, 1], y[:, 1], kx, ky
    )  # (nuv, npix)

    # Edge 2: vertex 1 → vertex 2
    edge2 = edge_fourier_contribution(
        x[:, 1], y[:, 1], x[:, 2], y[:, 2], kx, ky
    )  # (nuv, npix)

    # Edge 3: vertex 2 → vertex 3
    edge3 = edge_fourier_contribution(
        x[:, 2], y[:, 2], x[:, 3], y[:, 3], kx, ky
    )  # (nuv, npix)

    # Edge 4: vertex 3 → vertex 0 (close the polygon)
    edge4 = edge_fourier_contribution(
        x[:, 3], y[:, 3], x[:, 0], y[:, 0], kx, ky
    )  # (nuv, npix)

    # Sum edge contributions
    edge_sum = edge1 + edge2 + edge3 + edge4  # (nuv, npix)

    # Apply normalization factor: -i/(2π) / (kx² + ky²)
    k_squared = kx[:, None]**2 + ky[:, None]**2  # (nuv, 1)

    # Handle k=0 case (DC component = total flux)
    # For k=0, FT should equal total flux
    k_squared = jnp.where(k_squared == 0, 1.0, k_squared)  # Avoid division by zero

    factor = -1j / (2 * jnp.pi) / k_squared  # (nuv, 1)

    # Polygon FT matrix: (nuv, npix)
    polyft = factor * edge_sum  # (nuv, npix)

    # Compute complex visibilities
    # cvis = (polyft @ flux) / total_flux
    total_flux = jnp.sum(flux)
    cvis = jnp.dot(polyft, flux) / total_flux  # (nuv,)

    return cvis


def setup_polyft_matrix(
    x: jnp.ndarray,
    y: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Precompute polygon Fourier transform matrix.

    This matrix transforms pixel fluxes into complex visibilities:
    cvis = polyft_matrix @ pixel_fluxes / total_flux

    Args:
        x: (npix, 4) x-coordinates of vertices
        y: (npix, 4) y-coordinates of vertices
        u: (nuv,) u spatial frequencies
        v: (nuv,) v spatial frequencies

    Returns:
        polyft_matrix: (nuv, npix) complex matrix

    Notes:
        - Precomputing this matrix speeds up forward model evaluation
        - Only depends on geometry, not on pixel brightness values
        - Can be reused for different brightness distributions
    """
    npix = x.shape[0]
    nuv = u.shape[0]

    # Convert u,v to kx,ky (same as in polygon_fourier_transform)
    mas_to_rad = jnp.pi / (180.0 * 3600.0 * 1000.0)
    kx = u * (-mas_to_rad)
    ky = v * mas_to_rad

    # Compute edge contributions
    edge1 = edge_fourier_contribution(x[:, 0], y[:, 0], x[:, 1], y[:, 1], kx, ky)
    edge2 = edge_fourier_contribution(x[:, 1], y[:, 1], x[:, 2], y[:, 2], kx, ky)
    edge3 = edge_fourier_contribution(x[:, 2], y[:, 2], x[:, 3], y[:, 3], kx, ky)
    edge4 = edge_fourier_contribution(x[:, 3], y[:, 3], x[:, 0], y[:, 0], kx, ky)

    edge_sum = edge1 + edge2 + edge3 + edge4

    # Normalization
    k_squared = kx[:, None]**2 + ky[:, None]**2
    k_squared = jnp.where(k_squared == 0, 1.0, k_squared)
    factor = -1j / (2 * jnp.pi) / k_squared

    polyft_matrix = factor * edge_sum  # (nuv, npix)

    return polyft_matrix


def compute_polyflux(
    geom: StarGeometry,
    intensity: jnp.ndarray,
) -> jnp.ndarray:
    """Compute flux from each pixel (area * intensity).

    Args:
        geom: StarGeometry with x_sky, y_sky, visible_mask
        intensity: (npix,) surface intensity map

    Returns:
        polyflux: (npix,) flux from each pixel

    Notes:
        - Only visible pixels contribute
        - Flux = area * intensity
        - Uses Shoelace formula for projected area
    """
    # Extract visible pixels only
    x_visible = geom.x_sky[geom.visible_mask, :4]  # (nvis, 4)
    y_visible = geom.y_sky[geom.visible_mask, :4]  # (nvis, 4)
    intensity_visible = intensity[geom.visible_mask]  # (nvis,)

    # Compute projected areas
    areas = polygon_area(x_visible, y_visible)  # (nvis,)

    # Compute flux = area * intensity
    flux_visible = areas * intensity_visible  # (nvis,)

    # Create full flux array (zeros for invisible pixels)
    polyflux = jnp.zeros(geom.npix)
    polyflux = polyflux.at[geom.visible_mask].set(flux_visible)

    return polyflux


def cvis_to_v2(cvis: jnp.ndarray) -> jnp.ndarray:
    """Convert complex visibilities to squared visibilities.

    Args:
        cvis: (nuv,) complex visibilities

    Returns:
        v2: (nuv,) squared visibility amplitudes
    """
    return jnp.abs(cvis) ** 2


def cvis_to_t3(
    cvis: jnp.ndarray,
    indx1: jnp.ndarray,
    indx2: jnp.ndarray,
    indx3: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert complex visibilities to closure phase observables.

    The bispectrum (triple product) is:
    t3 = cvis[indx1] * cvis[indx2] * cvis[indx3]

    Args:
        cvis: (nuv,) complex visibilities at all baselines
        indx1, indx2, indx3: (nt3,) indices for the 3 baselines of each triangle

    Returns:
        t3: (nt3,) complex bispectrum
        t3amp: (nt3,) bispectrum amplitude
        t3phi: (nt3,) closure phase in degrees

    Notes:
        - Closure phase is phase-invariant to source position
        - t3phi = angle(t3) * 180/π
        - Matches cvis_to_t3 in oichi2_spheroid.jl lines 115-120
    """
    t3 = cvis[indx1] * cvis[indx2] * cvis[indx3]
    t3amp = jnp.abs(t3)
    t3phi = jnp.angle(t3) * (180.0 / jnp.pi)

    return t3, t3amp, t3phi


def mod360(phi: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to [-180, 180] degrees.

    Args:
        phi: (n,) angles in degrees

    Returns:
        phi_wrapped: (n,) angles wrapped to [-180, 180]

    Notes:
        - Matches mod360 in oichi2_spheroid.jl lines 107-109
        - Essential for closure phase chi-squared calculation
    """
    return ((phi + 180.0) % 360.0) - 180.0
