"""Regularization functions for ROTIR - mathematical priors for image reconstruction.

Regularizers impose prior knowledge to make ill-posed inverse problems well-posed.
They penalize "undesirable" features while allowing "reasonable" solutions.

Key regularizers:
- Maximum Entropy (MEM): Encourages smoothness, penalizes structure
- Total Variation (TV): Preserves edges, allows piecewise constant regions
- L2 smoothness: Penalizes roughness, encourages smooth gradients
- Mean: Penalizes deviations from mean intensity
- Bias: Asymmetric penalty (e.g., spots darker than photosphere)

Physical motivation:
- Stars are mostly smooth (limit artifacts)
- Sharp features exist (spots, plages) → preserve edges
- Conservation laws (flux, mass)
- Physical constraints (T > 0, spots cooler than photosphere)

Mathematical framework:
- Objective: minimize χ² + Σλᵢ Rᵢ(x)
- χ²: data fidelity term
- Rᵢ(x): regularization functionals
- λᵢ: regularization weights (hyperparameters)

JAX compatible: all functions return (value, gradient) for autodiff optimization.

References:
- Thiébaut (2008): Image reconstruction in interferometry
- Renard et al. (2011): SPARCO image reconstruction
- Baron & Monnier (2012): Principles of image reconstruction
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from functools import partial

import sys
sys.path.append('..')
from rotir_jax.datatypes import Tessellation


def maximum_entropy(
    x: jnp.ndarray,
    epsilon: float = 1e-9,
) -> Tuple[float, jnp.ndarray]:
    """Maximum Entropy Method (MEM) regularizer.

    MEM encourages uniform distributions and penalizes structure.
    Maximizes entropy = -Σ xᵢ log(xᵢ) (or minimizes neg-entropy).

    Args:
        x: Image intensities (npix,)
        epsilon: Small value to avoid log(0) (default 1e-9)

    Returns:
        f: Regularization value (scalar)
        g: Gradient ∂f/∂x (npix,)

    Notes:
        - Normalized by mean to make scale-invariant
        - f = Σ (xᵢ/mean(x)) * log(xᵢ/mean(x))
        - Encourages flat distributions
        - Penalizes sharp peaks and structure
        - Good for featureless stars or prior-free reconstruction

    Physical interpretation:
        - Maximum information entropy
        - Minimum assumptions about image
        - Default to "most uniform" solution consistent with data

    Reference:
        oichi2_spheroid.jl lines 342-353
        Skilling & Bryan (1984): Maximum entropy image reconstruction
    """
    x_mean = jnp.mean(x)

    # Normalized intensity
    xm = x / (x_mean + epsilon)

    # Entropy: Σ xᵢ log(xᵢ)
    f = jnp.sum(xm * jnp.log(xm + epsilon))

    # Gradient: ∂f/∂x
    # d/dx [x/μ * log(x/μ)] = (1/μ) * (log(x/μ) + 1)
    # But μ = mean(x) depends on x, so chain rule applies
    g = ((x_mean - x / len(x)) / (x_mean**2 + epsilon)) * (jnp.log(xm + epsilon) + 1)

    return f, g


def total_variation_l1(
    x: jnp.ndarray,
    diff_matrix: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    """Total Variation L1 regularizer (edge-preserving).

    TV-L1 penalizes the L1 norm of image gradients.
    Encourages piecewise constant regions while preserving sharp edges.

    Args:
        x: Image intensities (npix,)
        diff_matrix: Sparse difference operator D (nedges, npix)
                     Computes differences between neighboring pixels

    Returns:
        f: TV value = ||Dx||₁
        g: Gradient ∂f/∂x

    Notes:
        - f = ||Dx||₁ = sum(|∇x|)
        - g = Dᵀ (Dx / ||Dx||)  (when ||Dx|| > 0)
        - Preserves edges (non-differentiable at jumps)
        - Allows discontinuities (good for spots, plages)
        - Can create staircase artifacts (piecewise constant bias)

    Physical interpretation:
        - Minimizes total edge length
        - Favors simple shapes with few boundaries
        - Good for spot modeling (sharp temp contrasts)

    Reference:
        oichi2_spheroid.jl lines 303-314
        Rudin et al. (1992): Nonlinear total variation
    """
    # Compute differences
    Dx = diff_matrix @ x

    # TV-L1 norm
    tv_norm = jnp.linalg.norm(Dx, ord=1)

    # Gradient
    if tv_norm > 0:
        # g = Dᵀ (Dx / ||Dx||₁)
        # For L1: gradient is sign of Dx
        g = diff_matrix.T @ jnp.sign(Dx)
    else:
        g = jnp.zeros_like(x)

    return tv_norm, g


def total_variation_l2(
    x: jnp.ndarray,
    diff_matrix: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    """Total Variation L2 regularizer (smoothness).

    TV-L2 penalizes the L2 norm squared of image gradients.
    Standard Tikhonov smoothness regularization.

    Args:
        x: Image intensities (npix,)
        diff_matrix: Difference operator D (nedges, npix)

    Returns:
        f: TV-L2 value = ||Dx||₂²
        g: Gradient ∂f/∂x = 2 Dᵀ Dx

    Notes:
        - f = ||Dx||₂² = sum(|∇x|²)
        - g = 2 Dᵀ Dx (quadratic, smooth gradient)
        - Encourages smooth variations
        - Differentiable everywhere (no edge preservation)
        - Blurs edges (less artifacts than TV-L1)

    Physical interpretation:
        - Minimizes curvature / roughness
        - Assumes smooth temperature distributions
        - Good for differential rotation, limb darkening

    Reference:
        oichi2_spheroid.jl lines 293-300
        Tikhonov regularization
    """
    # Compute differences
    Dx = diff_matrix @ x

    # L2 norm squared
    f = jnp.sum(Dx**2)

    # Gradient: 2 Dᵀ Dx
    g = 2 * (diff_matrix.T @ Dx)

    return f, g


def mean_regularization(
    x: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    """Mean regularization - penalizes deviations from mean.

    Encourages solutions close to uniform intensity.
    Useful for enforcing flux conservation or limiting contrast.

    Args:
        x: Image intensities (npix,)

    Returns:
        f: Sum of absolute deviations from mean
        g: Gradient (sign of deviation)

    Notes:
        - f = Σ |xᵢ - mean(x)|
        - g = sign(xᵢ - mean(x))
        - L1 penalty on deviations
        - Less aggressive than MEM
        - Useful for limiting spot contrast

    Reference:
        oichi2_spheroid.jl lines 316-324
    """
    x_mean = jnp.mean(x)

    # L1 deviation from mean
    f = jnp.sum(jnp.abs(x - x_mean))

    # Gradient: sign of deviation
    g = jnp.sign(x - x_mean)

    return f, g


def bias_regularization(
    x: jnp.ndarray,
    bias_factor: float = 2.0,
) -> Tuple[float, jnp.ndarray]:
    """Asymmetric bias regularization (spots vs. faculae).

    Penalizes deviations from mean asymmetrically.
    Useful for spots (cooler) vs. faculae (hotter) with different weights.

    Args:
        x: Image intensities (npix,)
        bias_factor: Asymmetry factor B ≥ 1.0
                     B > 1: penalizes hot features more
                     B < 1: penalizes cool features more

    Returns:
        f: Weighted sum of squared deviations
        g: Gradient with asymmetric weighting

    Notes:
        - f = Σ wᵢ (xᵢ - mean(x))²
        - wᵢ = B if xᵢ > mean, else 1
        - Encourages spots (cool) over faculae (hot) if B > 1
        - Physical: spots more common than faculae on cool stars

    Physical motivation:
        - Solar-type: spots (cool, dark)
        - A-type: may have metallic spots
        - Hot stars: rare spots, possible faculae
        - Bias toward physically expected features

    Reference:
        oichi2_spheroid.jl lines 326-340
    """
    n = len(x)
    x_mean = jnp.mean(x)

    # Asymmetric weights
    # B for positive deviations, 1 for negative
    weights = jnp.where(x > x_mean, bias_factor, 1.0)

    # Weighted squared deviations
    deviations = x - x_mean
    f = jnp.sum(weights * deviations**2) / n

    # Gradient: 2 wᵢ (xᵢ - mean(x)) / n
    g_raw = 2 * weights * deviations / n

    # Subtract mean of gradient (to maintain flux conservation?)
    g = g_raw - jnp.mean(g_raw)

    return f, g


def build_difference_matrix(
    tess: Tessellation,
    use_spherical_neighbors: bool = True,
) -> jnp.ndarray:
    """Build finite-difference matrix for Total Variation regularizer.

    Constructs sparse matrix D that computes differences between neighboring pixels.
    For HEALPix, uses natural pixel neighbors on the sphere.

    Args:
        tess: Tessellation with neighbor information
        use_spherical_neighbors: Use HEALPix neighbors (default True)

    Returns:
        D: Difference matrix (nedges, npix)
           Each row represents an edge: D[i,j] - D[i,k] for neighbors j,k

    Notes:
        - For HEALPix: each pixel has 8 neighbors
        - For each edge (i,j): one row with +1 at i, -1 at j
        - Result: Dx gives gradient at each edge
        - Used by TV-L1 and TV-L2 regularizers

    Example:
        >>> tess = tessellation_healpix(n=3)
        >>> D = build_difference_matrix(tess)
        >>> x = jnp.ones(tess.npix)
        >>> Dx = D @ x  # All zeros (uniform image)

    Reference:
        - Uses HEALPix neighbor structure
        - Similar to finite difference operators in PDEs
    """
    npix = tess.npix

    # Get HEALPix neighbors (8 per pixel)
    # Each pixel has up to 8 neighbors
    # We'll construct differences for each unique edge

    # For simplicity, create differences between each pixel and its "east" and "south" neighbors
    # This avoids double-counting edges

    edges = []
    edge_indices = []

    for i in range(npix):
        # Get neighbors (this would come from HEALPix)
        # For now, use a simple grid-like structure as placeholder
        # In production, use healpy.get_all_neighbours()

        # Placeholder: assume grid-like connectivity
        # East neighbor: i+1 (if valid)
        if (i + 1) < npix and (i + 1) % int(np.sqrt(npix)) != 0:
            edge_indices.append((i, i + 1))

        # South neighbor: i + sqrt(npix) (if valid)
        width = int(np.sqrt(npix))
        if (i + width) < npix:
            edge_indices.append((i, i + width))

    nedges = len(edge_indices)

    # Build sparse difference matrix
    # Each row: +1 at pixel i, -1 at pixel j
    D = np.zeros((nedges, npix))

    for edge_idx, (i, j) in enumerate(edge_indices):
        D[edge_idx, i] = 1.0
        D[edge_idx, j] = -1.0

    return jnp.array(D)


def build_healpix_difference_matrix(
    nside: int,
) -> jnp.ndarray:
    """Build difference matrix using actual HEALPix neighbor structure.

    Uses healpy to get true spherical neighbors for each pixel.

    Args:
        nside: HEALPix nside parameter

    Returns:
        D: Difference matrix (nedges, npix)

    Notes:
        - Requires healpy library
        - Uses get_all_neighbours() for 8-connectivity
        - Handles boundary cases (missing neighbors)
        - More accurate than grid approximation

    Example:
        >>> D = build_healpix_difference_matrix(nside=8)
        >>> npix = 12 * 8**2  # = 768
        >>> x = jnp.random.normal(size=npix)
        >>> Dx = D @ x  # Gradient at each edge
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError("healpy required for HEALPix neighbor structure")

    npix = hp.nside2npix(nside)

    edges = []

    for ipix in range(npix):
        # Get 8 neighbors (NW, N, NE, E, SE, S, SW, W)
        neighbors = hp.get_all_neighbours(nside, ipix)

        # Add edges to "later" neighbors to avoid double-counting
        # Use only 4 directions: E, SE, S, SW (indices 3,4,5,6)
        for direction in [3, 4, 5, 6]:
            neighbor = neighbors[direction]
            if neighbor >= 0 and neighbor > ipix:  # Valid and not already processed
                edges.append((ipix, neighbor))

    nedges = len(edges)

    # Build sparse matrix
    D = np.zeros((nedges, npix))
    for edge_idx, (i, j) in enumerate(edges):
        D[edge_idx, i] = 1.0
        D[edge_idx, j] = -1.0

    return jnp.array(D)


def apply_regularizers(
    x: jnp.ndarray,
    regularizers: List[Dict],
    diff_matrix: Optional[jnp.ndarray] = None,
) -> Tuple[float, jnp.ndarray]:
    """Apply multiple regularizers with specified weights.

    Computes weighted sum of regularization terms and gradients.

    Args:
        x: Image intensities (npix,)
        regularizers: List of regularizer specifications, each a dict with:
            {
                "type": str,  # "mem", "tv", "tv2", "mean", "bias"
                "weight": float,  # Regularization weight λ
                "params": dict,  # Optional parameters (e.g., bias_factor)
                "pixels": array or None,  # Pixel subset (default: all)
            }
        diff_matrix: Difference matrix for TV (required if using TV)

    Returns:
        f_reg: Total regularization value
        g_reg: Total regularization gradient

    Notes:
        - f_total = Σᵢ λᵢ Rᵢ(x)
        - g_total = Σᵢ λᵢ ∇Rᵢ(x)
        - Pixel subsets allow regularizing only visible pixels

    Example:
        >>> regularizers = [
        ...     {"type": "mem", "weight": 0.1},
        ...     {"type": "tv", "weight": 0.01},
        ... ]
        >>> f_reg, g_reg = apply_regularizers(x, regularizers, diff_matrix=D)

    Reference:
        oichi2_spheroid.jl lines 356-374
    """
    f_total = 0.0
    g_total = jnp.zeros_like(x)

    for reg_spec in regularizers:
        reg_type = reg_spec["type"]
        weight = reg_spec["weight"]
        params = reg_spec.get("params", {})
        pixel_subset = reg_spec.get("pixels", None)

        # Extract pixel subset if specified
        if pixel_subset is not None:
            x_sub = x[pixel_subset]
            g_sub = jnp.zeros_like(x_sub)
        else:
            x_sub = x
            g_sub = jnp.zeros_like(x)

        # Apply regularizer
        if reg_type == "mem":
            f_reg, g_sub = maximum_entropy(x_sub, epsilon=params.get("epsilon", 1e-9))

        elif reg_type == "tv" or reg_type == "tv1":
            if diff_matrix is None:
                raise ValueError("diff_matrix required for TV regularizer")
            f_reg, g_sub = total_variation_l1(x_sub, diff_matrix)

        elif reg_type == "tv2":
            if diff_matrix is None:
                raise ValueError("diff_matrix required for TV2 regularizer")
            f_reg, g_sub = total_variation_l2(x_sub, diff_matrix)

        elif reg_type == "mean":
            f_reg, g_sub = mean_regularization(x_sub)

        elif reg_type == "bias":
            bias_factor = params.get("bias_factor", 2.0)
            f_reg, g_sub = bias_regularization(x_sub, bias_factor)

        else:
            raise ValueError(f"Unknown regularizer type: {reg_type}")

        # Add weighted contribution
        f_total += weight * f_reg

        # Place gradient back in full array
        if pixel_subset is not None:
            g_total = g_total.at[pixel_subset].add(weight * g_sub)
        else:
            g_total += weight * g_sub

    return f_total, g_total


# JAX-compatible versions with value_and_grad


@partial(jax.value_and_grad, has_aux=False)
def maximum_entropy_value_and_grad(x: jnp.ndarray, epsilon: float = 1e-9) -> float:
    """JAX value_and_grad compatible version of maximum_entropy."""
    f, _ = maximum_entropy(x, epsilon)
    return f


@partial(jax.value_and_grad, has_aux=False)
def total_variation_l1_value_and_grad(x: jnp.ndarray, diff_matrix: jnp.ndarray) -> float:
    """JAX value_and_grad compatible version of total_variation_l1."""
    f, _ = total_variation_l1(x, diff_matrix)
    return f


@partial(jax.value_and_grad, has_aux=False)
def total_variation_l2_value_and_grad(x: jnp.ndarray, diff_matrix: jnp.ndarray) -> float:
    """JAX value_and_grad compatible version of total_variation_l2."""
    f, _ = total_variation_l2(x, diff_matrix)
    return f
