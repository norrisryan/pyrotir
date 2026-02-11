"""HEALPix tessellation for ROTIR.

This module implements HEALPix tessellation of the unit sphere for stellar
surface reconstruction. Each pixel is represented as a quadrilateral with 4
vertices, computed in NESTED ordering.

Reference: HEALPix nested pixelization scheme (Górski et al. 2005)
"""

import numpy as np
import healpy as hp
from typing import Optional, Tuple, List
from scipy import sparse

import sys
sys.path.append('..')
from rotir_jax.datatypes import Tessellation


def tessellation_healpix(n: int) -> Tessellation:
    """Create HEALPix tessellation of order n.

    Args:
        n: HEALPix order. nside = 2^n, npix = 12 * nside^2.
           n=3 → 768 pixels, n=4 → 3072 pixels.

    Returns:
        Tessellation with unit_xyz and unit_spherical arrays.

    Notes:
        - Uses NESTED ordering throughout (not RING)
        - Vertex ordering: North, West, South, East (indices 0, 1, 2, 3)
        - Index 4 is the pixel center
        - All vertices are on the unit sphere
    """
    nside = 2**n
    npix = hp.nside2npix(nside)

    # Step 1: Get pixel centers using healpy (in NESTED ordering)
    theta_center, phi_center = hp.pix2ang(nside, np.arange(npix), nest=True)

    # Step 2: Compute pixel vertices
    # This is the critical part - we need to compute vertices in the exact
    # N, W, S, E ordering that ROTIR expects
    unit_xyz, unit_spherical = _compute_vertices_nested(nside, npix)

    # Step 3: Add centers to the arrays (index 4)
    # Convert center (theta, phi) to xyz
    center_x = np.sin(theta_center) * np.cos(phi_center)
    center_y = np.sin(theta_center) * np.sin(phi_center)
    center_z = np.cos(theta_center)

    unit_xyz[:, 4, 0] = center_x
    unit_xyz[:, 4, 1] = center_y
    unit_xyz[:, 4, 2] = center_z

    unit_spherical[:, 4, 0] = 1.0  # r = 1 (unit sphere)
    unit_spherical[:, 4, 1] = theta_center
    unit_spherical[:, 4, 2] = phi_center

    # Step 4: Recompute xyz from spherical to ensure consistency
    # This matches the Julia code pattern (tessellation_healpix.jl lines 20-23)
    unit_xyz[:, :, 0] = np.sin(unit_spherical[:, :, 1]) * np.cos(unit_spherical[:, :, 2])
    unit_xyz[:, :, 1] = np.sin(unit_spherical[:, :, 1]) * np.sin(unit_spherical[:, :, 2])
    unit_xyz[:, :, 2] = np.cos(unit_spherical[:, :, 1])

    return Tessellation(
        tessellation_type=0,  # 0 = HEALPix
        npix=npix,
        nside=nside,
        n=n,
        unit_xyz=unit_xyz.astype(np.float32),
        unit_spherical=unit_spherical.astype(np.float32),
    )


def _compute_vertices_nested(nside: int, npix: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vertices for all HEALPix pixels in NESTED ordering.

    This implements the vertex computation algorithm, which is the core of
    the tessellation. We use healpy.boundaries() as a starting point, then
    reorder to match ROTIR's N,W,S,E convention.

    Args:
        nside: HEALPix nside parameter
        npix: Number of pixels (12 * nside^2)

    Returns:
        unit_xyz: (npix, 5, 3) xyz coordinates (4 vertices + center)
        unit_spherical: (npix, 5, 3) spherical coordinates (r, theta, phi)

    Notes:
        healpy.boundaries() returns vertices in a specific order that may
        differ from ROTIR's N,W,S,E convention. We need to identify and
        reorder them correctly.
    """
    unit_xyz = np.zeros((npix, 5, 3), dtype=np.float32)
    unit_spherical = np.zeros((npix, 5, 3), dtype=np.float32)

    # For each pixel, get the 4 corner vertices
    for i in range(npix):
        # healpy.boundaries returns (3, nstep*4) array of xyz vectors
        # with nstep=1, we get the 4 corners
        boundaries = hp.boundaries(nside, i, step=1, nest=True)
        # boundaries shape: (3, 4) with rows = (x, y, z), cols = 4 vertices

        # Extract the 4 vertices
        vertices_xyz = boundaries.T  # Shape: (4, 3)

        # Determine the vertex ordering by their positions
        # We need to map healpy's ordering to ROTIR's N,W,S,E ordering
        # Strategy: compute theta (colatitude) for each vertex to identify N/S
        # and phi (longitude) to identify E/W
        vertices_theta = np.arccos(vertices_xyz[:, 2])  # theta from z
        vertices_phi = np.arctan2(vertices_xyz[:, 1], vertices_xyz[:, 0])  # phi from x,y
        vertices_phi = vertices_phi % (2 * np.pi)  # Wrap to [0, 2π)

        # Identify vertices by position:
        # North: smallest theta (closest to pole)
        # South: largest theta (farthest from pole)
        # For East/West at mid latitudes: compare phi values
        idx_north = np.argmin(vertices_theta)
        idx_south = np.argmax(vertices_theta)

        # Get remaining two indices (for W and E)
        remaining = [j for j in range(4) if j not in [idx_north, idx_south]]

        # West has smaller phi, East has larger phi (generally)
        if vertices_phi[remaining[0]] < vertices_phi[remaining[1]]:
            idx_west, idx_east = remaining[0], remaining[1]
        else:
            idx_west, idx_east = remaining[1], remaining[0]

        # Assign in ROTIR order: N, W, S, E
        ordering = [idx_north, idx_west, idx_south, idx_east]

        for local_idx, hp_idx in enumerate(ordering):
            unit_xyz[i, local_idx, :] = vertices_xyz[hp_idx, :]

            # Compute spherical coordinates
            x, y, z = vertices_xyz[hp_idx, :]
            r = 1.0  # Unit sphere
            theta = np.arccos(np.clip(z, -1, 1))
            phi = np.arctan2(y, x)
            phi = phi % (2 * np.pi)  # Wrap to [0, 2π)

            unit_spherical[i, local_idx, 0] = r
            unit_spherical[i, local_idx, 1] = theta
            unit_spherical[i, local_idx, 2] = phi

    # Set r = 1 for all points (unit sphere)
    unit_spherical[:, :, 0] = 1.0

    return unit_xyz, unit_spherical


def get_neighbors(n: int) -> List[np.ndarray]:
    """Get neighbor indices for each pixel.

    Uses healpy.get_all_neighbours in nested ordering.

    Args:
        n: HEALPix order

    Returns:
        List of arrays, neighbors[i] = array of neighbor indices for pixel i.
        Length is typically 8 for most pixels, 7 for some corner pixels.
        Indices where neighbor = -1 are excluded.
    """
    nside = 2**n
    npix = hp.nside2npix(nside)

    neighbors = []
    for i in range(npix):
        nb = hp.get_all_neighbours(nside, i, nest=True)
        # Remove -1 entries (missing neighbors at corners)
        neighbors.append(nb[nb >= 0])

    return neighbors


def tv_regularization_matrices(
    n: int,
    visible_idx: Optional[np.ndarray] = None
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, List[np.ndarray]]:
    """Build sparse gradient and Laplacian matrices for TV regularization.

    The gradient matrix ∇ has:
        ∇[k, k] = number_of_neighbors(k)
        ∇[neighbor, k] = -1 for each neighbor

    Args:
        n: HEALPix order
        visible_idx: if provided, restrict to visible pixels only

    Returns:
        grad_matrix: sparse (npix, npix) gradient operator
        laplacian: sparse (npix, npix) = grad_matrix.T @ grad_matrix
        neighbors: list of neighbor arrays
    """
    nside = 2**n
    npix = hp.nside2npix(nside)

    neighbors = get_neighbors(n)

    # Build gradient matrix in COO format for efficiency
    row_indices = []
    col_indices = []
    data = []

    for k in range(npix):
        nb = neighbors[k]
        n_neighbors = len(nb)

        # Diagonal element: number of neighbors
        row_indices.append(k)
        col_indices.append(k)
        data.append(n_neighbors)

        # Off-diagonal elements: -1 for each neighbor
        for neighbor in nb:
            row_indices.append(int(neighbor))
            col_indices.append(k)
            data.append(-1.0)

    # Create sparse matrix
    grad_matrix = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(npix, npix)
    ).tocsr()

    # Laplacian = ∇^T @ ∇
    laplacian = grad_matrix.T @ grad_matrix

    # If visible_idx is provided, restrict to visible pixels
    if visible_idx is not None:
        grad_matrix = grad_matrix[np.ix_(visible_idx, visible_idx)]
        laplacian = laplacian[np.ix_(visible_idx, visible_idx)]

    return grad_matrix, laplacian, neighbors
