"""Base geometry module for ROTIR.

Handles coordinate transformations, rotations, and visibility calculations
for stellar surface tessellations.

This module implements the foundational geometry operations needed to project
stellar surfaces onto the sky plane.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

import sys
sys.path.append('..')
from rotir_jax.datatypes import Tessellation, StellarGeometry, Star


def rotation_matrix(inc: float, PA: float, obliq: float) -> jnp.ndarray:
    """Compute 3D rotation matrix for stellar orientation.

    This implements the compound rotation that transforms from the stellar
    coordinate frame to the observer's sky frame. The rotation convention
    matches the Julia ROTIR implementation (rot_vertex function).

    Args:
        inc: Inclination in degrees (0 = pole-on, 90 = equator-on)
        PA: Position angle in degrees (orientation of rotation axis on sky)
        obliq: Obliquity/rotation angle in degrees (rotation phase)

    Returns:
        3×3 rotation matrix (JAX array)

    Notes:
        - Rotation sequence: Z(PA) → X(inc) → Z(obliq)
        - This is a 3-1-3 Euler angle sequence
        - Matches geometry.jl lines 120-131 (rot_vertex function)
    """
    # Convert degrees to radians
    angle_r1 = jnp.deg2rad(inc)
    angle_r2 = jnp.deg2rad(PA)
    angle_r3 = jnp.deg2rad(obliq)

    # Precompute sines and cosines
    c1 = jnp.cos(angle_r1)
    s1 = jnp.sin(angle_r1)
    c2 = jnp.cos(angle_r2)
    s2 = jnp.sin(angle_r2)
    c3 = jnp.cos(angle_r3)
    s3 = jnp.sin(angle_r3)

    # Construct rotation matrix (matches Julia exactly)
    # dcm = [-s1*c2*s3+c1*c3  s1*c3*c2+c1*s3 -s1*s2;
    #        -c1*c2*s3-s1*c3  c1*c3*c2-s1*s3 -c1*s2 ;
    #                 -s2*s3           s2*c3     c2 ];
    dcm = jnp.array([
        [-s1*c2*s3 + c1*c3,  s1*c3*c2 + c1*s3, -s1*s2],
        [-c1*c2*s3 - s1*c3,  c1*c3*c2 - s1*s3, -c1*s2],
        [        -s2*s3,              s2*c3,      c2]
    ])

    return dcm


def apply_rotation(
    xyz: jnp.ndarray,
    rotation_mat: jnp.ndarray
) -> jnp.ndarray:
    """Apply rotation matrix to xyz coordinates.

    Args:
        xyz: (npix, 5, 3) array of xyz coordinates
        rotation_mat: (3, 3) rotation matrix

    Returns:
        xyz_rot: (npix, 5, 3) rotated coordinates

    Notes:
        Uses batched matrix multiplication for efficiency.
    """
    # Reshape to (npix*5, 3) for batched matmul
    npix, nvert, _ = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)

    # Apply rotation: xyz_rot = rotation_mat @ xyz.T → transpose back
    xyz_rot_flat = (rotation_mat @ xyz_flat.T).T

    # Reshape back to (npix, 5, 3)
    xyz_rot = xyz_rot_flat.reshape(npix, nvert, 3)

    return xyz_rot


def visible_mask(xyz_rot: jnp.ndarray) -> jnp.ndarray:
    """Determine which pixels are visible from observer.

    A pixel is visible if its center (index 4) has z > 0 in the rotated
    observer frame.

    Args:
        xyz_rot: (npix, 5, 3) rotated xyz coordinates

    Returns:
        mask: (npix,) boolean array, True if pixel is visible

    Notes:
        - Only checks center point (index 4)
        - z > 0 means facing the observer
        - Matches Julia: sometimes_visible, never_visible functions
    """
    # Extract z-coordinate of pixel centers (index 4)
    z_center = xyz_rot[:, 4, 2]

    # Visible if z > 0
    return z_center > 0


def sky_plane_projection(xyz_rot: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Project rotated 3D coordinates onto the 2D sky plane.

    The sky plane is the x-y plane in the observer's frame (z points toward
    observer). This is the standard projection for interferometric observations.

    Args:
        xyz_rot: (npix, 5, 3) rotated xyz coordinates

    Returns:
        x_sky: (npix, 5) x-coordinates on sky plane
        y_sky: (npix, 5) y-coordinates on sky plane

    Notes:
        - Simple orthographic projection (ignore z)
        - Units are in stellar radii by default
        - Will be scaled by angular diameter for observations
    """
    x_sky = xyz_rot[:, :, 0]  # Shape: (npix, 5)
    y_sky = xyz_rot[:, :, 1]  # Shape: (npix, 5)

    return x_sky, y_sky


def create_star(
    tess: Tessellation,
    inclination: float,
    orientation: float,
    intensities: jnp.ndarray,
    diameter: float,
    ld_coeffs: Optional[jnp.ndarray] = None,
) -> Star:
    """Create a Star object for image reconstruction.

    This is the main API for creating stars in the reconstruction workflow.

    Args:
        tess: HEALPix tessellation of unit sphere
        inclination: Inclination in degrees (0 = pole-on, 90 = edge-on)
        orientation: Orientation/rotation angle in degrees
        intensities: (npix,) surface intensity map (normalized [0, 1])
        diameter: Stellar diameter in mas
        ld_coeffs: Optional limb darkening coefficients [u1, u2]

    Returns:
        Star object with geometry and intensities

    Example:
        >>> tess = tessellation_healpix(n=4)
        >>> intensities = jnp.ones(tess.npix)  # Uniform
        >>> star = create_star(
        ...     tess=tess,
        ...     inclination=60.0,
        ...     orientation=0.0,
        ...     intensities=intensities,
        ...     diameter=44.0,  # mas
        ... )
    """
    if ld_coeffs is None:
        ld_coeffs = jnp.array([0.0, 0.0])

    # Compute radius from diameter
    radius = diameter / 2.0

    # Step 1: Scale tessellation to stellar radius
    vertices_xyz = radius * jnp.array(tess.unit_xyz)

    # Step 2: Compute rotation matrix
    rot_mat = rotation_matrix(inclination, 0.0, orientation)  # PA=0 for now

    # Step 3: Apply rotation
    vertices_xyz_rot = apply_rotation(vertices_xyz, rot_mat)

    # Step 4: Determine visibility
    vis_mask = visible_mask(vertices_xyz_rot)

    # Step 5: Extract pixel centers
    centers_xyz_rot = vertices_xyz_rot[:, 4, :]  # (npix, 3)
    x = centers_xyz_rot[:, 0]
    y = centers_xyz_rot[:, 1]
    z = centers_xyz_rot[:, 2]

    # Step 6: Get spherical coordinates from tessellation
    # Use the center point (index 4) from unit_spherical
    theta = jnp.array(tess.unit_spherical[:, 4, 1])  # colatitude
    phi = jnp.array(tess.unit_spherical[:, 4, 2])    # longitude

    return Star(
        tess=tess,
        theta=theta,
        phi=phi,
        x=x,
        y=y,
        z=z,
        visible=vis_mask,
        intensities=intensities,
        diameter=diameter,
        inclination=inclination,
        orientation=orientation,
        ld_coeffs=ld_coeffs,
    )


def compute_limb_darkening(
    mu: jnp.ndarray,
    u1: float = 0.0,
    u2: float = 0.0
) -> jnp.ndarray:
    """Compute limb darkening using quadratic law.

    I(μ) / I(1) = 1 - u1*(1-μ) - u2*(1-μ)^2

    Args:
        mu: (npix,) cosine of angle to line of sight
        u1: Linear limb darkening coefficient
        u2: Quadratic limb darkening coefficient

    Returns:
        intensity: (npix,) relative intensity (1 at disk center)

    Notes:
        - Default u1=u2=0 gives uniform disk
        - Typical values: u1 ~ 0.5-0.8, u2 ~ 0.0-0.3 for stars
    """
    one_minus_mu = 1.0 - mu
    intensity = 1.0 - u1 * one_minus_mu - u2 * one_minus_mu**2

    # Clamp to non-negative
    intensity = jnp.maximum(intensity, 0.0)

    return intensity


def rotate_and_project(
    tessellation: Tessellation,
    radius: float,
    inc: float,
    PA: float,
    obliq: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience function: rotate and project in one call.

    Args:
        tessellation: HEALPix tessellation
        radius: Stellar radius
        inc: Inclination (degrees)
        PA: Position angle (degrees)
        obliq: Obliquity (degrees)

    Returns:
        x_sky: (npix, 5) x-coordinates on sky
        y_sky: (npix, 5) y-coordinates on sky
        vis_mask: (npix,) visibility mask
        mu: (npix,) foreshortening factor
    """
    geom = create_star(tessellation, radius, inc, PA, obliq)
    return (
        jnp.array(geom.x_sky),
        jnp.array(geom.y_sky),
        jnp.array(geom.visible_mask),
        jnp.array(geom.mu)
    )
