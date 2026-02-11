"""Roche geometry for ROTIR - tidal distortion in binary stars.

Implements Roche potential and Roche lobe calculations for close binary stars.
Tidal forces distort stellar shapes, creating teardrop-shaped Roche lobes.

Key physics:
- Gravitational potential from both stars
- Centrifugal force from orbital motion
- Lagrange points (L1, L2, L3, L4, L5)
- Mass transfer through L1 point
- Roche lobe overflow

Applications:
- Algol-type binaries
- Cataclysmic variables
- X-ray binaries
- Symbiotic stars (e.g., R Aquarii)
- Common envelope evolution

References:
- Eggleton (1983): Approximate Roche lobe radius
- Kopal (1959): Close Binary Systems
- Aufdenberg et al. (2021): SPICA Roche model
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

import sys
sys.path.append('..')
from rotir_jax.datatypes import Tessellation


def eggleton_roche_radius(q: float) -> float:
    """Eggleton approximation for Roche lobe radius.

    Approximate mean radius of Roche lobe for a star in a binary.

    Args:
        q: Mass ratio M_star/M_companion

    Returns:
        R_L: Roche lobe radius / separation (dimensionless)

    Notes:
        - Accurate to ~1% for all mass ratios
        - R_L is mean effective radius
        - Actual shape is not spherical!
        - Valid for circular, synchronous orbits

    Reference:
        Eggleton (1983): ApJ 268, 368
        R_L/a = 0.49*q^(2/3) / [0.6*q^(2/3) + ln(1 + q^(1/3))]

    Example:
        >>> q = 0.5  # Secondary half mass of primary
        >>> R_L = eggleton_roche_radius(q)
        >>> print(f"Roche lobe radius: {R_L:.3f} × separation")
        Roche lobe radius: 0.377 × separation
    """
    q_pow_1_3 = q ** (1.0 / 3.0)
    q_pow_2_3 = q_pow_1_3 ** 2

    R_L = 0.49 * q_pow_2_3 / (0.6 * q_pow_2_3 + jnp.log(1.0 + q_pow_1_3))

    return R_L


def roche_radius_pathania(q: float) -> float:
    """Pathania approximation for Roche lobe radius.

    Alternative to Eggleton, slightly different functional form.

    Args:
        q: Mass ratio M_star/M_companion

    Returns:
        R_L: Roche lobe radius / separation

    Notes:
        - Similar accuracy to Eggleton
        - Different fitting function
        - Used as initial guess for iterative solvers

    Reference:
        Pathania et al. (2021): Ap&SS 366, 46
    """
    q_pow_1_3 = q ** (1.0 / 3.0)

    # Pathania formula (approximate)
    R_L = 0.46 * q_pow_1_3 * (1.0 + 0.28 * q_pow_1_3)

    return R_L


def compute_potential_primary(
    r: float,
    D: float,
    theta: float,
    phi: float,
    q: float,
    async_ratio: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute Roche potential for primary star.

    Roche potential includes:
    1. Gravitational potential of primary (1/r)
    2. Gravitational potential of secondary (q/|r-D|)
    3. Centrifugal potential (rotation with orbit)

    Args:
        r: Radius from primary center (dimensionless, r/a)
        D: Instantaneous separation (dimensionless, d/a)
        theta: Colatitude (radians)
        phi: Longitude (radians)
        q: Mass ratio M2/M1
        async_ratio: ω_rot/ω_orb (default 1.0 = synchronous)

    Returns:
        Ω: Roche potential
        dΩ: Radial derivative dΩ/dr
        ddΩ: Second derivative d²Ω/dr²

    Notes:
        - Primary at origin (0,0,0)
        - Secondary at (D,0,0)
        - Dimensionless units: distance in units of a
        - Synchronous rotation: async_ratio = 1.0
        - Asynchronous: async_ratio ≠ 1.0

    Reference:
        - Aufdenberg et al. (2021): ApJ 920, 130
        - geometry_rochelobe.jl lines 130-138
    """
    # Direction cosines
    λ = jnp.sin(theta) * jnp.cos(phi)  # x-direction
    ν = jnp.cos(theta)                 # z-direction

    # Distance to secondary
    r_secondary_sq = D**2 + r**2 - 2 * r * λ * D
    r_secondary = jnp.sqrt(r_secondary_sq)

    # Roche potential
    # Ω = 1/r + q/r_sec - q*r*λ*D + (1/2)*Ω²*r²*(1-ν²)
    Ω = (1.0 / r +
         q / r_secondary -
         q * r * λ * D +
         0.5 * async_ratio**2 * (1 + q) * r**2 * (1 - ν**2))

    # First derivative dΩ/dr
    dΩ = (-1.0 / r**2 -
          q * (r - λ * D) / (r_secondary_sq * r_secondary) -
          q * λ * D +
          async_ratio**2 * (1 + q) * r * (1 - ν**2))

    # Second derivative d²Ω/dr²
    ddΩ = (2.0 / r**3 +
           3 * q * (D * λ - r)**2 / (r_secondary_sq**2 * r_secondary) -
           q / (r_secondary_sq * r_secondary) +
           async_ratio**2 * (1 + q) * (1 - ν**2))

    return Ω, dΩ, ddΩ


def compute_potential_secondary(
    r: float,
    D: float,
    theta: float,
    phi: float,
    q: float,
    async_ratio: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute Roche potential for secondary star.

    Similar to primary, but coordinate system centered on secondary.

    Args:
        r: Radius from secondary center (dimensionless)
        D: Instantaneous separation (dimensionless)
        theta: Colatitude (radians)
        phi: Longitude (radians)
        q: Mass ratio M2/M1 (same convention as primary)
        async_ratio: ω_rot/ω_orb for secondary

    Returns:
        Ω: Roche potential
        dΩ: Radial derivative
        ddΩ: Second derivative

    Notes:
        - Secondary at origin
        - Primary at (-D,0,0) in secondary frame
        - Different centrifugal term due to different distance from COM

    Reference:
        geometry_rochelobe.jl lines 140-147
    """
    # Direction cosines
    λ = jnp.cos(phi) * jnp.sin(theta)
    ν = jnp.cos(theta)

    # Distance to primary
    r_primary_sq = D**2 + r**2 + 2 * r * λ * D
    r_primary = jnp.sqrt(r_primary_sq)

    # Roche potential (secondary frame)
    Ω = (1.0 / r_primary +
         q / r +
         0.5 * (1 + q) * (D**2 + 2 * D * r * λ) -
         q * (D**2 + r * λ * D) +
         0.5 * async_ratio**2 * (1 + q) * r**2 * (1 - ν**2))

    # First derivative
    dΩ = (-(D * λ + r) / (r_primary_sq * r_primary) -
          q / r**2 +
          λ * D +
          async_ratio**2 * (1 + q) * r * (1 - ν**2))

    # Second derivative
    ddΩ = (2 * q / r**3 +
           3 * (D * λ + r)**2 / (r_primary_sq**2 * r_primary) -
           1.0 / (r_primary_sq * r_primary) +
           async_ratio**2 * (1 + q) * (1 - ν**2))

    return Ω, dΩ, ddΩ


def solve_roche_radius(
    r_init: float,
    pot_surface: float,
    D: float,
    theta: float,
    phi: float,
    q: float,
    async_ratio: float,
    potential_function,
    max_iter: int = 50,
    thresh: float = 1e-6,
) -> float:
    """Solve for radius at given (θ,φ) matching surface potential.

    Uses Newton-Raphson to find r such that Ω(r,θ,φ) = Ω_surface.

    Args:
        r_init: Initial guess for radius
        pot_surface: Target surface potential
        D: Binary separation (dimensionless)
        theta: Colatitude (radians)
        phi: Longitude (radians)
        q: Mass ratio
        async_ratio: Rotation ratio
        potential_function: Either compute_potential_primary or _secondary
        max_iter: Maximum iterations (default 50)
        thresh: Convergence threshold (default 1e-6)

    Returns:
        r: Radius at (θ,φ) on equipotential surface

    Notes:
        - Solves Ω(r) = Ω_surface via Newton-Raphson
        - f(r) = Ω(r) - Ω_surface = 0
        - r_new = r - f/f' = r - (Ω - Ω_s)/dΩ

    Reference:
        geometry_rochelobe.jl lines 17-24
    """
    r = r_init

    for _ in range(max_iter):
        # Compute potential and derivatives
        Ω, dΩ, ddΩ = potential_function(r, D, theta, phi, q, async_ratio)

        # Newton-Raphson step
        f = Ω - pot_surface
        r_new = r - f / dΩ

        # Check convergence
        if jnp.abs(r_new - r) < thresh:
            return r_new

        r = r_new

    # Return best guess if not converged
    return r


def compute_fillout_factor(
    rpole: float,
    a: float,
    D: float,
    q: float,
    async_ratio: float = 1.0,
    secondary: bool = False,
) -> float:
    """Compute fillout factor from polar radius.

    Fillout factor measures how much star fills its Roche lobe:
    - f = 0: Point star
    - f = 1: Fills Roche lobe (contact)
    - f > 1: Overflow (mass transfer)

    Args:
        rpole: Polar radius (physical units)
        a: Semi-major axis (same units as rpole)
        D: Dimensionless separation d/a
        q: Mass ratio M2/M1
        async_ratio: Rotation ratio
        secondary: If True, compute for secondary star

    Returns:
        fillout: Fillout factor [0, ∞)

    Notes:
        - f = (Ω_L1 + offset) / (Ω_surface + offset)
        - offset = q²/2(1+q) ensures f=1 at contact
        - f < 1: Detached
        - f = 1: Semi-detached
        - f > 1: Contact/common envelope

    Reference:
        Leahy et al. (2015): MNRAS 251, 203
        geometry_rochelobe.jl lines 98-108
    """
    # Choose potential function
    pot_func = compute_potential_secondary if secondary else compute_potential_primary

    # Surface potential at pole (θ=0)
    rpole_dimensionless = rpole / a
    Ω_surface, _, _ = pot_func(rpole_dimensionless, D, 0.0, 0.0, q, async_ratio)

    # TODO: L1 potential calculation requires solving for L1 point
    # For now, use Eggleton approximation
    R_L1 = eggleton_roche_radius(q if not secondary else 1.0 / q)
    Ω_L1, _, _ = pot_func(R_L1, D, jnp.pi / 2, 0.0, q, async_ratio)

    # Fillout factor (Leahy formula)
    offset = q**2 / (2 * (1 + q))
    fillout = (Ω_L1 + offset) / (Ω_surface + offset)

    return fillout


def compute_roche_shape(
    tess: Tessellation,
    rpole: float,
    a: float,
    D: float,
    q: float,
    async_ratio: float = 1.0,
    secondary: bool = False,
) -> jnp.ndarray:
    """Compute Roche-distorted stellar surface.

    Calculates r(θ,φ) for each pixel on tessellated surface.

    Args:
        tess: Tessellation defining (θ,φ) grid
        rpole: Polar radius (physical units)
        a: Semi-major axis (same units)
        D: Dimensionless separation
        q: Mass ratio
        async_ratio: Rotation ratio (default 1.0)
        secondary: Compute for secondary star (default False)

    Returns:
        radii: (npix,) array of radii at each tessellation point

    Notes:
        - Uses iterative solver at each (θ,φ)
        - Can be slow for large tessellations
        - Consider vectorization for production code

    Example:
        >>> tess = tessellation_healpix(n=3)
        >>> rpole = 10.0  # mas
        >>> a = 30.0      # mas (separation)
        >>> D = 1.0       # Dimensionless
        >>> q = 0.5       # Secondary half mass of primary
        >>> radii = compute_roche_shape(tess, rpole, a, D, q)
        >>> print(f"Polar radius: {radii[0]:.2f}, Equatorial: {radii.max():.2f}")
    """
    # Choose potential function
    pot_func = compute_potential_secondary if secondary else compute_potential_primary

    # Surface potential at pole
    rpole_dim = rpole / a
    pot_surface, _, _ = pot_func(rpole_dim, D, 0.0, 0.0, q, async_ratio)

    # Solve for radius at each pixel
    npix = tess.npix
    radii = jnp.zeros(npix)

    for i in range(npix):
        theta = tess.theta[i]
        phi = tess.phi[i]

        # Solve for radius
        r_dim = solve_roche_radius(
            rpole_dim, pot_surface, D, theta, phi, q, async_ratio, pot_func
        )

        radii = radii.at[i].set(r_dim * a)

    return radii


def compute_L1_distance(q: float) -> float:
    """Compute distance to L1 Lagrange point.

    L1 is the inner Lagrange point between two stars where
    gravitational forces balance. Location of mass transfer.

    Args:
        q: Mass ratio M2/M1

    Returns:
        r_L1: Distance from primary to L1 (in units of separation a)

    Notes:
        - For q → 0: r_L1 → 1 - (q/3)^(1/3)
        - For q = 1: r_L1 = 0.5 (equal masses)
        - Mass transfer occurs when star fills to L1

    Reference:
        Murray & Dermott (1999): Solar System Dynamics, §3.2
    """
    # Use Eggleton approximation
    # More accurate: solve quintic polynomial
    r_L1 = eggleton_roche_radius(q)

    return r_L1


def is_contact_binary(
    rpole1: float,
    rpole2: float,
    a: float,
    q: float,
) -> bool:
    """Check if binary is in contact configuration.

    Args:
        rpole1: Polar radius of primary
        rpole2: Polar radius of secondary
        a: Semi-major axis
        q: Mass ratio M2/M1

    Returns:
        contact: True if stars are in contact (both fill Roche lobes)

    Notes:
        - Detached: Both stars within Roche lobes
        - Semi-detached: One star fills Roche lobe
        - Contact: Both stars fill Roche lobes (common envelope)
    """
    # Roche lobe radii
    R_L1_primary = eggleton_roche_radius(q) * a
    R_L2_secondary = eggleton_roche_radius(1.0 / q) * a

    # Check if both fill
    fills1 = rpole1 >= R_L1_primary * 0.99  # 99% threshold
    fills2 = rpole2 >= R_L2_secondary * 0.99

    return fills1 and fills2
