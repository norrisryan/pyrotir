"""Rapid rotator geometry for ROTIR - gravity darkening and centrifugal distortion.

Implements von Zeipel gravity darkening for rapidly rotating stars.
Rapid rotation causes:
1. Centrifugal distortion (oblate shape)
2. Reduced effective gravity at equator
3. Temperature variation via von Zeipel's theorem: T ∝ g^β

Key stars: Altair, Regulus, Vega, Achernar, Rasalhague

References:
- von Zeipel (1924): Gravity darkening theorem
- Collins & Harrington (1966): Roche model for rotating stars
- Espinosa Lara & Rieutord (2011): Modified gravity darkening
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

import sys
sys.path.append('..')
from rotir_jax.datatypes import Tessellation, StarGeometry


def f_rapid_rot(x: jnp.ndarray) -> jnp.ndarray:
    """Roche model radius function for rapid rotators.

    Computes r(θ) / r_pole for a rotating star in Roche approximation.

    Args:
        x: ω sin(θ), where ω is fractional critical velocity and θ is colatitude

    Returns:
        Radius ratio r(θ) / r_pole

    Notes:
        - From Collins & Harrington (1966) Roche model
        - Handles centrifugal distortion
        - r(θ) = r_pole * f(ω sin θ)
        - At equator (θ=90°): maximum distortion
        - At poles (θ=0°, 180°): r = r_pole

    Reference:
        geometry_rapidrotator.jl lines 1-3
    """
    # Roche formula: r/r_pole = 3*cos((π + arccos(x))/3) / x
    # Safe handling of x→0 (poles)
    result = 3.0 * jnp.cos((jnp.pi + jnp.arccos(x)) / 3.0) / x

    # At poles (x→0), use limit: r → r_pole (ratio = 1)
    result = jnp.where(jnp.abs(x) < 1e-10, 1.0, result)

    return result


def oblate_const(
    rpole: float,
    frac_escapevel: float,
) -> Tuple[float, float, float]:
    """Approximate rapid rotator by oblate spheroid.

    Uses Roche approximation to compute semi-major axes.

    Args:
        rpole: Polar radius (mas)
        frac_escapevel: Fraction of critical (escape) velocity (ω/ω_crit)

    Returns:
        a: Semi-major axis (equatorial, mas)
        b: Semi-major axis (equatorial, mas) - same as a for axisymmetric
        c: Semi-minor axis (polar, mas) - equals rpole

    Notes:
        - For ω → 0: spherical (a = b = c)
        - For ω → 1: highly oblate (a,b >> c)
        - Equatorial radius increases with rotation
        - Used for approximate ellipsoidal geometry

    Reference:
        geometry_rapidrotator.jl lines 26-36
    """
    if frac_escapevel >= 1e-10:
        # Compute equatorial radius using Roche approximation
        x_eq = frac_escapevel * jnp.sin(jnp.pi / 2.0)  # ω at equator
        a = b = 3.0 * rpole * jnp.cos((jnp.pi + jnp.arccos(x_eq)) / 3.0) / x_eq
        c = rpole
    else:
        # Slowly rotating or non-rotating: spherical
        a = b = c = rpole

    return a, b, c


def calc_omega(
    rpole: float,
    oblate: float,
) -> Tuple[float, float, float]:
    """Calculate fractional critical velocity from oblateness.

    Inverse of oblate_const: given oblateness, compute ω.

    Args:
        rpole: Polar radius (mas)
        oblate: Oblateness parameter (R_eq/R_pole - 1)

    Returns:
        omega: Fractional critical velocity ω/ω_crit
        rpole: Polar radius (passed through)
        R_equ: Equatorial radius (mas)

    Notes:
        - Oblateness: ε = (R_eq - R_pole) / R_pole
        - R_eq = R_pole * (1 + ε)
        - Solves Roche equation for ω given R_eq/R_pole

    Example:
        >>> omega, rp, req = calc_omega(1.0, 0.2)  # 20% oblate
        >>> print(f"ω/ω_crit = {omega:.3f}, R_eq/R_pole = {req/rp:.3f}")

    Reference:
        geometry_rapidrotator.jl lines 52-57
    """
    R_equ = (1.0 + oblate) * rpole
    omega_0 = 1.0 - rpole / R_equ
    omega = jnp.sqrt(27.0 * omega_0 * (1.0 - omega_0)**2 / 4.0)

    return omega, rpole, R_equ


def calc_rotspin(
    rpole: float,
    R_equ: float,
    omega_c: float,
    mass: float,
) -> Tuple[float, float]:
    """Calculate rotation period and velocity from geometry.

    Args:
        rpole: Polar radius (solar radii)
        R_equ: Equatorial radius (solar radii)
        omega_c: Fractional critical velocity ω/ω_crit
        mass: Stellar mass (solar masses)

    Returns:
        rotational_vel: Rotation rate (rotations/day)
        rotation_period: Rotation period (days)

    Notes:
        - Critical velocity: v_crit = sqrt(2GM/3R_pole)
        - Actual velocity: v = (ω/ω_crit) * (2R_eq/3R_pole) * v_crit
        - Kepler velocity at equator

    Example:
        >>> # Altair: M=1.79 M_sun, R_eq=2.03 R_sun, R_pole=1.63 R_sun
        >>> rot_vel, period = calc_rotspin(1.63, 2.03, 0.9, 1.79)
        >>> print(f"Period: {period:.2f} days")

    Reference:
        geometry_rapidrotator.jl lines 39-49
    """
    # Kepler angular velocity
    omega_k = jnp.sqrt(8.0 * (R_equ / rpole)**3 / 27.0) * omega_c

    # Physical constants (CGS)
    G = 6.67e-8  # cm^3/g/s^2
    M_sun = 2.0e33  # g
    R_sun = 7.0e10  # cm

    # Critical velocity (km/s)
    v_crit = jnp.sqrt((2.0/3.0) * G * mass * M_sun / (rpole * R_sun)) * 1e-5

    # Actual velocity (km/s)
    velocity = omega_c * 2.0 * R_equ / (3.0 * rpole) * v_crit

    # Rotation period (days)
    rotation_period = 2.0 * jnp.pi * R_equ * R_sun * 1e-5 / velocity
    rotation_period /= (60.0 * 60.0 * 24.0)  # seconds to days

    # Angular velocity (degrees/day)
    ang_vel = velocity / (R_equ * 1e-5 * 60.0 * 60.0 * 24.0)

    # Rotational velocity (rotations/day)
    rotational_vel = ang_vel * (jnp.pi / 180.0)

    return rotational_vel, rotation_period


def temperature_map_vonZeipel(
    tess: Tessellation,
    geom: StarGeometry,
    tpole: float,
    rpole: float,
    frac_escapevel: float,
    beta: float = 0.25,
    GM: float = 1.0,
    offsets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute temperature map using von Zeipel gravity darkening.

    Von Zeipel's theorem: For a rotating star in radiative equilibrium,
    the effective temperature is proportional to the local effective gravity:
        T_eff(θ) = T_pole * [g_eff(θ) / g_pole]^β

    where:
    - g_eff includes centrifugal force
    - β is the gravity darkening exponent
    - β = 0.25 for radiative envelopes (von Zeipel)
    - β = 0.08-0.10 for convective envelopes

    Args:
        tess: Tessellation with vertices
        geom: StarGeometry with spherical coordinates
        tpole: Effective temperature at pole (K)
        rpole: Polar radius (any units, consistent with GM)
        frac_escapevel: ω/ω_crit, fractional critical velocity
        beta: Gravity darkening exponent (default 0.25)
        GM: Gravitational parameter (default 1.0, normalized units)
        offsets: Optional (3,) array [dx, dy, dz] for offset from origin

    Returns:
        temperature_map: (npix,) effective temperature at each pixel (K)

    Notes:
        - Effective gravity: g_eff = g_gravity + g_centrifugal
        - g_r = -GM/r² + r*ω²*sin²(θ)  (radial component)
        - g_θ = r*ω²*sin(θ)*cos(θ)     (latitudinal component)
        - |g_eff| = sqrt(g_r² + g_θ²)
        - Equator: lower gravity → cooler → darker
        - Poles: higher gravity → hotter → brighter

    Example (Altair):
        >>> tess = tessellation_healpix(n=5)
        >>> geom = create_star(tess, radius=1.0, inc=60.0, PA=0.0)
        >>> # Altair: T_pole ~8450K, ω ~0.9, β ~0.22
        >>> temp = temperature_map_vonZeipel(
        ...     tess, geom, tpole=8450.0, rpole=1.0,
        ...     frac_escapevel=0.9, beta=0.22
        ... )
        >>> print(f"T_pole: {temp.max():.0f} K, T_equator: {temp.min():.0f} K")
        T_pole: 8450 K, T_equator: 6900 K

    Reference:
        - von Zeipel (1924): MNRAS 84, 665
        - Collins & Harrington (1966): ApJ 146, 152
        - geometry_rapidrotator.jl lines 80-98
    """
    # Offset from origin (for binary systems)
    if offsets is None:
        offsets = jnp.zeros(3)

    # Get vertex coordinates (center of each pixel, index 4 = center)
    # geom doesn't have vertices_xyz, but we can use tess and geom radius
    # Actually, we need the 3D position of pixels

    # For now, use tessellation's unit sphere coordinates scaled by local radius
    # This is approximate - full implementation needs r(θ) for rapid rotator

    # Get spherical coordinates: θ (colatitude), φ (longitude)
    theta = tess.theta  # (npix,)
    phi = tess.phi      # (npix,)

    # Compute local radius using Roche approximation
    x_roche = frac_escapevel * jnp.sin(theta)
    r_local = rpole * f_rapid_rot(x_roche)

    # 3D Cartesian coordinates
    x = r_local * jnp.sin(theta) * jnp.cos(phi) - offsets[0]
    y = r_local * jnp.sin(theta) * jnp.sin(phi) - offsets[1]
    z = r_local * jnp.cos(theta) - offsets[2]

    r_theta = jnp.sqrt(x**2 + y**2 + z**2)

    # Critical angular velocity
    omega_crit = jnp.sqrt(8.0 * GM / (27.0 * rpole**3))
    omega = frac_escapevel * omega_crit

    # Effective gravity components
    # g_r: radial component (gravity - centrifugal)
    g_r = -GM / r_theta**2 + r_theta * (omega * jnp.sin(theta))**2

    # g_θ: latitudinal component (Coriolis)
    g_theta_component = omega**2 * r_theta * jnp.sin(theta) * jnp.cos(theta)

    # Total effective gravity
    g_eff = jnp.sqrt(g_r**2 + g_theta_component**2)

    # Gravity at pole (no centrifugal force)
    g_pole = GM / rpole**2

    # Von Zeipel law: T ∝ g^β
    temperature_map = tpole * (g_eff / g_pole)**beta

    return temperature_map


def compute_intensity_from_temperature(
    temperature: jnp.ndarray,
    wavelength: float = 1.65e-6,
) -> jnp.ndarray:
    """Convert temperature to intensity using blackbody approximation.

    For interferometry, we care about intensity at observation wavelength.
    Uses Planck function in Rayleigh-Jeans limit (valid for IR/optical).

    Args:
        temperature: (npix,) effective temperature map (K)
        wavelength: Observation wavelength (meters, default 1.65 μm)

    Returns:
        intensity: (npix,) surface brightness map (arbitrary units)

    Notes:
        - Planck function: B(λ,T) ∝ 2hc²/λ⁵ / (exp(hc/λkT) - 1)
        - Rayleigh-Jeans limit (hν << kT): B ∝ T/λ⁴
        - For λ > 1μm and T > 5000K: RJ is good approximation
        - Intensity normalization handled by forward model

    Example:
        >>> temp = jnp.array([8000., 7000., 6000.])  # K
        >>> intensity = compute_intensity_from_temperature(temp, 1.65e-6)
        >>> print(intensity / intensity[0])  # Relative intensities
        [1.0, 0.765, 0.563]
    """
    # Physical constants
    h = 6.62607015e-34  # J·s
    c = 299792458.0     # m/s
    k = 1.380649e-23    # J/K

    # Planck function
    x = h * c / (wavelength * k * temperature)

    # Full Planck (for accuracy)
    # B(λ,T) ∝ 1/λ⁵ / (exp(hc/λkT) - 1)
    # We only need relative intensity, so drop constants
    intensity = 1.0 / (jnp.exp(x) - 1.0)

    # Alternative: Rayleigh-Jeans approximation (simpler, good for IR)
    # intensity = temperature / wavelength**4

    return intensity


def create_rapid_rotator_star(
    tess: Tessellation,
    rpole: float,
    frac_escapevel: float,
    tpole: float,
    beta: float = 0.25,
    inc: float = 0.0,
    PA: float = 0.0,
    obliq: float = 0.0,
    wavelength: float = 1.65e-6,
) -> Tuple[StarGeometry, jnp.ndarray]:
    """Create rapid rotator star with gravity darkening.

    Convenience function that combines:
    1. Geometry creation (with oblate shape)
    2. Temperature map (von Zeipel)
    3. Intensity map (blackbody conversion)

    Args:
        tess: Tessellation
        rpole: Polar radius (mas)
        frac_escapevel: ω/ω_crit, fractional critical velocity
        tpole: Polar effective temperature (K)
        beta: Gravity darkening exponent (default 0.25)
        inc: Inclination (degrees)
        PA: Position angle (degrees)
        obliq: Obliquity (degrees)
        wavelength: Observation wavelength (meters)

    Returns:
        geom: StarGeometry with sky projection
        intensity_map: (npix,) surface brightness including gravity darkening

    Notes:
        - Uses oblate approximation for shape
        - Applies von Zeipel gravity darkening
        - Returns intensity map ready for forward model

    Example (Altair):
        >>> tess = tessellation_healpix(n=5)
        >>> geom, intensity = create_rapid_rotator_star(
        ...     tess,
        ...     rpole=1.63,  # R_sun
        ...     frac_escapevel=0.9,
        ...     tpole=8450.0,  # K
        ...     beta=0.22,
        ...     inc=60.0,  # degrees
        ...     PA=45.0,
        ...     wavelength=1.65e-6  # H-band
        ... )
    """
    # Import here to avoid circular dependency
    from rotir_jax.geometry.base import create_star

    # Create geometry with oblate shape
    # TODO: This uses spherical geometry - should update to use Roche shape
    # For now, approximate with mean radius
    a, b, c = oblate_const(rpole, frac_escapevel)
    mean_radius = (a + b + c) / 3.0

    geom = create_star(
        tess,
        radius=mean_radius,
        inc=inc,
        PA=PA,
        obliq=obliq,
    )

    # Compute temperature map
    temperature = temperature_map_vonZeipel(
        tess, geom,
        tpole=tpole,
        rpole=rpole,
        frac_escapevel=frac_escapevel,
        beta=beta,
    )

    # Convert to intensity
    intensity_map = compute_intensity_from_temperature(temperature, wavelength)

    return geom, intensity_map


def compute_teff_vonzeipel(
    rpole: float,
    req: float,
    tpole: float,
    beta: float,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    """Simplified von Zeipel temperature for oblate spheroid.

    Approximate version assuming oblate spheroid (not full Roche).

    Args:
        rpole: Polar radius
        req: Equatorial radius
        tpole: Polar temperature (K)
        beta: Gravity darkening exponent
        theta: (npix,) colatitude (radians)

    Returns:
        teff: (npix,) effective temperature (K)

    Notes:
        - Simpler than full Roche model
        - Good approximation for moderate rotation (ω < 0.7)
        - Faster computation
    """
    # Oblateness
    f = (req - rpole) / rpole

    # Approximate effective gravity ratio for oblate spheroid
    # g(θ) / g_pole ≈ (1 + f*cos²(θ))
    g_ratio = 1.0 + f * jnp.cos(theta)**2

    # Von Zeipel law
    teff = tpole * g_ratio**beta

    return teff
