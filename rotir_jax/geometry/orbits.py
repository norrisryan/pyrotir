"""Orbital mechanics for ROTIR - binary star systems.

Implements Keplerian orbit calculations for binary stars:
- Kepler's equation solver (Newton-Raphson)
- Eccentric and true anomaly
- 3D orbital positions (absolute and relative)
- Orbital velocity
- Time-dependent geometry for multi-epoch observations

Key applications:
- Eclipsing binaries
- Spectroscopic binaries
- Visual binaries resolved by interferometry
- Mass transfer systems
- Symbiotic stars

References:
- Murray & Dermott (1999): Solar System Dynamics
- Broucke & Cefola (1972): True/eccentric anomaly relations
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

# Physical constants
TWO_PI = 2.0 * jnp.pi


def compute_E_NR(
    M: jnp.ndarray,
    e: float,
    max_iter: int = 50,
    thresh: float = 1e-6,
) -> jnp.ndarray:
    """Solve Kepler's equation using Newton-Raphson method.

    Kepler's equation: M = E - e*sin(E)
    where M is mean anomaly, E is eccentric anomaly, e is eccentricity.

    Args:
        M: Mean anomaly (radians)
        e: Orbital eccentricity (0 ≤ e < 1)
        max_iter: Maximum number of iterations (default 50)
        thresh: Convergence threshold (default 1e-6)

    Returns:
        E: Eccentric anomaly (radians)

    Notes:
        - Uses Smith (1979) initial guess for faster convergence
        - Different iteration scheme for high eccentricity (e > 0.5)
        - Vectorized for multiple epochs

    Reference:
        orbits.jl lines 2-26
    """
    # Initial guess from Smith (1979)
    E = M + e * jnp.sin(M) / (1.0 - jnp.sin(M + e) + jnp.sin(M))

    # Newton-Raphson iteration
    if e > 0.5:
        # High eccentricity: full Newton-Raphson
        for _ in range(max_iter):
            E_old = E
            # E_new = E - f(E)/f'(E) where f(E) = E - e*sin(E) - M
            dE = (M + e * jnp.sin(E) - E) / (1.0 - e * jnp.cos(E))
            E = E + dE

            # Check convergence
            if jnp.max(jnp.abs(E - E_old)) < thresh:
                break
    else:
        # Low eccentricity: simplified iteration
        for _ in range(max_iter):
            E_old = E
            E = M + e * jnp.sin(E_old)

            if jnp.max(jnp.abs(E - E_old)) < thresh:
                break

    return E


def compute_eccentric_anomaly(
    P: float,
    e: float,
    T0: float,
    tepoch: jnp.ndarray,
    dP: float = 0.0,
) -> jnp.ndarray:
    """Calculate eccentric anomaly at given epoch(s).

    Args:
        P: Orbital period (days)
        e: Eccentricity
        T0: Time of periastron passage (MJD)
        tepoch: Observation epoch(s) (MJD)
        dP: Period derivative dP/dt for decaying orbits (default 0.0)

    Returns:
        E: Eccentric anomaly (radians), range [0, 2π)

    Notes:
        - Handles orbital decay if dP > 0
        - For circular orbits (e=0): E = M
        - Vectorized for multiple epochs

    Reference:
        orbits.jl lines 134-149
    """
    # Mean motion
    n = TWO_PI / P

    # Mean anomaly
    if dP > 1e-12:
        # Orbital decay case
        dt = tepoch - T0
        M = 4.0 * jnp.pi * dt * (1.0 / (1.0 + jnp.sqrt(1.0 + 2.0 * dP * dt / P))) / P
    else:
        # Standard Keplerian orbit
        M = n * (tepoch - T0)

    # Solve Kepler's equation
    E = compute_E_NR(M, e)

    # Wrap to [0, 2π)
    E = jnp.mod(E, TWO_PI)

    return E


def compute_true_anomaly(
    P: float,
    e: float,
    T0: float,
    tepoch: jnp.ndarray,
    dP: float = 0.0,
) -> jnp.ndarray:
    """Calculate true anomaly from orbital elements.

    Args:
        P: Orbital period (days)
        e: Eccentricity
        T0: Time of periastron passage (MJD)
        tepoch: Observation epoch(s) (MJD)
        dP: Period derivative (default 0.0)

    Returns:
        υ: True anomaly (radians)

    Notes:
        - Uses Broucke & Cefola (1972) formula
        - Avoids numerical issues near υ = ±π
        - More stable than classic tan(υ/2) = sqrt((1+e)/(1-e)) * tan(E/2)

    Reference:
        - Broucke & Cefola (1972): Celestial Mechanics 5, 303
        - orbits.jl lines 152-163
    """
    # Eccentric anomaly
    E = compute_eccentric_anomaly(P, e, T0, tepoch, dP)

    # True anomaly using Broucke & Cefola formula
    # Avoids numerical issues when arguments are near ±π
    beta = e / (1.0 + jnp.sqrt(1.0 - e**2))
    υ = E + 2.0 * jnp.arctan((beta * jnp.sin(E)) / (1.0 - beta * jnp.cos(E)))

    # Alternative classic formula (commented out, less stable):
    # υ = 2.0 * jnp.arctan(jnp.sqrt((1+e)/(1-e)) * jnp.tan(E/2))

    return υ


def compute_orbital_coefficients(
    Omega: float,
    i: float,
    omega: float,
) -> Tuple[float, float, float, float, float, float]:
    """Compute transformation coefficients for orbital equations.

    These coefficients transform from orbital plane to sky plane.

    Args:
        Omega: Longitude of ascending node (radians)
        i: Orbital inclination (radians)
        omega: Argument of periastron (radians)

    Returns:
        L1, M1, N1: Coefficients for radial direction
        L2, M2, N2: Coefficients for tangential direction

    Notes:
        - Convention: x=North, y=East, z=toward observer
        - L,M,N transform orbital (r,θ) to sky (x,y,z)

    Reference:
        orbits.jl lines 28-36
    """
    cos_Omega = jnp.cos(Omega)
    sin_Omega = jnp.sin(Omega)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)
    cos_omega = jnp.cos(omega)
    sin_omega = jnp.sin(omega)

    L1 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
    M1 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
    N1 = sin_omega * sin_i

    L2 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
    M2 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
    N2 = cos_omega * sin_i

    return L1, M1, N1, L2, M2, N2


def compute_separation(
    a: float,
    e: float,
    P: float,
    T0: float,
    tepoch: jnp.ndarray,
    dP: float = 0.0,
) -> jnp.ndarray:
    """Compute instantaneous separation between binary components.

    Args:
        a: Semi-major axis (mas)
        e: Eccentricity
        P: Period (days)
        T0: Time of periastron (MJD)
        tepoch: Observation epoch(s) (MJD)
        dP: Period derivative (default 0.0)

    Returns:
        D: Dimensionless separation (actual separation = D * a)

    Notes:
        - D varies with orbital phase
        - At periastron: D = 1 - e
        - At apastron: D = 1 + e
        - For circular orbits: D = 1

    Reference:
        orbits.jl lines 119-124
    """
    E = compute_eccentric_anomaly(P, e, T0, tepoch, dP)
    D = 1.0 - e * jnp.cos(E)
    return D


def compute_separation_from_true_anomaly(
    a: float,
    e: float,
    P: float,
    T0: float,
    tepoch: jnp.ndarray,
) -> jnp.ndarray:
    """Compute separation using true anomaly (alternative method).

    Args:
        a: Semi-major axis (mas)
        e: Eccentricity
        P: Period (days)
        T0: Time of periastron (MJD)
        tepoch: Observation epoch(s) (MJD)

    Returns:
        D: Dimensionless separation

    Notes:
        - Uses r = a(1-e²)/(1+e*cos(υ))
        - Equivalent to compute_separation but via true anomaly

    Reference:
        orbits.jl lines 112-117
    """
    υ = compute_true_anomaly(P, e, T0, tepoch)
    D = (1.0 - e**2) / (1.0 + e * jnp.cos(υ))
    return D


def binary_orbit_relative(
    a: float,
    e: float,
    P: float,
    T0: float,
    Omega: float,
    i: float,
    omega: float,
    tepoch: float,
    dP: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Compute relative binary orbit (secondary relative to primary).

    Args:
        a: Semi-major axis (mas)
        e: Eccentricity
        P: Period (days)
        T0: Time of periastron (MJD)
        Omega: Longitude of ascending node (degrees)
        i: Inclination (degrees)
        omega: Argument of periastron (degrees)
        tepoch: Observation epoch (MJD)
        dP: Period derivative (default 0.0)

    Returns:
        x1, y1, z1: Position of primary (0, 0, 0 for relative orbit)
        x2, y2, z2: Position of secondary (mas)

    Notes:
        - Primary at origin
        - x: North (RA increasing)
        - y: East (Dec increasing)
        - z: toward observer (negative = away)

    Reference:
        orbits.jl lines 38-47
    """
    # Convert angles to radians
    Omega_rad = jnp.deg2rad(Omega)
    i_rad = jnp.deg2rad(i)
    omega_rad = jnp.deg2rad(omega)

    # Eccentric anomaly
    E = compute_eccentric_anomaly(P, e, T0, jnp.array([tepoch]), dP)[0]

    # Orbital coefficients
    L1, M1, N1, L2, M2, N2 = compute_orbital_coefficients(Omega_rad, i_rad, omega_rad)

    # Compute position
    beta = jnp.sqrt(1.0 - e**2)
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)

    x = a * (L1 * cos_E + beta * L2 * sin_E - e * L1)
    y = a * (M1 * cos_E + beta * M2 * sin_E - e * M1)
    z = a * (N1 * cos_E + beta * N2 * sin_E - e * N1)

    # Flip z convention: negative z = toward observer
    z = -z

    return 0.0, 0.0, 0.0, x, y, z


def binary_orbit_absolute(
    a: float,
    e: float,
    P: float,
    T0: float,
    q: float,
    Omega: float,
    i: float,
    omega: float,
    tepoch: float,
    dP: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Compute absolute binary orbit (both stars around center of mass).

    Args:
        a: Semi-major axis (mas)
        e: Eccentricity
        P: Period (days)
        T0: Time of periastron (MJD)
        q: Mass ratio M2/M1
        Omega: Longitude of ascending node (degrees)
        i: Inclination (degrees)
        omega: Argument of periastron (degrees)
        tepoch: Observation epoch (MJD)
        dP: Period derivative (default 0.0)

    Returns:
        x1, y1, z1: Position of primary (mas)
        x2, y2, z2: Position of secondary (mas)

    Notes:
        - Both stars orbit center of mass
        - Distances: r1 = D/(1/q+1), r2 = D/(1+q)
        - Center of mass at origin
        - q = M2/M1 (mass ratio)

    Reference:
        orbits.jl lines 65-81
    """
    # Convert angles to radians
    Omega_rad = jnp.deg2rad(Omega)
    i_rad = jnp.deg2rad(i)
    omega_rad = jnp.deg2rad(omega)

    # True anomaly
    υ = compute_true_anomaly(P, e, T0, jnp.array([tepoch]), dP)[0]

    # Instantaneous separation
    D = a * (1.0 - e**2) / (1.0 + e * jnp.cos(υ))

    # Distance from center of mass
    r1 = D / (1.0 / q + 1.0)  # Primary
    r2 = D / (1.0 + q)         # Secondary

    # Orbital coefficients
    L1, M1, N1, L2, M2, N2 = compute_orbital_coefficients(Omega_rad, i_rad, omega_rad)

    # Primary position
    x1 = r1 * (L1 * jnp.cos(υ) + L2 * jnp.sin(υ))
    y1 = r1 * (M1 * jnp.cos(υ) + M2 * jnp.sin(υ))
    z1 = r1 * (N1 * jnp.cos(υ) + N2 * jnp.sin(υ))

    # Secondary position (opposite side of COM)
    # Phase shifted by π
    υ2 = υ + jnp.pi if (υ >= 0.0 and υ <= jnp.pi) else υ - jnp.pi

    x2 = r2 * (L1 * jnp.cos(υ2) + L2 * jnp.sin(υ2))
    y2 = r2 * (M1 * jnp.cos(υ2) + M2 * jnp.sin(υ2))
    z2 = r2 * (N1 * jnp.cos(υ2) + N2 * jnp.sin(υ2))

    # Flip z: negative = toward observer
    z1 = -z1
    z2 = -z2

    return x1, y1, z1, x2, y2, z2


def compute_orbital_phase(
    P: float,
    T0: float,
    tepoch: jnp.ndarray,
) -> jnp.ndarray:
    """Compute orbital phase at given epoch(s).

    Args:
        P: Orbital period (days)
        T0: Zero-point epoch (MJD)
        tepoch: Observation epoch(s) (MJD)

    Returns:
        phase: Orbital phase [0, 1], where 0 = T0

    Notes:
        - Phase 0.0 = periastron (if T0 is time of periastron)
        - Phase 0.5 = apastron
        - Useful for phase-folded observations
    """
    phase = jnp.mod((tepoch - T0) / P, 1.0)
    return phase


def compute_radial_velocity(
    K: float,
    e: float,
    P: float,
    T0: float,
    omega: float,
    tepoch: jnp.ndarray,
    gamma: float = 0.0,
) -> jnp.ndarray:
    """Compute radial velocity for spectroscopic binary.

    Args:
        K: Velocity semi-amplitude (km/s)
        e: Eccentricity
        P: Period (days)
        T0: Time of periastron (MJD)
        omega: Argument of periastron (degrees)
        tepoch: Observation epoch(s) (MJD)
        gamma: Systemic velocity (km/s, default 0)

    Returns:
        RV: Radial velocity (km/s)

    Notes:
        - RV = γ + K[cos(ω+υ) + e*cos(ω)]
        - K related to masses and inclination
        - For double-lined: K1*M1 = K2*M2

    Example:
        >>> # Algol-like system
        >>> K = 50.0  # km/s
        >>> tepochs = jnp.linspace(0, 100, 1000)  # 100 days
        >>> RV = compute_radial_velocity(K, 0.1, 2.87, 0.0, 45.0, tepochs)
    """
    # True anomaly
    υ = compute_true_anomaly(P, e, T0, tepoch)

    # Argument of periastron in radians
    omega_rad = jnp.deg2rad(omega)

    # Radial velocity equation
    RV = gamma + K * (jnp.cos(omega_rad + υ) + e * jnp.cos(omega_rad))

    return RV


def compute_masses_from_orbit(
    P: float,
    a: float,
    i: float,
    distance: float,
) -> Tuple[float, float]:
    """Compute total mass from orbital parameters (Kepler's 3rd law).

    Args:
        P: Orbital period (days)
        a: Semi-major axis (mas)
        i: Inclination (degrees)
        distance: Distance to system (pc)

    Returns:
        M_total: Total system mass (solar masses)
        a_au: Semi-major axis (AU)

    Notes:
        - Kepler's 3rd law: M_total = a³/P² (in solar units)
        - Requires distance to convert mas → AU
        - Individual masses need mass ratio q

    Example:
        >>> # Binary at 10 pc with a=100 mas, P=1 year
        >>> M_total, a_au = compute_masses_from_orbit(365.25, 100, 90, 10)
        >>> print(f"Total mass: {M_total:.2f} M_sun")
    """
    # Convert mas to AU
    a_au = a * distance / 1000.0  # mas * pc / 1000 = AU

    # Kepler's 3rd law: P² = a³ / M_total (with P in years, a in AU, M in M_sun)
    P_years = P / 365.25
    M_total = a_au**3 / P_years**2

    return M_total, a_au
