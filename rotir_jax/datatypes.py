"""Core data structures for ROTIR.

Design principles:
- Geometry parameters use a class hierarchy (base + geometry-specific subclasses)
- Runtime data (tessellation, stellar geometry, OI data) are plain dataclasses
- JAX arrays for anything entering the forward model
- NumPy arrays for one-time setup data
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
import jax.numpy as jnp


# ============================================================
# Geometry parameter hierarchy
# ============================================================

@dataclass
class GeometricParams:
    """Base parameters shared by all geometry types.

    Units:
        inclination: degrees (0 = pole-on, 90 = edge-on)
        position_angle: degrees (measured East of North)
        rotation_period: days
    """
    inclination: float  # degrees
    position_angle: float  # degrees
    rotation_period: float  # days

    # Limb darkening
    ld_type: Literal[1, 2, 3] = 1  # 1=linear, 2=quadratic, 3=Hestroffer
    ld1: float = 0.0
    ld2: float = 0.0


@dataclass
class SphereParams(GeometricParams):
    """Simple sphere."""
    surface_type: int = field(default=0, init=False)
    radius: float = 1.0  # mas


@dataclass
class EllipsoidParams(GeometricParams):
    """Triaxial ellipsoid."""
    surface_type: int = field(default=1, init=False)
    radius_x: float = 1.0  # mas
    radius_y: float = 1.0  # mas
    radius_z: float = 1.0  # mas
    tpole: float = 10000.0  # K (for von Zeipel)
    beta: float = 0.25  # gravity darkening exponent


@dataclass
class RapidRotatorParams(GeometricParams):
    """Rapidly rotating star (Roche model for single star rotation)."""
    surface_type: int = field(default=2, init=False)
    rpole: float = 1.0  # mas (polar radius)
    frac_escapevel: float = 0.0  # ω/ω_crit, dimensionless [0, 1)
    tpole: float = 10000.0  # K (for von Zeipel)
    beta: float = 0.25  # gravity darkening exponent


@dataclass
class RocheParams(GeometricParams):
    """Roche lobe geometry (single visible star in a binary).

    The companion is treated as a point mass. The Roche geometry
    is parameterized by orbital elements + mass ratio.

    Units:
        rpole: mas (polar radius of the visible star)
        a: mas (semi-major axis of the binary orbit)
        P: days (orbital period)
        T0: JD (time of periastron passage)
        All angles: degrees unless noted
    """
    surface_type: int = field(default=3, init=False)

    # Stellar
    rpole: float = 1.0  # mas (polar radius)
    tpole: float = 10000.0  # K
    beta: float = 0.25  # gravity darkening exponent

    # Binary orbital
    a: float = 5.0  # mas (semi-major axis)
    P: float = 100.0  # days (orbital period)
    T0: float = 0.0  # JD (periastron time)
    e: float = 0.0  # eccentricity
    omega: float = 0.0  # argument of periapsis (degrees)
    Omega: float = 0.0  # longitude of ascending node (degrees, NOT USED for single star)
    q: float = 1.0  # mass ratio M2/M1
    dP: float = 0.0  # period derivative (days/day)

    # Roche-specific
    fillout_factor: float = -1.0  # if > 0, use fillout instead of rpole to define surface

    # Async rotation: ratio of stellar rotation to orbital rotation
    # async_ratio = P / rotation_period
    # (rotation_period is inherited from GeometricParams)


# ============================================================
# Tessellation
# ============================================================

@dataclass
class Tessellation:
    """HEALPix tessellation of the unit sphere.

    Arrays:
        unit_xyz: (npix, 5, 3) — xyz coords of 4 vertices + center, on unit sphere
        unit_spherical: (npix, 5, 3) — (r, theta, phi) for each vertex + center

    Index convention for the 5 points per pixel:
        0-3: four quadrilateral vertices (N, W, S, E in HEALPix convention)
        4: pixel center

    Coordinate convention:
        theta: colatitude [0, pi], 0 = North Pole
        phi: longitude [0, 2*pi], increasing East
        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)
    """
    tessellation_type: int  # 0=HEALPix, 1=LatLong
    npix: int
    nside: int  # HEALPix nside (nside = 2^n)
    n: int  # HEALPix order (n in nside=2^n)
    unit_xyz: np.ndarray  # (npix, 5, 3) float32
    unit_spherical: np.ndarray  # (npix, 5, 3) float32


# ============================================================
# Stellar geometry (per-epoch, after rotation + projection)
# ============================================================

@dataclass
class StellarGeometry:
    """Geometry of a star at a specific epoch, after rotation and projection.

    This is the result of: tessellation + params + epoch → rotated, projected star.

    Arrays:
        vertices_xyz: (npix, 5, 3) — rotated xyz coordinates
        vertices_spherical: (npix, 5, 3) — (r, theta, phi) before rotation
        normals: (npix, 3) — outward normal of each pixel face
        projx: (nvis, 4) — x-coords of visible pixel vertices on sky plane (mas)
        projy: (nvis, 4) — y-coords of visible pixel vertices on sky plane (mas)
        visible_idx: (nvis,) — indices of visible pixels into the full npix array
        ldmap: (npix,) — limb darkening weight per pixel (full array)
        polyflux: (nvis,) — pixel projected areas (set later by polyft module)
        polyft: (nuv, nvis) complex — polygon FT matrix (set later)
    """
    surface_type: int
    npix: int
    nvis: int
    vertices_xyz: jnp.ndarray  # (npix, 5, 3)
    vertices_spherical: np.ndarray  # (npix, 5, 3)
    normals: jnp.ndarray  # (npix, 3)
    visible_idx: jnp.ndarray  # (nvis,) int
    projx: jnp.ndarray  # (nvis, 4)
    projy: jnp.ndarray  # (nvis, 4)
    ldmap: jnp.ndarray  # (npix,)
    epoch: float  # time of this geometry

    # These are set later by setup_polygon_ft:
    polyflux: Optional[jnp.ndarray] = None  # (nvis,)
    polyft: Optional[jnp.ndarray] = None  # (nuv, nvis) complex64


# ============================================================
# Interferometric data
# ============================================================

@dataclass
class OIData:
    """Interferometric observables from one OIFITS file / epoch.

    UV coordinates:
        uv: (2, nuv) — (u, v) spatial frequencies in units that match
            the conversion factor: kx = uv[0,:] * (-pi / (180*3600*1000))
            i.e., uv is in cycles/radian if you want, but typically
            the OIFITS stores them in meters and the wavelength scaling
            converts them. The key is: uv values here are ALREADY in
            the correct units such that the Julia convention holds:
                kx = uv[0,:] * (-pi / (180*3600000))
            This means uv is in "1/mas" effective units.

            ACTUALLY: OIFITS stores u,v in meters. The conversion to
            spatial frequency happens in setup_polyft. The Julia code does:
                kx = data.uv[1,:] * Float32(-pi / (180*3600000))
                ky = data.uv[2,:] * Float32(-pi / (180*3600000)) # NOTE: see gotchas
            So uv should be stored as-is from OIFITS (in meters/wavelength = cycles/radian).

            CORRECTION: Looking at the Julia code more carefully:
            uv is stored as u/lambda, v/lambda in units of 1/radian.
            The OITOOLS package pre-divides by wavelength.
            The conversion factor -pi/(180*3600*1000) converts from
            1/radian to 1/mas (with the pi factor for the sinc convention).

            For our Python code: store uv in OIFITS native units (1/radian,
            already divided by wavelength), and apply the same conversion
            in setup_polyft.

    Indexing arrays (all 0-based in Python, were 1-based in Julia):
        indx_v2: (nv2,) — index into cvis for each V² measurement
        indx_t3_1/2/3: (nt3,) — three baseline indices for each closure phase
    """
    # V² observables
    v2: jnp.ndarray  # (nv2,)
    v2_err: jnp.ndarray  # (nv2,)
    nv2: int

    # Closure phase observables
    t3phi: jnp.ndarray  # (nt3,)
    t3phi_err: jnp.ndarray  # (nt3,)
    nt3: int

    # T3 amplitude (optional, often not used in chi2)
    t3amp: jnp.ndarray  # (nt3,)
    t3amp_err: jnp.ndarray  # (nt3,)

    # UV coordinates
    uv: jnp.ndarray  # (2, nuv) — spatial frequencies
    nuv: int

    # Indexing: which UV points correspond to which observables
    indx_v2: jnp.ndarray  # (nv2,) int — index into cvis array
    indx_t3_1: jnp.ndarray  # (nt3,) int
    indx_t3_2: jnp.ndarray  # (nt3,) int
    indx_t3_3: jnp.ndarray  # (nt3,) int

    # Metadata
    mean_mjd: float = 0.0
    filename: str = ""
