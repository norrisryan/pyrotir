"""ROTIR JAX: Python/JAX port of ROTIR for stellar surface reconstruction.

ROTIR is a package for stellar surface reconstruction from optical interferometry data.
This Python/JAX port provides GPU acceleration and modern optimization tools.
"""

from .datatypes import (
    GeometricParams,
    SphereParams,
    EllipsoidParams,
    RapidRotatorParams,
    RocheParams,
    Tessellation,
    StellarGeometry,
    OIData,
)

__version__ = "0.1.0"

__all__ = [
    "GeometricParams",
    "SphereParams",
    "EllipsoidParams",
    "RapidRotatorParams",
    "RocheParams",
    "Tessellation",
    "StellarGeometry",
    "OIData",
]
