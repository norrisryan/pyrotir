"""ROTIR JAX: Python/JAX port of ROTIR for stellar surface reconstruction.

ROTIR is a package for stellar surface reconstruction from optical interferometry data.
This Python/JAX port provides GPU acceleration and modern optimization tools.

Includes Bayesian inference via NIFTy8.re for posterior sampling of temperature
maps and geometric parameters with full uncertainty quantification.
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

# Inference module (lazy import to avoid hard dependency on nifty8/blackjax)
from .inference.roche_diff import (
    solve_roche_radius_diff,
    compute_roche_shape_diff,
)
from .inference.nifty_likelihood import (
    RotirLikelihood,
    create_nifty_likelihood,
)
from .inference.bayesian_model import (
    BayesianStellarModel,
    GeometricPrior,
    TemperatureFieldConfig,
    run_inference,
)
from .inference.posterior import (
    PosteriorSummary,
    summarize_posterior,
    compute_credible_intervals,
)

__version__ = "0.1.0"

__all__ = [
    # Data types
    "GeometricParams",
    "SphereParams",
    "EllipsoidParams",
    "RapidRotatorParams",
    "RocheParams",
    "Tessellation",
    "StellarGeometry",
    "OIData",
    # Inference
    "solve_roche_radius_diff",
    "compute_roche_shape_diff",
    "RotirLikelihood",
    "create_nifty_likelihood",
    "BayesianStellarModel",
    "GeometricPrior",
    "TemperatureFieldConfig",
    "run_inference",
    "PosteriorSummary",
    "summarize_posterior",
    "compute_credible_intervals",
]
