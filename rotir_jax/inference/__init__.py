"""Bayesian inference module for ROTIR.

Integrates NIFTy8.re for posterior sampling of stellar surface temperature
maps and geometric parameters (diameter, inclination, orientation) with
full uncertainty quantification.

Submodules:
    roche_diff: Differentiable Roche geometry via JAX custom VJP
    nifty_likelihood: Bridge between ROTIR forward model and NIFTy8.re
    bayesian_model: Full Bayesian model with correlated field priors
    posterior: Posterior analysis, summaries, and error bars
"""

from rotir_jax.inference.roche_diff import (
    solve_roche_radius_diff,
    compute_roche_shape_diff,
)
from rotir_jax.inference.nifty_likelihood import (
    RotirLikelihood,
    build_signal_response,
)
from rotir_jax.inference.bayesian_model import (
    BayesianStellarModel,
    run_inference,
)
from rotir_jax.inference.posterior import (
    PosteriorSummary,
    summarize_posterior,
    compute_credible_intervals,
)

__all__ = [
    "solve_roche_radius_diff",
    "compute_roche_shape_diff",
    "RotirLikelihood",
    "build_signal_response",
    "BayesianStellarModel",
    "run_inference",
    "PosteriorSummary",
    "summarize_posterior",
    "compute_credible_intervals",
]
