"""Posterior analysis utilities for ROTIR Bayesian inference.

Provides tools for analyzing posterior samples:
- Summary statistics (mean, std, quantiles) for all parameters
- Credible intervals for temperature maps and geometric parameters
- Per-pixel uncertainty maps
- Convergence diagnostics

These utilities work with output from any inference method
(MGVI, NUTS, MAP + Laplace approximation).
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class PosteriorSummary:
    """Summary of posterior inference results.

    Attributes:
        temperature_map_mean: (npix,) mean temperature at each pixel.
        temperature_map_std: (npix,) standard deviation at each pixel.
        temperature_map_quantiles: Dict of quantile arrays (e.g., 0.16, 0.84).
        geometric_params: Dict of {name: {'mean': float, 'std': float, 'ci_68': tuple, 'ci_95': tuple}}.
        chi2_reduced: Reduced chi-squared at posterior mean.
        n_samples: Number of posterior samples used.
        n_effective: Effective sample size (if MCMC).
    """
    temperature_map_mean: jnp.ndarray
    temperature_map_std: jnp.ndarray
    temperature_map_quantiles: Dict[float, jnp.ndarray]
    geometric_params: Dict[str, Dict[str, Any]]
    chi2_reduced: float
    n_samples: int
    n_effective: Optional[Dict[str, float]] = None


def summarize_posterior(
    samples: List[Dict[str, jnp.ndarray]],
    likelihood=None,
    quantiles: Tuple[float, ...] = (0.025, 0.16, 0.50, 0.84, 0.975),
    geometric_param_names: Optional[List[str]] = None,
) -> PosteriorSummary:
    """Compute posterior summary statistics from samples.

    Args:
        samples: List of parameter dictionaries (posterior samples).
        likelihood: Optional RotirLikelihood for chi2 computation.
        quantiles: Quantiles to compute (default: 2.5%, 16%, 50%, 84%, 97.5%).
        geometric_param_names: Names of geometric parameters to summarize.
            Default: ['diameter', 'inclination', 'position_angle'].

    Returns:
        PosteriorSummary with mean, std, quantiles, and diagnostics.
    """
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError("No samples provided")

    if geometric_param_names is None:
        geometric_param_names = ['diameter', 'inclination', 'position_angle']

    # Stack temperature maps
    temp_maps = jnp.stack([s['temperature_map'] for s in samples])

    # Temperature map statistics
    temp_mean = jnp.mean(temp_maps, axis=0)
    temp_std = jnp.std(temp_maps, axis=0)
    temp_quantiles = {}
    for q in quantiles:
        temp_quantiles[q] = jnp.quantile(temp_maps, q, axis=0)

    # Geometric parameter statistics
    geom_summary = {}
    for name in geometric_param_names:
        values = []
        for s in samples:
            if name in s:
                values.append(float(s[name]))
        if values:
            arr = np.array(values)
            geom_summary[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'ci_68': (float(np.percentile(arr, 16)), float(np.percentile(arr, 84))),
                'ci_95': (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
                'samples': arr,
            }

    # Compute chi2 at posterior mean
    chi2_red = 0.0
    if likelihood is not None:
        mean_params = jax.tree.map(
            lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
            *samples,
        )
        chi2_red = float(likelihood.reduced_chi2(mean_params, n_params=0))

    return PosteriorSummary(
        temperature_map_mean=temp_mean,
        temperature_map_std=temp_std,
        temperature_map_quantiles=temp_quantiles,
        geometric_params=geom_summary,
        chi2_reduced=chi2_red,
        n_samples=n_samples,
    )


def compute_credible_intervals(
    samples: List[Dict[str, jnp.ndarray]],
    param_name: str,
    levels: Tuple[float, ...] = (0.68, 0.95),
) -> Dict[float, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Compute credible intervals for a parameter.

    Args:
        samples: List of posterior sample dicts.
        param_name: Name of the parameter.
        levels: Credible interval levels (e.g., 0.68 for 1-sigma).

    Returns:
        intervals: Dict mapping level to (lower, upper) arrays.
    """
    values = jnp.stack([s[param_name] for s in samples])
    intervals = {}
    for level in levels:
        alpha = (1 - level) / 2
        lower = jnp.quantile(values, alpha, axis=0)
        upper = jnp.quantile(values, 1 - alpha, axis=0)
        intervals[level] = (lower, upper)
    return intervals


def compute_pixel_significance(
    samples: List[Dict[str, jnp.ndarray]],
    reference_temperature: Optional[float] = None,
) -> jnp.ndarray:
    """Compute per-pixel significance of temperature deviations.

    Returns the number of sigma each pixel deviates from the
    reference temperature (or mean temperature if not specified).

    Args:
        samples: Posterior samples.
        reference_temperature: Reference T (K). Default: posterior mean.

    Returns:
        significance: (npix,) array of sigma values.
    """
    temp_maps = jnp.stack([s['temperature_map'] for s in samples])
    temp_mean = jnp.mean(temp_maps, axis=0)
    temp_std = jnp.std(temp_maps, axis=0)

    if reference_temperature is None:
        reference_temperature = jnp.mean(temp_mean)

    significance = jnp.abs(temp_mean - reference_temperature) / jnp.maximum(temp_std, 1e-10)
    return significance


def compute_spot_detection_probability(
    samples: List[Dict[str, jnp.ndarray]],
    threshold_sigma: float = 3.0,
    cool_only: bool = True,
) -> jnp.ndarray:
    """Compute probability of a spot at each pixel.

    For each pixel, computes the fraction of posterior samples
    where the temperature is more than threshold_sigma below
    the mean stellar temperature.

    Args:
        samples: Posterior samples.
        threshold_sigma: Detection threshold in sigma units.
        cool_only: Only detect cool spots (not hot faculae).

    Returns:
        probability: (npix,) array of spot probabilities [0, 1].
    """
    temp_maps = jnp.stack([s['temperature_map'] for s in samples])
    n_samples = temp_maps.shape[0]

    # Mean temperature per sample
    sample_means = jnp.mean(temp_maps, axis=1, keepdims=True)
    sample_stds = jnp.std(temp_maps, axis=1, keepdims=True)

    # Deviation from mean in sigma units
    deviations = (temp_maps - sample_means) / jnp.maximum(sample_stds, 1e-10)

    if cool_only:
        is_spot = deviations < -threshold_sigma
    else:
        is_spot = jnp.abs(deviations) > threshold_sigma

    probability = jnp.mean(is_spot.astype(jnp.float32), axis=0)
    return probability


def format_parameter_table(
    summary: PosteriorSummary,
) -> str:
    """Format posterior summary as a readable text table.

    Args:
        summary: PosteriorSummary from summarize_posterior.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("ROTIR Bayesian Inference - Posterior Summary")
    lines.append("=" * 72)
    lines.append(f"Posterior samples:  {summary.n_samples}")
    lines.append(f"Reduced chi^2:     {summary.chi2_reduced:.3f}")
    lines.append("")

    # Temperature map summary
    lines.append("Temperature Map:")
    lines.append(f"  Mean:     {float(jnp.mean(summary.temperature_map_mean)):8.1f} K")
    lines.append(f"  Range:    [{float(jnp.min(summary.temperature_map_mean)):8.1f}, "
                 f"{float(jnp.max(summary.temperature_map_mean)):8.1f}] K")
    lines.append(f"  Avg uncertainty: {float(jnp.mean(summary.temperature_map_std)):8.1f} K")
    lines.append(f"  Max uncertainty: {float(jnp.max(summary.temperature_map_std)):8.1f} K")
    lines.append("")

    # Geometric parameters
    if summary.geometric_params:
        lines.append("Geometric Parameters:")
        lines.append(f"  {'Parameter':<20s} {'Mean':>10s} {'Std':>10s} {'68% CI':>24s}")
        lines.append("  " + "-" * 64)
        for name, stats in summary.geometric_params.items():
            ci68 = stats['ci_68']
            lines.append(
                f"  {name:<20s} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                f"[{ci68[0]:>10.4f}, {ci68[1]:>10.4f}]"
            )
    lines.append("=" * 72)

    return "\n".join(lines)


def compute_correlation_matrix(
    samples: List[Dict[str, jnp.ndarray]],
    param_names: List[str],
) -> Tuple[jnp.ndarray, List[str]]:
    """Compute correlation matrix between scalar parameters.

    Args:
        samples: Posterior samples.
        param_names: Names of scalar parameters to include.

    Returns:
        corr_matrix: (n_params, n_params) correlation matrix.
        names: List of parameter names in matrix order.
    """
    n_samples = len(samples)
    values = []
    names = []

    for name in param_names:
        if name in samples[0] and samples[0][name].ndim == 0:
            vals = jnp.array([float(s[name]) for s in samples])
            values.append(vals)
            names.append(name)

    if not values:
        return jnp.array([[]]), []

    data = jnp.stack(values)  # (n_params, n_samples)
    # Correlation matrix
    data_centered = data - jnp.mean(data, axis=1, keepdims=True)
    cov = (data_centered @ data_centered.T) / (n_samples - 1)
    std = jnp.sqrt(jnp.diag(cov))
    outer_std = jnp.outer(std, std)
    corr = cov / jnp.where(outer_std > 0, outer_std, 1.0)

    return corr, names
