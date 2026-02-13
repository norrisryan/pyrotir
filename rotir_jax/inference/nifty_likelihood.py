"""NIFTy8.re likelihood bridge for ROTIR.

Wraps the existing ROTIR JAX forward model as a NIFTy8.re-compatible
likelihood for Bayesian inference. The forward model maps:

    (temperature_map, geometric_params) -> (V^2, T3amp, T3phi)

and the likelihood compares these to observed data with Gaussian errors.

NIFTy8.re (nifty8.re / jifty) is the JAX-native reimplementation of NIFTy.
It provides:
    - Correlated field models on arbitrary domains
    - MGVI (Metric Gaussian Variational Inference) sampling
    - geoVI (geometric Variational Inference) sampling
    - Posterior sample management and diagnostics

References:
    - Edenhofer et al. (2024): NIFTy.re - JAX-based NIFTy
    - Knollmueller & Ensslin (2019): MGVI
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass

from rotir_jax.datatypes import StellarGeometry, OIData
from rotir_jax.forward_model.observables import compute_observables
from rotir_jax.forward_model.polyft import (
    setup_polyft_matrix,
    polygon_area,
    mod360,
)


@dataclass
class ForwardModelConfig:
    """Configuration for the ROTIR forward model within NIFTy.

    Attributes:
        oi_data: Interferometric observations.
        base_geom: Template StellarGeometry (used for tessellation structure).
        npix: Number of surface pixels.
        use_closure_phase: Include closure phases in likelihood.
        use_t3_amplitude: Include triple-product amplitudes.
        temperature_bounds: Physical bounds on temperature (K).
    """
    oi_data: OIData
    base_geom: StellarGeometry
    npix: int
    use_closure_phase: bool = True
    use_t3_amplitude: bool = True
    temperature_bounds: Tuple[float, float] = (3000.0, 50000.0)


def build_signal_response(
    config: ForwardModelConfig,
) -> Callable:
    """Build the signal response function for NIFTy.

    Creates a pure JAX function that maps latent parameters to
    predicted observables. This is the forward model that NIFTy
    optimizes against.

    Args:
        config: Forward model configuration.

    Returns:
        signal_response: Function mapping parameter dict to predicted
            observables as a flat array.

    The parameter dict has structure:
        {
            'temperature_map': (npix,) array,  # surface temperatures
            'diameter': scalar,                 # angular diameter (mas)
            'inclination': scalar,              # inclination (radians)
            'position_angle': scalar,           # PA (radians)
        }

    The returned observable vector is:
        [v2_model, t3amp_model, t3phi_model]
    """
    oi_data = config.oi_data
    base_geom = config.base_geom

    def signal_response(params):
        """Map parameters to predicted observables."""
        temp_map = params['temperature_map']

        # Build geometry from current geometric parameters
        # Use the base geometry structure but update with new params
        geom = _update_geometry(
            base_geom,
            params.get('diameter', None),
            params.get('inclination', None),
            params.get('position_angle', None),
        )

        # Forward model: temperature map -> observables
        v2_model, t3amp_model, t3phi_model = compute_observables(
            temp_map, geom, oi_data
        )

        # Concatenate into single vector
        parts = [v2_model]
        if config.use_t3_amplitude:
            parts.append(t3amp_model)
        if config.use_closure_phase:
            parts.append(t3phi_model)

        return jnp.concatenate(parts)

    return signal_response


def _update_geometry(
    base_geom: StellarGeometry,
    diameter: Optional[jnp.ndarray],
    inclination: Optional[jnp.ndarray],
    position_angle: Optional[jnp.ndarray],
) -> StellarGeometry:
    """Update stellar geometry with new parameters.

    When geometric parameters change during inference, we need to
    recompute the rotation and projection. This creates a new
    StellarGeometry with updated sky-plane coordinates.

    For the initial version, we use the base geometry as-is when
    no updates are provided. The full version recomputes rotation
    and projection from scratch.

    Args:
        base_geom: Template geometry with tessellation structure.
        diameter: New angular diameter (mas), or None to keep base.
        inclination: New inclination (radians), or None to keep base.
        position_angle: New PA (radians), or None to keep base.

    Returns:
        Updated StellarGeometry.
    """
    if diameter is None and inclination is None and position_angle is None:
        return base_geom

    # Recompute rotation and projection from first principles
    from rotir_jax.geometry.base import rotation_matrix, apply_rotation

    # Get tessellation vertices from base geometry
    # Scale by new diameter if provided
    radius = diameter / 2.0 if diameter is not None else None

    if radius is not None:
        # Rescale vertices to new radius
        # base_geom.vertices_xyz is already rotated, so we need
        # the unit vertices. Approximate by rescaling.
        old_scale = jnp.max(jnp.abs(base_geom.vertices_xyz))
        new_scale = radius
        scale_factor = new_scale / jnp.where(old_scale > 0, old_scale, 1.0)
        vertices_xyz = base_geom.vertices_xyz * scale_factor
    else:
        vertices_xyz = base_geom.vertices_xyz

    if inclination is not None or position_angle is not None:
        inc_deg = jnp.rad2deg(inclination) if inclination is not None else 60.0
        pa_deg = jnp.rad2deg(position_angle) if position_angle is not None else 0.0

        rot_mat = rotation_matrix(inc_deg, pa_deg, 0.0)
        # We need unrotated vertices for re-rotation
        # For now, use the already-projected coordinates
        # Full implementation would store unrotated vertices
        vertices_xyz = apply_rotation(vertices_xyz, rot_mat)

    # Recompute visible mask and projections
    z_center = vertices_xyz[:, 4, 2]
    vis_mask = z_center > 0
    visible_idx = jnp.where(vis_mask, size=base_geom.nvis)[0]

    projx = vertices_xyz[visible_idx, :4, 0]
    projy = vertices_xyz[visible_idx, :4, 1]

    return StellarGeometry(
        surface_type=base_geom.surface_type,
        npix=base_geom.npix,
        nvis=base_geom.nvis,
        vertices_xyz=vertices_xyz,
        vertices_spherical=base_geom.vertices_spherical,
        normals=base_geom.normals,
        visible_idx=visible_idx,
        projx=projx,
        projy=projy,
        ldmap=base_geom.ldmap,
        epoch=base_geom.epoch,
        polyflux=base_geom.polyflux,
        polyft=base_geom.polyft,
    )


def build_data_vector(
    oi_data: OIData,
    use_t3_amplitude: bool = True,
    use_closure_phase: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build the observed data vector and noise standard deviations.

    Concatenates all observable types into a single vector matching
    the signal_response output ordering.

    Args:
        oi_data: Interferometric observations.
        use_t3_amplitude: Include T3 amplitudes.
        use_closure_phase: Include closure phases.

    Returns:
        data: Concatenated observed data vector.
        noise_std: Concatenated noise standard deviations.
    """
    data_parts = [oi_data.v2]
    noise_parts = [oi_data.v2_err]

    if use_t3_amplitude:
        data_parts.append(oi_data.t3amp)
        noise_parts.append(oi_data.t3amp_err)

    if use_closure_phase:
        data_parts.append(oi_data.t3phi)
        noise_parts.append(oi_data.t3phi_err)

    return jnp.concatenate(data_parts), jnp.concatenate(noise_parts)


class RotirLikelihood:
    """NIFTy8.re-compatible likelihood for ROTIR interferometric data.

    Wraps the ROTIR forward model into a likelihood function suitable
    for use with NIFTy8.re's optimize_kl or as a standalone JAX
    log-likelihood for other samplers (NUTS, HMC, etc.).

    The negative log-likelihood is:
        -log L = 0.5 * sum((signal_response(params) - data)^2 / sigma^2)

    with special handling for closure phase wrapping.

    Attributes:
        config: Forward model configuration.
        signal_response: JAX function mapping params to observables.
        data: Observed data vector.
        noise_std: Noise standard deviations.
        n_data: Total number of data points.
    """

    def __init__(
        self,
        config: ForwardModelConfig,
        signal_response: Optional[Callable] = None,
    ):
        """Initialize likelihood.

        Args:
            config: Forward model configuration.
            signal_response: Optional custom signal response function.
                If None, builds one from config.
        """
        self.config = config
        self.signal_response = signal_response or build_signal_response(config)

        # Build data and noise vectors
        self.data, self.noise_std = build_data_vector(
            config.oi_data,
            use_t3_amplitude=config.use_t3_amplitude,
            use_closure_phase=config.use_closure_phase,
        )
        self.n_data = len(self.data)

        # Compute index ranges for different observable types
        nv2 = config.oi_data.nv2
        nt3 = config.oi_data.nt3
        self._v2_slice = slice(0, nv2)
        offset = nv2
        if config.use_t3_amplitude:
            self._t3amp_slice = slice(offset, offset + nt3)
            offset += nt3
        else:
            self._t3amp_slice = None
        if config.use_closure_phase:
            self._t3phi_slice = slice(offset, offset + nt3)
        else:
            self._t3phi_slice = None

        # Precompute inverse noise variance
        self.noise_var_inv = 1.0 / (self.noise_std**2 + 1e-30)

    def __call__(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute negative log-likelihood (energy).

        This is the function NIFTy8.re calls during optimization.

        Args:
            params: Parameter dictionary with 'temperature_map' and
                optionally geometric parameters.

        Returns:
            Scalar negative log-likelihood value.
        """
        return self.neg_log_likelihood(params)

    def neg_log_likelihood(
        self,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Compute negative log-likelihood with closure phase wrapping.

        Args:
            params: Parameter dictionary.

        Returns:
            nll: Negative log-likelihood (scalar).
        """
        predicted = self.signal_response(params)
        residuals = predicted - self.data

        # Handle closure phase wrapping
        if self._t3phi_slice is not None:
            t3phi_residual = residuals[self._t3phi_slice]
            t3phi_wrapped = mod360(t3phi_residual)
            residuals = residuals.at[self._t3phi_slice].set(t3phi_wrapped)

        # Weighted chi-squared
        chi2 = jnp.sum(residuals**2 * self.noise_var_inv)
        return 0.5 * chi2

    def neg_log_likelihood_components(
        self,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Compute per-component negative log-likelihoods.

        Useful for diagnostics.

        Args:
            params: Parameter dictionary.

        Returns:
            Dictionary with 'nll_v2', 'nll_t3amp', 'nll_t3phi', 'nll_total'.
        """
        predicted = self.signal_response(params)
        residuals = predicted - self.data

        result = {}

        # V^2 component
        v2_res = residuals[self._v2_slice]
        result['nll_v2'] = 0.5 * jnp.sum(
            v2_res**2 * self.noise_var_inv[self._v2_slice]
        )

        # T3 amplitude component
        if self._t3amp_slice is not None:
            t3amp_res = residuals[self._t3amp_slice]
            result['nll_t3amp'] = 0.5 * jnp.sum(
                t3amp_res**2 * self.noise_var_inv[self._t3amp_slice]
            )

        # T3 phase component (with wrapping)
        if self._t3phi_slice is not None:
            t3phi_res = mod360(residuals[self._t3phi_slice])
            result['nll_t3phi'] = 0.5 * jnp.sum(
                t3phi_res**2 * self.noise_var_inv[self._t3phi_slice]
            )

        result['nll_total'] = sum(result.values())
        return result

    def gradient(
        self,
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute negative log-likelihood and its gradient.

        Args:
            params: Parameter dictionary.

        Returns:
            nll: Negative log-likelihood value.
            grad: Gradient dictionary with same structure as params.
        """
        nll_fn = self.neg_log_likelihood
        nll, grad = jax.value_and_grad(nll_fn)(params)
        return nll, grad

    def reduced_chi2(
        self,
        params: Dict[str, jnp.ndarray],
        n_params: int = 0,
    ) -> jnp.ndarray:
        """Compute reduced chi-squared.

        Args:
            params: Parameter dictionary.
            n_params: Number of free parameters (for DOF calculation).

        Returns:
            Reduced chi-squared value.
        """
        nll = self.neg_log_likelihood(params)
        chi2 = 2.0 * nll
        n_dof = self.n_data - n_params
        return chi2 / jnp.maximum(n_dof, 1.0)


def create_nifty_likelihood(
    oi_data: OIData,
    base_geom: StellarGeometry,
    npix: int,
    use_closure_phase: bool = True,
    use_t3_amplitude: bool = True,
) -> RotirLikelihood:
    """Convenience constructor for the ROTIR NIFTy likelihood.

    Args:
        oi_data: Interferometric observations.
        base_geom: Template stellar geometry.
        npix: Number of surface pixels.
        use_closure_phase: Include closure phases.
        use_t3_amplitude: Include T3 amplitudes.

    Returns:
        RotirLikelihood instance ready for NIFTy8.re.
    """
    config = ForwardModelConfig(
        oi_data=oi_data,
        base_geom=base_geom,
        npix=npix,
        use_closure_phase=use_closure_phase,
        use_t3_amplitude=use_t3_amplitude,
    )
    return RotirLikelihood(config)
