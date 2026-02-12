"""Multi-epoch reconstruction for ROTIR - time-series stellar imaging.

Enables reconstruction from observations at multiple epochs:
1. Stellar rotation: different phases → surface map
2. Binary orbits: orbital motion → component separation
3. Evolution: spot emergence, decay, migration
4. Differential rotation: latitude-dependent rotation

Key concept: Joint reconstruction across epochs
- Single intensity map (if static)
- OR separate maps per epoch (if evolving)
- Orbital parameters (for binaries)
- Rotation phases

Applications:
- Doppler imaging time series
- Multi-night interferometry
- Binary star orbits
- Spot evolution tracking
- Differential rotation measurement

References:
- Vogt et al. (1987): Doppler imaging
- Collier Cameron & Unruh (1994): Differential rotation
- Donati et al. (1997): Zeeman-Doppler imaging
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass

import sys
sys.path.append('..')
from rotir_jax.datatypes import OIData, Star
from rotir_jax.geometry.base import apply_rotation
from rotir_jax.reconstruction.optimizer import (
    StellarImageReconstructor,
    OptimizationResult,
)


@dataclass
class Epoch:
    """Single observation epoch.

    Attributes:
        oi_data: Interferometric data for this epoch
        rotation_phase: Rotation phase [0, 1] (0 = reference phase)
        orbital_phase: Orbital phase [0, 1] (for binaries)
        mjd: Modified Julian Date
        weight: Epoch weight (default 1.0)
    """
    oi_data: OIData
    rotation_phase: float = 0.0
    orbital_phase: float = 0.0
    mjd: float = 0.0
    weight: float = 1.0


class MultiEpochReconstructor:
    """Multi-epoch image reconstruction.

    Reconstructs stellar surface from observations at multiple epochs.

    Two modes:
    1. **Static reconstruction**: Single map, different viewing angles
       - Assumes surface doesn't change
       - Different rotation/orbital phases
       - Best for short time series (hours-days)

    2. **Dynamic reconstruction**: Separate maps per epoch
       - Allows surface evolution
       - Temporal regularization between epochs
       - Best for long time series (weeks-months)

    Example (static):
        >>> # Observations at 4 rotation phases
        >>> epochs = [
        ...     Epoch(oi_data=data1, rotation_phase=0.00),
        ...     Epoch(oi_data=data2, rotation_phase=0.25),
        ...     Epoch(oi_data=data3, rotation_phase=0.50),
        ...     Epoch(oi_data=data4, rotation_phase=0.75),
        ... ]
        >>>
        >>> reconstructor = MultiEpochReconstructor(
        ...     epochs=epochs,
        ...     star_template=star,
        ...     mode="static",
        ... )
        >>>
        >>> result = reconstructor.reconstruct(maxiter=200)
        >>> surface_map = result.x_solution  # Single map

    Example (dynamic):
        >>> # Observations over 30 days
        >>> epochs = [Epoch(oi_data=data, mjd=mjd) for data, mjd in zip(...]
        >>>
        >>> reconstructor = MultiEpochReconstructor(
        ...     epochs=epochs,
        ...     star_template=star,
        ...     mode="dynamic",
        ...     temporal_regularization=0.01,
        ... )
        >>>
        >>> result = reconstructor.reconstruct(maxiter=200)
        >>> maps = result.x_solution.reshape(n_epochs, npix)  # One map per epoch
    """

    def __init__(
        self,
        epochs: List[Epoch],
        star_template: Star,
        regularizers: Optional[List[Dict]] = None,
        mode: str = "static",
        temporal_regularization: float = 0.0,
        epoch_weights: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Initialize multi-epoch reconstructor.

        Args:
            epochs: List of observation epochs
            star_template: Template star geometry
            regularizers: Spatial regularizers (applied per epoch)
            mode: "static" (single map) or "dynamic" (map per epoch)
            temporal_regularization: Weight for temporal smoothness (dynamic mode)
            epoch_weights: Weight for each epoch (default: equal)
            verbose: Print progress
        """
        self.epochs = epochs
        self.star_template = star_template
        self.regularizers = regularizers or []
        self.mode = mode
        self.temporal_reg = temporal_regularization
        self.verbose = verbose

        self.n_epochs = len(epochs)
        self.npix = star_template.tess.npix

        # Epoch weights
        if epoch_weights is None:
            self.epoch_weights = np.ones(self.n_epochs) / self.n_epochs
        else:
            self.epoch_weights = epoch_weights / np.sum(epoch_weights)

        # Parameter size
        if mode == "static":
            self.n_params = self.npix  # Single map
        elif mode == "dynamic":
            self.n_params = self.n_epochs * self.npix  # Map per epoch
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'static' or 'dynamic'")

        if self.verbose:
            print(f"Multi-epoch reconstructor ({mode} mode):")
            print(f"  Epochs: {self.n_epochs}")
            print(f"  Pixels: {self.npix}")
            print(f"  Parameters: {self.n_params}")

    def objective_function(
        self,
        x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Multi-epoch objective function.

        Args:
            x: Image parameters (npix or n_epochs × npix)

        Returns:
            f: Total objective (sum over epochs + regularization)
            g: Gradient
        """
        x_jax = jnp.array(x)

        # Extract maps for each epoch
        if self.mode == "static":
            # Same map for all epochs
            maps = [x_jax for _ in range(self.n_epochs)]
        else:  # dynamic
            # Separate map per epoch
            maps = [x_jax[i*self.npix:(i+1)*self.npix] for i in range(self.n_epochs)]

        # Compute χ² for each epoch
        total_chi2 = 0.0
        total_grad = jnp.zeros_like(x_jax)

        for i, (epoch, intensity_map) in enumerate(zip(self.epochs, maps)):
            # Rotate star to this epoch's phase
            star_epoch = self._rotate_star(
                intensity_map,
                epoch.rotation_phase,
                epoch.orbital_phase,
            )

            # Create single-epoch reconstructor
            reconstructor_i = StellarImageReconstructor(
                oi_data=epoch.oi_data,
                star=star_epoch,
                regularizers=self.regularizers,
                verbose=False,
            )

            # Compute χ² and gradient
            f_i, g_i = reconstructor_i.objective_function(np.array(intensity_map))

            # Weighted sum
            total_chi2 += self.epoch_weights[i] * f_i

            if self.mode == "static":
                # Accumulate gradients (all epochs contribute)
                total_grad += self.epoch_weights[i] * jnp.array(g_i)
            else:  # dynamic
                # Gradient for this epoch's map
                idx_start = i * self.npix
                idx_end = (i + 1) * self.npix
                total_grad = total_grad.at[idx_start:idx_end].set(
                    self.epoch_weights[i] * jnp.array(g_i)
                )

        # Temporal regularization (dynamic mode only)
        f_temporal = 0.0
        if self.mode == "dynamic" and self.temporal_reg > 0:
            f_temporal, g_temporal = self._temporal_smoothness(x_jax)
            f_temporal *= self.temporal_reg
            total_grad += self.temporal_reg * g_temporal

        # Total objective
        f_total = total_chi2 + f_temporal

        if self.verbose and (np.random.rand() < 0.1):  # Print 10% of iterations
            print(f"χ²={total_chi2:.2f}, temporal={f_temporal:.2f}, total={f_total:.2f}")

        return float(f_total), np.array(total_grad)

    def _rotate_star(
        self,
        intensities: jnp.ndarray,
        rotation_phase: float,
        orbital_phase: float,
    ) -> Star:
        """Rotate star to given phase.

        Args:
            intensities: Surface intensities
            rotation_phase: Rotation phase [0, 1]
            orbital_phase: Orbital phase [0, 1]

        Returns:
            star: Rotated star model
        """
        # Rotation angle (radians)
        rotation_angle = 2 * np.pi * rotation_phase

        # Apply rotation to coordinates
        # TODO: Implement proper rotation
        # For now, just update intensities

        star_rotated = Star(
            tess=self.star_template.tess,
            theta=self.star_template.theta,
            phi=self.star_template.phi,
            x=self.star_template.x,
            y=self.star_template.y,
            z=self.star_template.z,
            visible=self.star_template.visible,
            intensities=intensities,
            ld_coeffs=self.star_template.ld_coeffs,
        )

        return star_rotated

    def _temporal_smoothness(
        self,
        x: jnp.ndarray,
    ) -> Tuple[float, jnp.ndarray]:
        """Temporal regularization: penalize changes between epochs.

        Args:
            x: Stacked maps (n_epochs × npix)

        Returns:
            f: Temporal smoothness penalty
            g: Gradient
        """
        # Reshape to (n_epochs, npix)
        maps = x.reshape(self.n_epochs, self.npix)

        # Differences between consecutive epochs
        diffs = maps[1:] - maps[:-1]

        # L2 penalty on differences
        f = jnp.sum(diffs**2)

        # Gradient
        g = jnp.zeros_like(maps)

        # First epoch: +diff[0]
        g = g.at[0].set(2 * (maps[0] - maps[1]))

        # Middle epochs: -diff[i-1] + diff[i]
        for i in range(1, self.n_epochs - 1):
            g = g.at[i].set(2 * (2*maps[i] - maps[i-1] - maps[i+1]))

        # Last epoch: -diff[-1]
        g = g.at[-1].set(2 * (maps[-1] - maps[-2]))

        return f, g.ravel()

    def reconstruct(
        self,
        x_start: Optional[jnp.ndarray] = None,
        bounds: Tuple[float, float] = (0.0, 50000.0),
        maxiter: int = 200,
    ) -> OptimizationResult:
        """Reconstruct surface from multi-epoch data.

        Args:
            x_start: Initial guess (default: uniform 5000 K)
            bounds: Temperature bounds (K)
            maxiter: Maximum iterations

        Returns:
            result: OptimizationResult
                - Static mode: x_solution is (npix,)
                - Dynamic mode: x_solution is (n_epochs × npix,)
        """
        # Default initial guess
        if x_start is None:
            x_start = jnp.ones(self.n_params) * 5000.0

        # Use scipy optimizer
        from scipy.optimize import minimize

        if self.verbose:
            print("="*80)
            print(f"Multi-Epoch Reconstruction ({self.mode} mode)")
            print("="*80)
            print(f"Epochs: {self.n_epochs}")
            print(f"Parameters: {self.n_params}")
            print(f"Temporal regularization: {self.temporal_reg}")
            print("="*80)

        # Optimize
        result = minimize(
            fun=self.objective_function,
            x0=np.array(x_start),
            method="L-BFGS-B",
            jac=True,
            bounds=[(bounds[0], bounds[1])] * self.n_params,
            options={'maxiter': maxiter, 'disp': self.verbose},
        )

        # Package results
        opt_result = OptimizationResult(
            x_solution=jnp.array(result.x),
            chi2_final=result.fun,  # Approximate
            reg_final=0.0,  # Not separated
            f_final=result.fun,
            iterations=result.nit,
            function_evaluations=result.nfev,
            gradient_evaluations=result.nfev,
            success=result.success,
            message=result.message,
            history={},
            time_elapsed=0.0,
        )

        if self.verbose:
            print("="*80)
            if result.success:
                print("SUCCESS!")
            else:
                print(f"FAILED: {result.message}")
            print(f"Final objective: {result.fun:.2f}")
            print(f"Iterations: {result.nit}")
            print("="*80)

        return opt_result


def reconstruct_multi_epoch(
    epochs: List[Epoch],
    star: Star,
    mode: str = "static",
    temporal_regularization: float = 0.0,
    regularizers: Optional[List[Dict]] = None,
    maxiter: int = 200,
    verbose: bool = True,
) -> OptimizationResult:
    """Convenience function for multi-epoch reconstruction.

    Args:
        epochs: List of observation epochs
        star: Template star geometry
        mode: "static" or "dynamic"
        temporal_regularization: Temporal smoothness weight (dynamic mode)
        regularizers: Spatial regularizers
        maxiter: Maximum iterations
        verbose: Print progress

    Returns:
        result: OptimizationResult

    Example:
        >>> # Simple static reconstruction
        >>> epochs = [Epoch(data, rotation_phase=p) for data, p in ...]
        >>> result = reconstruct_multi_epoch(epochs, star, mode="static")
        >>> map = result.x_solution
    """
    reconstructor = MultiEpochReconstructor(
        epochs=epochs,
        star_template=star,
        regularizers=regularizers,
        mode=mode,
        temporal_regularization=temporal_regularization,
        verbose=verbose,
    )

    result = reconstructor.reconstruct(maxiter=maxiter)

    return result


def compute_rotation_phase(
    mjd: float,
    mjd_ref: float,
    period: float,
) -> float:
    """Compute rotation phase from time.

    Args:
        mjd: Modified Julian Date
        mjd_ref: Reference MJD (phase = 0)
        period: Rotation period (days)

    Returns:
        phase: Rotation phase [0, 1]

    Example:
        >>> # Star with 2-day period
        >>> phase = compute_rotation_phase(
        ...     mjd=59000.5,
        ...     mjd_ref=59000.0,
        ...     period=2.0,
        ... )
        >>> print(f"Phase = {phase:.3f}")  # 0.25
    """
    phase = ((mjd - mjd_ref) / period) % 1.0
    return phase


def extract_epoch_maps(
    x_solution: jnp.ndarray,
    n_epochs: int,
    npix: int,
) -> jnp.ndarray:
    """Extract individual maps from dynamic reconstruction.

    Args:
        x_solution: Flattened solution (n_epochs × npix,)
        n_epochs: Number of epochs
        npix: Pixels per map

    Returns:
        maps: (n_epochs, npix) array

    Example:
        >>> result = reconstruct_multi_epoch(..., mode="dynamic")
        >>> maps = extract_epoch_maps(result.x_solution, n_epochs=4, npix=48)
        >>> map_epoch_1 = maps[0]  # First epoch
    """
    return x_solution.reshape(n_epochs, npix)
