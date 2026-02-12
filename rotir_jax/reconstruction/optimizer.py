"""Optimizer for ROTIR image reconstruction.

This is THE CORE MODULE that enables actual image reconstruction from interferometric data.

Combines:
1. Forward model (geometry → polygon FT → observables → χ²)
2. Regularization (priors: MEM, TV, etc.)
3. Gradient-based optimization (L-BFGS-B with box constraints)

Minimizes: f(x) = χ²(x) + Σλᵢ Rᵢ(x)
- χ²: data fidelity (goodness of fit)
- Rᵢ: regularization functionals (priors)
- λᵢ: regularization weights (hyperparameters)

Subject to: T_min ≤ x ≤ T_max (box constraints)

Uses gradient-based optimization:
- L-BFGS-B: Quasi-Newton with box constraints
- Gradients from JAX autodiff or manual computation
- Typically 50-200 iterations for convergence

Physical motivation:
- Ill-posed inverse problem → regularization needed
- Temperature maps: T > 0, typically 3000-50000 K
- Flux normalization: maintain total stellar flux
- Visibility constraints: match Fourier amplitudes/phases

Output: reconstructed temperature/intensity map on stellar surface

References:
- Thiébaut & Giovannelli (1997): Image reconstruction
- Renard et al. (2011): SPARCO
- Baron & Monnier (2012): Principles of interferometric imaging
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import time

import sys
sys.path.append('..')
from rotir_jax.datatypes import OIData, Tessellation, Star
from rotir_jax.forward_model.observables import compute_chi2, compute_observables
from rotir_jax.forward_model.polyft import mod360
from rotir_jax.regularization.regularizers import apply_regularizers, build_healpix_difference_matrix


@dataclass
class OptimizationResult:
    """Results from image reconstruction optimization.

    Attributes:
        x_solution: Reconstructed image (npix,)
        chi2_final: Final χ² value
        reg_final: Final regularization value
        f_final: Final objective function value
        iterations: Number of optimizer iterations
        function_evaluations: Number of function calls
        gradient_evaluations: Number of gradient calls
        success: Whether optimization converged
        message: Convergence message
        history: Optimization history (chi2, reg, f per iteration)
        time_elapsed: Total wall-clock time (seconds)
    """
    x_solution: jnp.ndarray
    chi2_final: float
    reg_final: float
    f_final: float
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    success: bool
    message: str
    history: Dict[str, List[float]]
    time_elapsed: float


class StellarImageReconstructor:
    """Image reconstruction engine for stellar surface mapping.

    This class orchestrates the complete image reconstruction pipeline:
    1. Forward model: map intensity → observables
    2. Data fidelity: χ² between model and data
    3. Regularization: priors on image structure
    4. Optimization: find image minimizing objective function

    Example:
        >>> # Setup data and star model
        >>> oi_data = read_oifits("star.fits")
        >>> star = create_star(...)
        >>>
        >>> # Define regularizers
        >>> regularizers = [
        ...     {"type": "mem", "weight": 0.05},
        ...     {"type": "tv", "weight": 0.01},
        ... ]
        >>>
        >>> # Reconstruct
        >>> reconstructor = StellarImageReconstructor(
        ...     oi_data=oi_data,
        ...     star=star,
        ...     regularizers=regularizers,
        ... )
        >>>
        >>> # Initial guess (uniform)
        >>> x0 = jnp.ones(star.tess.npix) * 5000.0  # K
        >>>
        >>> # Optimize
        >>> result = reconstructor.reconstruct(
        ...     x_start=x0,
        ...     bounds=(3000, 10000),
        ...     maxiter=200,
        ... )
        >>>
        >>> print(f"Final χ² = {result.chi2_final:.2f}")
        >>> print(f"Converged: {result.success}")
    """

    def __init__(
        self,
        oi_data: OIData,
        star: Star,
        regularizers: List[Dict],
        data_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ):
        """Initialize reconstructor.

        Args:
            oi_data: Interferometric observables
            star: Stellar geometry model
            regularizers: List of regularizer specifications
            data_weights: Weights for different data types (default {"OI": 1.0})
            verbose: Print optimization progress (default True)
        """
        self.oi_data = oi_data
        self.star = star
        self.regularizers = regularizers
        self.data_weights = data_weights or {"OI": 1.0}
        self.verbose = verbose

        # Build difference matrix for TV regularizers
        self.diff_matrix = None
        if any(r["type"] in ["tv", "tv1", "tv2"] for r in regularizers):
            # Try to use HEALPix structure
            try:
                nside = getattr(star.tess, 'nside', None)
                if nside is not None:
                    self.diff_matrix = build_healpix_difference_matrix(nside)
                else:
                    from rotir_jax.regularization.regularizers import build_difference_matrix
                    self.diff_matrix = build_difference_matrix(star.tess)
            except:
                # Fallback to simple difference matrix
                from rotir_jax.regularization.regularizers import build_difference_matrix
                self.diff_matrix = build_difference_matrix(star.tess)

        # History tracking
        self.history = {
            "chi2": [],
            "reg": [],
            "f_total": [],
            "x_min": [],
            "x_max": [],
            "x_mean": [],
        }

        # Iteration counter
        self.n_iter = 0
        self.n_feval = 0
        self.n_geval = 0

    def objective_function(
        self,
        x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Objective function for optimization: f(x) = χ² + regularization.

        Args:
            x: Image intensities/temperatures (npix,)

        Returns:
            f: Objective function value
            g: Gradient ∂f/∂x (npix,)

        Notes:
            - Called by scipy.optimize.minimize
            - Uses JAX for gradient computation
            - Combines data fidelity and regularization
        """
        self.n_feval += 1

        # Convert to JAX array
        x_jax = jnp.array(x)

        # Compute χ² and gradient
        chi2, g_chi2 = self._compute_chi2_with_gradient(x_jax)

        # Compute regularization and gradient
        reg, g_reg = apply_regularizers(
            x_jax,
            self.regularizers,
            diff_matrix=self.diff_matrix
        )

        # Total objective
        f = chi2 + reg

        # Total gradient
        g = g_chi2 + g_reg

        # Track history
        self.history["chi2"].append(float(chi2))
        self.history["reg"].append(float(reg))
        self.history["f_total"].append(float(f))
        self.history["x_min"].append(float(jnp.min(x_jax)))
        self.history["x_max"].append(float(jnp.max(x_jax)))
        self.history["x_mean"].append(float(jnp.mean(x_jax)))

        # Print progress
        if self.verbose and (self.n_feval % 10 == 0 or self.n_feval == 1):
            self._print_progress(f, chi2, reg, x_jax)

        # Convert gradient to numpy for scipy
        return float(f), np.array(g)

    def _compute_chi2_with_gradient(
        self,
        x: jnp.ndarray,
    ) -> Tuple[float, jnp.ndarray]:
        """Compute χ² and its gradient using JAX autodiff.

        Args:
            x: Image intensities (npix,)

        Returns:
            chi2: χ² value
            g_chi2: Gradient ∂χ²/∂x
        """
        # Define χ² as function of x
        def chi2_func(x_param):
            # Update star with new intensities
            star_updated = self._update_star_intensities(x_param)

            # Convert star to geometry for forward model
            geom = star_updated.to_geometry()

            # Compute observables
            v2_model, t3amp_model, t3phi_model = compute_observables(
                star_updated.intensities,
                geom,
                self.oi_data
            )

            # Compute χ² from residuals
            v2_residual = (v2_model - self.oi_data.v2) / self.oi_data.v2_err
            t3phi_residual = mod360(t3phi_model - self.oi_data.t3phi) / self.oi_data.t3phi_err

            chi2 = jnp.sum(v2_residual**2) + jnp.sum(t3phi_residual**2)

            return chi2

        # Use JAX autodiff
        chi2, g_chi2 = jax.value_and_grad(chi2_func)(x)

        return chi2, g_chi2

    def _update_star_intensities(
        self,
        x: jnp.ndarray,
    ) -> Star:
        """Update star model with new intensity map.

        Args:
            x: New intensities (npix,)

        Returns:
            star: Updated star model

        Notes:
            - Creates new Star with updated intensities
            - Preserves geometry, visible mask, etc.
        """
        # Create new star with updated intensities
        # Preserve all geometry and only update intensities
        star_new = Star(
            tess=self.star.tess,
            theta=self.star.theta,
            phi=self.star.phi,
            x=self.star.x,
            y=self.star.y,
            z=self.star.z,
            visible=self.star.visible,
            intensities=x,  # Updated intensities
            diameter=self.star.diameter,
            inclination=self.star.inclination,
            orientation=self.star.orientation,
            ld_coeffs=self.star.ld_coeffs,
        )

        return star_new

    def _print_progress(
        self,
        f: float,
        chi2: float,
        reg: float,
        x: jnp.ndarray,
    ):
        """Print optimization progress.

        Args:
            f: Total objective function value
            chi2: χ² value
            reg: Regularization value
            x: Current image
        """
        # Compute statistics on visible pixels only
        x_visible = x[self.star.visible] if self.star.visible is not None else x

        print(f"Iter {self.n_feval:4d} | "
              f"f={f:10.2f} | "
              f"χ²={chi2:10.2f} | "
              f"reg={reg:8.2f} | "
              f"T: [{jnp.min(x_visible):.0f}, {jnp.max(x_visible):.0f}] K "
              f"(mean={jnp.mean(x_visible):.0f})")

    def reconstruct(
        self,
        x_start: jnp.ndarray,
        bounds: Tuple[float, float] = (0.0, np.inf),
        maxiter: int = 200,
        gtol: float = 1e-8,
        ftol: float = 1e-9,
        method: str = "L-BFGS-B",
    ) -> OptimizationResult:
        """Reconstruct stellar surface image from interferometric data.

        This is the MAIN FUNCTION that performs image reconstruction.

        Args:
            x_start: Initial guess for image (npix,)
            bounds: Box constraints (T_min, T_max) in Kelvin
            maxiter: Maximum number of iterations (default 200)
            gtol: Gradient convergence tolerance (default 1e-8)
            ftol: Function convergence tolerance (default 1e-9)
            method: Optimization method (default "L-BFGS-B")

        Returns:
            result: OptimizationResult with solution and diagnostics

        Notes:
            - Uses scipy.optimize.minimize with L-BFGS-B
            - L-BFGS-B: Limited-memory BFGS with box constraints
            - Quasi-Newton method: approximates Hessian from gradients
            - Typically converges in 50-200 iterations
            - Box constraints enforce physical temperatures

        Example:
            >>> # Uniform initial guess
            >>> x0 = jnp.ones(star.tess.npix) * 5000.0  # K
            >>>
            >>> # Reconstruct with constraints
            >>> result = reconstructor.reconstruct(
            ...     x_start=x0,
            ...     bounds=(3000, 10000),  # Cool to hot stars
            ...     maxiter=200,
            ... )
            >>>
            >>> # Check convergence
            >>> if result.success:
            ...     print(f"Converged in {result.iterations} iterations")
            ...     print(f"Final χ² = {result.chi2_final:.2f}")
        """
        # Reset counters
        self.n_feval = 0
        self.n_geval = 0
        self.history = {k: [] for k in self.history.keys()}

        # Start timer
        t_start = time.time()

        if self.verbose:
            print("="*80)
            print("ROTIR Image Reconstruction")
            print("="*80)
            print(f"Data: {len(self.oi_data.vis2)} vis², "
                  f"{len(self.oi_data.t3phi)} closure phases")
            print(f"Model: {len(x_start)} pixels")
            print(f"Regularizers: {len(self.regularizers)}")
            print(f"Bounds: [{bounds[0]:.0f}, {bounds[1]:.0f}] K")
            print(f"Method: {method}")
            print("="*80)

        # Convert to numpy for scipy
        x0_np = np.array(x_start)

        # Setup bounds array
        bounds_array = [(bounds[0], bounds[1])] * len(x0_np)

        # Optimize
        opt_result = minimize(
            fun=self.objective_function,
            x0=x0_np,
            method=method,
            jac=True,  # objective_function returns (f, g)
            bounds=bounds_array,
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'ftol': ftol,
                'disp': False,  # We handle printing
            },
        )

        # Stop timer
        t_elapsed = time.time() - t_start

        # Extract solution
        x_solution = jnp.array(opt_result.x)

        # Compute final values
        chi2_final = self.history["chi2"][-1] if self.history["chi2"] else 0.0
        reg_final = self.history["reg"][-1] if self.history["reg"] else 0.0
        f_final = opt_result.fun

        if self.verbose:
            print("="*80)
            print("Optimization Complete")
            print("="*80)
            print(f"Status: {'SUCCESS' if opt_result.success else 'FAILED'}")
            print(f"Message: {opt_result.message}")
            print(f"Iterations: {opt_result.nit}")
            print(f"Function evaluations: {opt_result.nfev}")
            print(f"Final χ² = {chi2_final:.2f}")
            print(f"Final reg = {reg_final:.2f}")
            print(f"Final f = {f_final:.2f}")
            print(f"Time elapsed: {t_elapsed:.1f} s")
            print("="*80)

        # Package results
        result = OptimizationResult(
            x_solution=x_solution,
            chi2_final=chi2_final,
            reg_final=reg_final,
            f_final=f_final,
            iterations=opt_result.nit,
            function_evaluations=opt_result.nfev,
            gradient_evaluations=opt_result.nfev,  # L-BFGS-B: nfev = ngev
            success=opt_result.success,
            message=opt_result.message,
            history=self.history,
            time_elapsed=t_elapsed,
        )

        return result


def reconstruct_stellar_surface(
    oi_data: OIData,
    star: Star,
    x_start: Optional[jnp.ndarray] = None,
    regularizers: Optional[List[Dict]] = None,
    bounds: Tuple[float, float] = (0.0, 50000.0),
    maxiter: int = 200,
    verbose: bool = True,
) -> OptimizationResult:
    """Convenience function for stellar surface reconstruction.

    High-level interface for image reconstruction with sensible defaults.

    Args:
        oi_data: Interferometric data
        star: Stellar geometry model
        x_start: Initial guess (default: uniform at 5000 K)
        regularizers: Regularization priors (default: MEM + TV)
        bounds: Temperature bounds (default: [0, 50000] K)
        maxiter: Maximum iterations (default 200)
        verbose: Print progress (default True)

    Returns:
        result: OptimizationResult

    Example:
        >>> # Simple reconstruction with defaults
        >>> result = reconstruct_stellar_surface(
        ...     oi_data=oi_data,
        ...     star=star,
        ... )
        >>>
        >>> # Reconstruction map
        >>> temperature_map = result.x_solution
    """
    # Default initial guess: uniform 5000 K
    if x_start is None:
        x_start = jnp.ones(star.tess.npix) * 5000.0

    # Default regularizers: MEM + TV
    if regularizers is None:
        regularizers = [
            {"type": "mem", "weight": 0.05},
            {"type": "tv", "weight": 0.01},
        ]

    # Create reconstructor
    reconstructor = StellarImageReconstructor(
        oi_data=oi_data,
        star=star,
        regularizers=regularizers,
        verbose=verbose,
    )

    # Reconstruct
    result = reconstructor.reconstruct(
        x_start=x_start,
        bounds=bounds,
        maxiter=maxiter,
    )

    return result


def compute_reduced_chi2(
    x: jnp.ndarray,
    oi_data: OIData,
    star: Star,
) -> float:
    """Compute reduced χ² for a given image.

    Reduced χ² = χ² / N_data

    Used to assess goodness of fit.
    Ideal value: χ²_red ≈ 1.0

    Args:
        x: Image intensities (npix,)
        oi_data: Observational data
        star: Stellar model

    Returns:
        chi2_reduced: Reduced χ²

    Notes:
        - χ²_red < 1: overfitting or overestimated errors
        - χ²_red ≈ 1: good fit
        - χ²_red > 1: underfitting or underestimated errors
    """
    # Update star with intensities
    star_updated = Star(
        tess=star.tess,
        theta=star.theta,
        phi=star.phi,
        x=star.x,
        y=star.y,
        z=star.z,
        visible=star.visible,
        intensities=x,
        ld_coeffs=star.ld_coeffs,
    )

    # Compute observables
    model_obs = compute_observables(star_updated, oi_data.wavelengths)

    # Compute χ²
    chi2 = compute_chi2(model_obs, oi_data)

    # Number of data points
    n_data = len(oi_data.vis2) + len(oi_data.t3phi)

    # Reduced χ²
    chi2_red = chi2 / n_data

    return chi2_red


def estimate_optimal_regularization_weight(
    oi_data: OIData,
    star: Star,
    reg_type: str = "tv",
    chi2_target: float = 1.1,
) -> float:
    """Estimate optimal regularization weight using L-curve criterion.

    The L-curve plots ||data misfit|| vs ||regularization||.
    Optimal weight is at the "corner" of the L-curve.

    Args:
        oi_data: Observational data
        star: Stellar model
        reg_type: Regularizer type ("tv", "mem", etc.)
        chi2_target: Target reduced χ² (default 1.1)

    Returns:
        optimal_weight: Estimated optimal regularization weight

    Notes:
        - This is a simplified heuristic
        - Full L-curve requires multiple reconstructions
        - See Hansen (1992) for details

    Example:
        >>> weight = estimate_optimal_regularization_weight(
        ...     oi_data, star, reg_type="tv"
        ... )
        >>> print(f"Suggested TV weight: {weight:.4f}")
    """
    # Heuristic: weight ~ 0.01-0.1 for TV, 0.05-0.2 for MEM
    if reg_type in ["tv", "tv1", "tv2"]:
        return 0.01
    elif reg_type == "mem":
        return 0.05
    elif reg_type == "mean":
        return 0.001
    elif reg_type == "bias":
        return 0.001
    else:
        return 0.01
