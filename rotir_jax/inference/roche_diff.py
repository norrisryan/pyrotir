"""Differentiable Roche geometry via JAX custom VJP.

The Roche potential solver uses Newton-Raphson iteration to find r(theta, phi)
on the equipotential surface. Standard JAX autodiff cannot trace through this
iterative solver. We use the implicit function theorem to compute exact
gradients through the solution.

Implicit function theorem:
    Given F(r*, params) = Omega(r*) - Omega_surface = 0 at the solution r*,
    dr*/dp = -(dOmega/dr)^{-1} * dF/dp
    where dOmega/dr is already computed in the Newton-Raphson solver.

This enables gradient-based inference over Roche parameters (rpole, q, D, etc.)
for Bayesian posterior sampling of binary star geometry.

References:
    - Blondel & Berthet (2022): Efficient and Modular Implicit Differentiation
    - Bai et al. (2019): Deep Equilibrium Models
    - Aufdenberg et al. (2021): SPICA Roche model
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


def _roche_potential_primary(
    r: jnp.ndarray,
    D: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    q: jnp.ndarray,
    async_ratio: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Roche potential for primary star (JAX-traceable, batched).

    Vectorized version of compute_potential_primary that works with
    batched inputs for all tessellation points simultaneously.

    Args:
        r: Radii from primary center (dimensionless, r/a). Shape: scalar or (npix,)
        D: Instantaneous separation (dimensionless). Scalar.
        theta: Colatitude (radians). Shape: scalar or (npix,)
        phi: Longitude (radians). Shape: scalar or (npix,)
        q: Mass ratio M2/M1. Scalar.
        async_ratio: omega_rot/omega_orb. Scalar.

    Returns:
        Omega: Roche potential. Same shape as r.
        dOmega: Radial derivative dOmega/dr.
        ddOmega: Second derivative d^2 Omega/dr^2.
    """
    lam = jnp.sin(theta) * jnp.cos(phi)
    nu = jnp.cos(theta)

    r_sec_sq = D**2 + r**2 - 2 * r * lam * D
    r_sec = jnp.sqrt(jnp.maximum(r_sec_sq, 1e-30))

    Omega = (1.0 / r
             + q / r_sec
             - q * r * lam * D
             + 0.5 * async_ratio**2 * (1 + q) * r**2 * (1 - nu**2))

    dOmega = (-1.0 / r**2
              - q * (r - lam * D) / (r_sec_sq * r_sec)
              - q * lam * D
              + async_ratio**2 * (1 + q) * r * (1 - nu**2))

    ddOmega = (2.0 / r**3
               + 3 * q * (D * lam - r)**2 / (r_sec_sq**2 * r_sec)
               - q / (r_sec_sq * r_sec)
               + async_ratio**2 * (1 + q) * (1 - nu**2))

    return Omega, dOmega, ddOmega


def _newton_solve_batched(
    r_init: jnp.ndarray,
    pot_surface: jnp.ndarray,
    D: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    q: jnp.ndarray,
    async_ratio: jnp.ndarray,
    max_iter: int = 30,
) -> jnp.ndarray:
    """Batched Newton-Raphson solver for Roche radius.

    Uses jax.lax.fori_loop for JIT compatibility. Fixed iteration count
    ensures static graph structure.

    Args:
        r_init: Initial guesses. Shape: (npix,)
        pot_surface: Target surface potential. Scalar.
        D: Binary separation. Scalar.
        theta: Colatitudes. Shape: (npix,)
        phi: Longitudes. Shape: (npix,)
        q: Mass ratio. Scalar.
        async_ratio: Rotation ratio. Scalar.
        max_iter: Fixed iteration count.

    Returns:
        r: Converged radii. Shape: (npix,)
    """
    def body_fn(_, r):
        Omega, dOmega, _ = _roche_potential_primary(
            r, D, theta, phi, q, async_ratio
        )
        f = Omega - pot_surface
        # Newton step with damping to prevent divergence
        step = f / jnp.where(jnp.abs(dOmega) > 1e-30, dOmega, 1e-30)
        r_new = r - step
        # Clamp to positive values
        r_new = jnp.maximum(r_new, 1e-10)
        return r_new

    r_solution = jax.lax.fori_loop(0, max_iter, body_fn, r_init)
    return r_solution


@jax.custom_vjp
def solve_roche_radius_diff(
    rpole_dim: jnp.ndarray,
    D: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    q: jnp.ndarray,
    async_ratio: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable Roche radius solver.

    Solves for r(theta, phi) such that Omega(r) = Omega_surface, where
    Omega_surface is determined by the polar radius rpole.

    Gradients are computed via implicit differentiation (custom VJP),
    not by differentiating through the Newton-Raphson iterations.

    Args:
        rpole_dim: Dimensionless polar radius (rpole/a). Scalar.
        D: Dimensionless binary separation. Scalar.
        theta: Colatitudes. Shape: (npix,)
        phi: Longitudes. Shape: (npix,)
        q: Mass ratio M2/M1. Scalar.
        async_ratio: omega_rot/omega_orb. Scalar.

    Returns:
        radii: Radii at each (theta, phi). Shape: (npix,)
    """
    # Compute surface potential from polar radius
    pot_surface, _, _ = _roche_potential_primary(
        rpole_dim, D, jnp.array(0.0), jnp.array(0.0), q, async_ratio
    )

    # Solve for radii at all tessellation points
    r_init = jnp.broadcast_to(rpole_dim, theta.shape)
    radii = _newton_solve_batched(
        r_init, pot_surface, D, theta, phi, q, async_ratio
    )

    return radii


def _solve_roche_fwd(rpole_dim, D, theta, phi, q, async_ratio):
    """Forward pass: solve and save residuals for backward pass."""
    radii = solve_roche_radius_diff(rpole_dim, D, theta, phi, q, async_ratio)
    # Save everything needed for backward pass
    return radii, (radii, rpole_dim, D, theta, phi, q, async_ratio)


def _solve_roche_bwd(res, g):
    """Backward pass: implicit differentiation via IFT.

    Given the implicit function:
        F(r, params) = Omega(r, D, theta, phi, q, F) - Omega_surface(rpole, D, q, F) = 0

    By the implicit function theorem:
        dr/dp = -(dF/dr)^{-1} * dF/dp

    where dF/dr = dOmega/dr at the solution point.

    For efficiency, we compute the VJP:
        v^T (dr/dp) = -v^T * (dOmega/dr)^{-1} * dF/dp
                    = -lambda^T * dF/dp
        where lambda = v / dOmega_dr (element-wise for diagonal Jacobian).

    We then use JAX's autodiff to compute lambda^T * dF/dp for each parameter.
    """
    radii, rpole_dim, D, theta, phi, q, async_ratio = res
    v = g  # cotangent vector, shape (npix,)

    # Compute dOmega/dr at the solution points
    _, dOmega_dr, _ = _roche_potential_primary(
        radii, D, theta, phi, q, async_ratio
    )

    # lambda = -v / dOmega_dr (adjoint variable)
    lam = -v / jnp.where(jnp.abs(dOmega_dr) > 1e-30, dOmega_dr, 1e-30)

    # Now compute dF/dp for each parameter using JAX autodiff.
    # F_i = Omega(r_i, D, theta_i, phi_i, q, F) - Omega_surface(rpole, D, 0, 0, q, F)
    #
    # We need: sum_i lam_i * dF_i/dp for each parameter p.
    #
    # Strategy: define f(p) = sum_i lam_i * F_i(r_i, p) and take grad w.r.t. p.
    # r_i are held constant (they are the converged solutions).

    def F_sum(rpole_dim_, D_, q_, async_ratio_):
        """Sum of lambda * F over all pixels.

        r, theta, phi are held fixed (not differentiated).
        """
        # Surface potential from pole
        Omega_surf, _, _ = _roche_potential_primary(
            rpole_dim_, D_, jnp.array(0.0), jnp.array(0.0), q_, async_ratio_
        )
        # Potential at solution points
        Omega_sol, _, _ = _roche_potential_primary(
            radii, D_, theta, phi, q_, async_ratio_
        )
        # F = Omega_sol - Omega_surf
        F = Omega_sol - Omega_surf
        return jnp.sum(lam * F)

    # Gradient of F_sum w.r.t. (rpole_dim, D, q, async_ratio)
    grad_F = jax.grad(F_sum, argnums=(0, 1, 2, 3))(
        rpole_dim, D, q, async_ratio
    )
    grad_rpole_dim, grad_D, grad_q, grad_async_ratio = grad_F

    # theta, phi: the implicit function also depends on these
    # through Omega(r, D, theta, phi, ...).
    # dF/dtheta_i = dOmega/dtheta_i at (r_i, theta_i, phi_i)
    # dF/dphi_i = dOmega/dphi_i at (r_i, theta_i, phi_i)
    def F_angles(theta_, phi_):
        Omega_surf, _, _ = _roche_potential_primary(
            rpole_dim, D, jnp.array(0.0), jnp.array(0.0), q, async_ratio
        )
        Omega_sol, _, _ = _roche_potential_primary(
            radii, D, theta_, phi_, q, async_ratio
        )
        F = Omega_sol - Omega_surf
        return jnp.sum(lam * F)

    grad_theta, grad_phi = jax.grad(F_angles, argnums=(0, 1))(theta, phi)

    return (grad_rpole_dim, grad_D, grad_theta, grad_phi, grad_q, grad_async_ratio)


solve_roche_radius_diff.defvjp(_solve_roche_fwd, _solve_roche_bwd)


def compute_roche_shape_diff(
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    rpole: jnp.ndarray,
    a: jnp.ndarray,
    D: jnp.ndarray,
    q: jnp.ndarray,
    async_ratio: jnp.ndarray = jnp.array(1.0),
) -> jnp.ndarray:
    """Compute differentiable Roche-distorted stellar surface.

    High-level interface that takes physical parameters and returns
    physical radii (in same units as rpole and a).

    All parameters are differentiable via implicit differentiation.

    Args:
        theta: Colatitudes of tessellation points. Shape: (npix,)
        phi: Longitudes of tessellation points. Shape: (npix,)
        rpole: Polar radius (physical units, e.g. mas).
        a: Semi-major axis (same units as rpole).
        D: Dimensionless separation d/a.
        q: Mass ratio M2/M1.
        async_ratio: omega_rot/omega_orb (default 1.0).

    Returns:
        radii: Physical radii at each (theta, phi). Shape: (npix,)

    Example:
        >>> theta = jnp.array([0.0, 0.5, 1.0, 1.5])
        >>> phi = jnp.array([0.0, 0.0, 0.0, 0.0])
        >>> rpole, a, D, q = 10.0, 30.0, 1.0, 0.5
        >>> radii = compute_roche_shape_diff(
        ...     jnp.array(theta), jnp.array(phi),
        ...     jnp.array(rpole), jnp.array(a),
        ...     jnp.array(D), jnp.array(q),
        ... )
        >>> # Gradient w.r.t. rpole:
        >>> grad_fn = jax.grad(lambda rp: jnp.sum(compute_roche_shape_diff(
        ...     theta, phi, rp, a, D, q)))
        >>> grad_rpole = grad_fn(jnp.array(rpole))
    """
    rpole_dim = rpole / a
    radii_dim = solve_roche_radius_diff(rpole_dim, D, theta, phi, q, async_ratio)
    return radii_dim * a


def build_differentiable_geometry(
    tess_theta: jnp.ndarray,
    tess_phi: jnp.ndarray,
    rpole: jnp.ndarray,
    a: jnp.ndarray,
    D: jnp.ndarray,
    q: jnp.ndarray,
    inclination: jnp.ndarray,
    position_angle: jnp.ndarray,
    async_ratio: jnp.ndarray = jnp.array(1.0),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build differentiable Roche geometry projected onto sky plane.

    Combines Roche shape computation with rotation and projection,
    all differentiable via JAX.

    Args:
        tess_theta: Tessellation colatitudes. Shape: (npix,)
        tess_phi: Tessellation longitudes. Shape: (npix,)
        rpole: Polar radius (mas).
        a: Semi-major axis (mas).
        D: Dimensionless separation.
        q: Mass ratio.
        inclination: Inclination in radians.
        position_angle: Position angle in radians.
        async_ratio: Rotation ratio.

    Returns:
        x_sky: Sky-plane x coordinates. Shape: (npix,)
        y_sky: Sky-plane y coordinates. Shape: (npix,)
        z_obs: Observer-direction z coordinates. Shape: (npix,)
    """
    # Compute Roche radii (differentiable)
    radii = compute_roche_shape_diff(
        tess_theta, tess_phi, rpole, a, D, q, async_ratio
    )

    # Convert to Cartesian (on stellar surface)
    x_star = radii * jnp.sin(tess_theta) * jnp.cos(tess_phi)
    y_star = radii * jnp.sin(tess_theta) * jnp.sin(tess_phi)
    z_star = radii * jnp.cos(tess_theta)

    # Rotation matrix (3-1-3 Euler: PA, inc, 0)
    ci = jnp.cos(inclination)
    si = jnp.sin(inclination)
    cp = jnp.cos(position_angle)
    sp = jnp.sin(position_angle)

    # Simplified rotation: Z(PA) * X(inc)
    x_sky = (-sp * ci * y_star + cp * x_star - sp * si * z_star)
    y_sky = (cp * ci * y_star + sp * x_star + cp * si * z_star)
    z_obs = (-si * y_star + ci * z_star)

    return x_sky, y_sky, z_obs
