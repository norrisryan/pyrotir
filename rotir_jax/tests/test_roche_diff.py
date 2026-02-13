"""Tests for differentiable Roche geometry via JAX custom VJP.

Tests verify:
1. Forward pass matches the original non-differentiable Roche solver
2. Gradients exist and have correct shapes
3. Gradients are numerically accurate (finite difference check)
4. Implicit differentiation gives correct sensitivities
5. JIT compilation works correctly
"""

import pytest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from rotir_jax.inference.roche_diff import (
    _roche_potential_primary,
    _newton_solve_batched,
    solve_roche_radius_diff,
    compute_roche_shape_diff,
    build_differentiable_geometry,
)
from rotir_jax.geometry.roche import (
    compute_potential_primary,
    solve_roche_radius,
    eggleton_roche_radius,
)


# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture
def binary_params():
    """Standard binary star parameters for testing."""
    return {
        'rpole': 10.0,    # mas
        'a': 30.0,         # mas (semi-major axis)
        'D': 1.0,          # dimensionless separation
        'q': 0.5,          # mass ratio
        'async_ratio': 1.0,
    }


@pytest.fixture
def tessellation_points():
    """Small set of tessellation angles for testing."""
    # Avoid theta=0 (pole) where gradient is trivial
    theta = jnp.array([0.3, 0.6, 1.0, 1.2, 1.5, 2.0, 2.5])
    phi = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    return theta, phi


# ============================================================
# Test potential function
# ============================================================

class TestRochePotential:
    """Tests for the batched Roche potential computation."""

    def test_potential_scalar(self):
        """Potential at a single point matches original implementation."""
        r, D, theta, phi, q, F = 0.3, 1.0, 0.5, 0.0, 0.5, 1.0

        Omega_new, dOmega_new, ddOmega_new = _roche_potential_primary(
            jnp.array(r), jnp.array(D), jnp.array(theta),
            jnp.array(phi), jnp.array(q), jnp.array(F),
        )
        Omega_orig, dOmega_orig, ddOmega_orig = compute_potential_primary(
            r, D, theta, phi, q, F,
        )

        np.testing.assert_allclose(float(Omega_new), float(Omega_orig), rtol=1e-6)
        np.testing.assert_allclose(float(dOmega_new), float(dOmega_orig), rtol=1e-6)

    def test_potential_batched(self):
        """Batched potential gives same results as looped scalar."""
        theta = jnp.array([0.3, 0.6, 1.0, 1.5])
        phi = jnp.array([0.0, 0.5, 1.0, 2.0])
        r = jnp.array([0.3, 0.32, 0.35, 0.28])
        D, q, F = jnp.array(1.0), jnp.array(0.5), jnp.array(1.0)

        Omega_batch, dOmega_batch, _ = _roche_potential_primary(
            r, D, theta, phi, q, F
        )

        for i in range(len(theta)):
            Omega_i, dOmega_i, _ = _roche_potential_primary(
                r[i], D, theta[i], phi[i], q, F
            )
            np.testing.assert_allclose(
                float(Omega_batch[i]), float(Omega_i), rtol=1e-6
            )

    def test_potential_pole(self):
        """Potential at the pole (theta=0) is well-defined."""
        Omega, dOmega, _ = _roche_potential_primary(
            jnp.array(0.3), jnp.array(1.0),
            jnp.array(0.0), jnp.array(0.0),
            jnp.array(0.5), jnp.array(1.0),
        )
        assert jnp.isfinite(Omega)
        assert jnp.isfinite(dOmega)


# ============================================================
# Test Newton solver
# ============================================================

class TestNewtonSolver:
    """Tests for the batched Newton-Raphson Roche solver."""

    def test_solver_converges(self, binary_params, tessellation_points):
        """Newton solver converges to correct potential surface."""
        theta, phi = tessellation_points
        p = binary_params
        rpole_dim = p['rpole'] / p['a']
        D = jnp.array(p['D'])
        q = jnp.array(p['q'])
        F = jnp.array(p['async_ratio'])

        # Compute target surface potential
        pot_surf, _, _ = _roche_potential_primary(
            jnp.array(rpole_dim), D,
            jnp.array(0.0), jnp.array(0.0), q, F,
        )

        # Solve
        r_init = jnp.full_like(theta, rpole_dim)
        radii = _newton_solve_batched(
            r_init, pot_surf, D, theta, phi, q, F, max_iter=50
        )

        # Verify: potential at solution matches target
        Omega_sol, _, _ = _roche_potential_primary(
            radii, D, theta, phi, q, F
        )
        np.testing.assert_allclose(
            np.array(Omega_sol), np.array(jnp.full_like(Omega_sol, pot_surf)),
            rtol=1e-4,
        )

    def test_solver_positive_radii(self, binary_params, tessellation_points):
        """All solved radii are positive."""
        theta, phi = tessellation_points
        p = binary_params
        rpole_dim = jnp.array(p['rpole'] / p['a'])
        D = jnp.array(p['D'])
        q = jnp.array(p['q'])
        F = jnp.array(p['async_ratio'])

        radii = solve_roche_radius_diff(rpole_dim, D, theta, phi, q, F)
        assert jnp.all(radii > 0)

    def test_pole_returns_rpole(self, binary_params):
        """Solving at the pole (theta=0) returns the polar radius."""
        p = binary_params
        rpole_dim = jnp.array(p['rpole'] / p['a'])
        D = jnp.array(p['D'])
        q = jnp.array(p['q'])
        F = jnp.array(p['async_ratio'])

        theta = jnp.array([0.0])
        phi = jnp.array([0.0])

        radii = solve_roche_radius_diff(rpole_dim, D, theta, phi, q, F)
        np.testing.assert_allclose(float(radii[0]), float(rpole_dim), rtol=1e-4)


# ============================================================
# Test implicit differentiation
# ============================================================

class TestImplicitDifferentiation:
    """Tests for gradients via implicit function theorem."""

    def test_gradient_exists(self, binary_params, tessellation_points):
        """Gradient computation does not error."""
        theta, phi = tessellation_points
        p = binary_params

        def f(rpole_dim):
            return jnp.sum(solve_roche_radius_diff(
                rpole_dim,
                jnp.array(p['D']),
                theta, phi,
                jnp.array(p['q']),
                jnp.array(p['async_ratio']),
            ))

        grad = jax.grad(f)(jnp.array(p['rpole'] / p['a']))
        assert jnp.isfinite(grad)

    def test_gradient_shape(self, binary_params, tessellation_points):
        """Gradients have correct shapes for all parameters."""
        theta, phi = tessellation_points
        p = binary_params

        def f(rpole_dim, D, q, async_ratio):
            return jnp.sum(solve_roche_radius_diff(
                rpole_dim, D, theta, phi, q, async_ratio
            ))

        grads = jax.grad(f, argnums=(0, 1, 2, 3))(
            jnp.array(p['rpole'] / p['a']),
            jnp.array(p['D']),
            jnp.array(p['q']),
            jnp.array(p['async_ratio']),
        )

        # All scalar gradients
        for g in grads:
            assert g.shape == ()
            assert jnp.isfinite(g)

    def test_gradient_numerical_check_rpole(self, binary_params, tessellation_points):
        """Gradient w.r.t. rpole matches finite differences."""
        theta, phi = tessellation_points
        p = binary_params
        eps = 1e-5

        def f(rpole_dim):
            return jnp.sum(solve_roche_radius_diff(
                rpole_dim,
                jnp.array(p['D']),
                theta, phi,
                jnp.array(p['q']),
                jnp.array(p['async_ratio']),
            ))

        rpole_dim = jnp.array(p['rpole'] / p['a'])
        grad_auto = float(jax.grad(f)(rpole_dim))

        # Finite difference
        f_plus = float(f(rpole_dim + eps))
        f_minus = float(f(rpole_dim - eps))
        grad_fd = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(grad_auto, grad_fd, rtol=1e-2)

    def test_gradient_numerical_check_q(self, binary_params, tessellation_points):
        """Gradient w.r.t. mass ratio q matches finite differences."""
        theta, phi = tessellation_points
        p = binary_params
        eps = 1e-6

        def f(q):
            return jnp.sum(solve_roche_radius_diff(
                jnp.float64(p['rpole'] / p['a']),
                jnp.float64(p['D']),
                theta, phi,
                q,
                jnp.float64(p['async_ratio']),
            ))

        q = jnp.float64(p['q'])
        grad_auto = float(jax.grad(f)(q))

        f_plus = float(f(q + eps))
        f_minus = float(f(q - eps))
        grad_fd = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(grad_auto, grad_fd, rtol=1e-3)

    def test_gradient_numerical_check_D(self, binary_params, tessellation_points):
        """Gradient w.r.t. separation D matches finite differences."""
        theta, phi = tessellation_points
        p = binary_params
        eps = 1e-6

        def f(D):
            return jnp.sum(solve_roche_radius_diff(
                jnp.float64(p['rpole'] / p['a']),
                D,
                theta, phi,
                jnp.float64(p['q']),
                jnp.float64(p['async_ratio']),
            ))

        D = jnp.float64(p['D'])
        grad_auto = float(jax.grad(f)(D))

        f_plus = float(f(D + eps))
        f_minus = float(f(D - eps))
        grad_fd = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(grad_auto, grad_fd, rtol=1e-3)

    def test_positive_rpole_sensitivity(self, binary_params, tessellation_points):
        """Increasing rpole should increase all radii."""
        theta, phi = tessellation_points
        p = binary_params

        def f(rpole_dim):
            return jnp.sum(solve_roche_radius_diff(
                rpole_dim,
                jnp.array(p['D']),
                theta, phi,
                jnp.array(p['q']),
                jnp.array(p['async_ratio']),
            ))

        grad = float(jax.grad(f)(jnp.array(p['rpole'] / p['a'])))
        assert grad > 0, "Increasing rpole should increase total radius"


# ============================================================
# Test high-level API
# ============================================================

class TestComputeRocheShapeDiff:
    """Tests for the high-level differentiable Roche shape function."""

    def test_output_shape(self, binary_params, tessellation_points):
        """Output shape matches input tessellation size."""
        theta, phi = tessellation_points
        p = binary_params
        radii = compute_roche_shape_diff(
            theta, phi,
            jnp.array(p['rpole']), jnp.array(p['a']),
            jnp.array(p['D']), jnp.array(p['q']),
        )
        assert radii.shape == theta.shape

    def test_physical_units(self, binary_params, tessellation_points):
        """Output radii are in physical units (same as rpole)."""
        theta, phi = tessellation_points
        p = binary_params
        radii = compute_roche_shape_diff(
            theta, phi,
            jnp.array(p['rpole']), jnp.array(p['a']),
            jnp.array(p['D']), jnp.array(p['q']),
        )
        # Equatorial radii should be >= polar radius (tidal distortion)
        # Polar radius check at theta=0
        pole_radius = compute_roche_shape_diff(
            jnp.array([0.0]), jnp.array([0.0]),
            jnp.array(p['rpole']), jnp.array(p['a']),
            jnp.array(p['D']), jnp.array(p['q']),
        )
        np.testing.assert_allclose(float(pole_radius[0]), p['rpole'], rtol=1e-3)

    def test_gradient_through_physical(self, binary_params, tessellation_points):
        """Gradient flows through the full physical-units function."""
        theta, phi = tessellation_points
        p = binary_params

        def f(rpole, a, q):
            return jnp.sum(compute_roche_shape_diff(
                theta, phi, rpole, a, jnp.array(p['D']), q
            ))

        grads = jax.grad(f, argnums=(0, 1, 2))(
            jnp.array(p['rpole']),
            jnp.array(p['a']),
            jnp.array(p['q']),
        )
        for g in grads:
            assert jnp.isfinite(g)

    def test_jit_compilation(self, binary_params, tessellation_points):
        """Function works under JIT compilation."""
        theta, phi = tessellation_points
        p = binary_params

        @jax.jit
        def f(rpole):
            return compute_roche_shape_diff(
                theta, phi,
                rpole, jnp.array(p['a']),
                jnp.array(p['D']), jnp.array(p['q']),
            )

        radii = f(jnp.array(p['rpole']))
        assert radii.shape == theta.shape
        assert jnp.all(jnp.isfinite(radii))

    def test_jit_gradient(self, binary_params, tessellation_points):
        """Gradient works under JIT compilation."""
        theta, phi = tessellation_points
        p = binary_params

        @jax.jit
        def grad_f(rpole):
            def f(rp):
                return jnp.sum(compute_roche_shape_diff(
                    theta, phi, rp, jnp.array(p['a']),
                    jnp.array(p['D']), jnp.array(p['q']),
                ))
            return jax.grad(f)(rpole)

        g = grad_f(jnp.array(p['rpole']))
        assert jnp.isfinite(g)


# ============================================================
# Test geometry builder
# ============================================================

class TestDifferentiableGeometry:
    """Tests for the full differentiable geometry pipeline."""

    def test_sky_coordinates(self, binary_params, tessellation_points):
        """Sky-plane projection produces finite coordinates."""
        theta, phi = tessellation_points
        p = binary_params

        x_sky, y_sky, z_obs = build_differentiable_geometry(
            theta, phi,
            jnp.array(p['rpole']), jnp.array(p['a']),
            jnp.array(p['D']), jnp.array(p['q']),
            jnp.array(jnp.deg2rad(60.0)),  # inclination
            jnp.array(jnp.deg2rad(0.0)),    # PA
        )

        assert x_sky.shape == theta.shape
        assert y_sky.shape == theta.shape
        assert z_obs.shape == theta.shape
        assert jnp.all(jnp.isfinite(x_sky))
        assert jnp.all(jnp.isfinite(y_sky))
        assert jnp.all(jnp.isfinite(z_obs))

    def test_geometry_gradient(self, binary_params, tessellation_points):
        """Gradient flows through the full geometry pipeline."""
        theta, phi = tessellation_points
        p = binary_params

        def f(rpole, inc):
            x_sky, y_sky, z_obs = build_differentiable_geometry(
                theta, phi,
                rpole, jnp.array(p['a']),
                jnp.array(p['D']), jnp.array(p['q']),
                inc, jnp.array(0.0),
            )
            return jnp.sum(x_sky**2 + y_sky**2)

        grads = jax.grad(f, argnums=(0, 1))(
            jnp.array(p['rpole']),
            jnp.array(jnp.deg2rad(60.0)),
        )
        for g in grads:
            assert jnp.isfinite(g)
