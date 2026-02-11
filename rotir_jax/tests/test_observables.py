"""Tests for observables module - complete forward model.

Tests verify:
1. Image → observables forward model
2. Chi-squared calculation
3. Gradient computation (JAX autodiff)
4. Forward model creation
5. Residuals computation
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')

from rotir_jax.tessellation.healpix import tessellation_healpix
from rotir_jax.geometry.base import create_star
from rotir_jax.forward_model.observables import (
    compute_observables,
    compute_chi2,
    compute_chi2_gradient,
    create_forward_model,
    compute_residuals,
    compute_reduced_chi2,
)
from rotir_jax.datatypes import OIData


def create_mock_oi_data(nuv=10, nt3=5):
    """Create mock OIFITS data for testing."""
    # Create mock UV coordinates
    u = np.random.randn(nuv) * 50  # cycles/mas
    v = np.random.randn(nuv) * 50
    uv = np.vstack([u, v])

    # Create mock V² data
    v2 = np.random.rand(nuv) * 0.5 + 0.3  # Between 0.3 and 0.8
    v2_err = np.ones(nuv) * 0.05
    indx_v2 = np.arange(nuv)

    # Create mock T3 data (triangles)
    t3amp = np.random.rand(nt3) * 0.2 + 0.1  # Between 0.1 and 0.3
    t3amp_err = np.ones(nt3) * 0.02

    t3phi = np.random.randn(nt3) * 30  # Degrees
    t3phi_err = np.ones(nt3) * 5.0

    # Triangle indices (circular: 0-1-2, 1-2-3, etc.)
    indx_t3_1 = np.array([0, 1, 2, 3, 4])
    indx_t3_2 = np.array([1, 2, 3, 4, 5])
    indx_t3_3 = np.array([2, 3, 4, 5, 6])

    return OIData(
        uv=uv,
        v2=v2,
        v2_err=v2_err,
        t3amp=t3amp,
        t3amp_err=t3amp_err,
        t3phi=t3phi,
        t3phi_err=t3phi_err,
        indx_v2=indx_v2,
        indx_t3_1=indx_t3_1,
        indx_t3_2=indx_t3_2,
        indx_t3_3=indx_t3_3,
        nv2=len(v2),
        nt3amp=len(t3amp),
        nt3phi=len(t3phi),
    )


def test_forward_model_uniform_disk():
    """Test forward model with uniform disk."""
    print("Testing forward model with uniform disk...")

    # Create geometry
    tess = tessellation_healpix(n=3)  # Small for speed
    geom = create_star(tess, radius=1.0, inc=45.0, PA=30.0, obliq=0.0)

    # Create uniform brightness map
    image = jnp.ones(geom.npix)

    # Create mock data
    oi_data = create_mock_oi_data(nuv=20, nt3=10)

    # Compute observables
    v2_model, t3amp_model, t3phi_model = compute_observables(
        image, geom, oi_data
    )

    print(f"  V² shape: {v2_model.shape}, range: [{v2_model.min():.4f}, {v2_model.max():.4f}]")
    print(f"  T3amp shape: {t3amp_model.shape}, range: [{t3amp_model.min():.4f}, {t3amp_model.max():.4f}]")
    print(f"  T3phi shape: {t3phi_model.shape}, range: [{t3phi_model.min():.2f}°, {t3phi_model.max():.2f}°]")

    # Check shapes
    assert v2_model.shape == (oi_data.nv2,), f"Wrong V² shape: {v2_model.shape}"
    assert t3amp_model.shape == (oi_data.nt3amp,), f"Wrong T3amp shape: {t3amp_model.shape}"
    assert t3phi_model.shape == (oi_data.nt3phi,), f"Wrong T3phi shape: {t3phi_model.shape}"

    # Check ranges
    assert jnp.all(v2_model >= 0), "V² should be non-negative"
    assert jnp.all(v2_model <= 1), "V² should be <= 1"
    assert jnp.all(t3amp_model >= 0), "T3amp should be non-negative"

    print("  ✓ Forward model uniform disk test passed")


def test_chi2_calculation():
    """Test chi-squared calculation."""
    print("\nTesting chi-squared calculation...")

    # Create geometry
    tess = tessellation_healpix(n=3)
    geom = create_star(tess, radius=1.0, inc=60.0, PA=0.0, obliq=0.0)

    # Create brightness map
    image = jnp.ones(geom.npix)

    # Create mock data
    oi_data = create_mock_oi_data(nuv=15, nt3=8)

    # Compute chi-squared
    chi2_total, chi2_v2, chi2_t3amp, chi2_t3phi = compute_chi2(
        image, geom, oi_data, return_components=True, verbose=True
    )

    print(f"  Total χ²: {chi2_total:.2f}")
    print(f"  χ²_V²: {chi2_v2:.2f}")
    print(f"  χ²_T3amp: {chi2_t3amp:.2f}")
    print(f"  χ²_T3phi: {chi2_t3phi:.2f}")

    # Check that chi2 components are non-negative
    assert chi2_v2 >= 0, "χ²_V² should be non-negative"
    assert chi2_t3amp >= 0, "χ²_T3amp should be non-negative"
    assert chi2_t3phi >= 0, "χ²_T3phi should be non-negative"

    # Check sum
    assert jnp.abs(chi2_total - (chi2_v2 + chi2_t3amp + chi2_t3phi)) < 1e-6, \
        "Total χ² should equal sum of components"

    print("  ✓ Chi-squared calculation test passed")


def test_gradient_computation():
    """Test gradient computation via JAX autodiff."""
    print("\nTesting gradient computation...")

    # Create geometry
    tess = tessellation_healpix(n=2)  # Small for speed
    geom = create_star(tess, radius=1.0, inc=45.0, PA=0.0, obliq=0.0)

    # Create brightness map
    image = jnp.ones(geom.npix) * 0.5

    # Create mock data
    oi_data = create_mock_oi_data(nuv=10, nt3=5)

    # Setup forward model with precomputed matrix
    fwd_model = create_forward_model(geom, oi_data)

    # Compute chi2 and gradient
    chi2, gradient = compute_chi2_gradient(
        image, geom, oi_data, fwd_model['polyft_matrix']
    )

    print(f"  χ²: {chi2:.4f}")
    print(f"  Gradient shape: {gradient.shape}")
    print(f"  Gradient range: [{gradient.min():.4e}, {gradient.max():.4e}]")
    print(f"  Gradient norm: {jnp.linalg.norm(gradient):.4e}")

    # Check gradient shape
    assert gradient.shape == image.shape, "Gradient shape mismatch"

    # Gradient should not be all zeros (unless at a critical point)
    assert jnp.linalg.norm(gradient) > 1e-10, "Gradient is zero"

    # Test finite differences to verify gradient
    eps = 1e-5
    idx_test = 10  # Test one pixel
    image_perturbed = image.at[idx_test].add(eps)

    chi2_perturbed = compute_chi2(
        image_perturbed, geom, oi_data, fwd_model['polyft_matrix']
    )

    # Finite difference approximation
    grad_fd = (chi2_perturbed - chi2) / eps

    # Compare with autodiff gradient
    grad_autodiff = gradient[idx_test]

    print(f"  Finite difference gradient[{idx_test}]: {grad_fd:.6e}")
    print(f"  Autodiff gradient[{idx_test}]: {grad_autodiff:.6e}")
    print(f"  Relative error: {abs(grad_fd - grad_autodiff) / abs(grad_fd):.2%}")

    # Should match within 1% (finite difference is approximate)
    assert abs(grad_fd - grad_autodiff) / abs(grad_fd) < 0.01, \
        "Gradient doesn't match finite difference"

    print("  ✓ Gradient computation test passed")


def test_forward_model_creation():
    """Test forward model creation and precomputation."""
    print("\nTesting forward model creation...")

    # Create geometry
    tess = tessellation_healpix(n=3)
    geom = create_star(tess, radius=1.0, inc=30.0, PA=45.0, obliq=0.0)

    # Create mock data
    oi_data = create_mock_oi_data(nuv=20, nt3=10)

    # Create forward model
    fwd_model = create_forward_model(geom, oi_data)

    # Check components
    assert 'polyft_matrix' in fwd_model, "Missing polyft_matrix"
    assert 'geom' in fwd_model, "Missing geom"
    assert 'oi_data' in fwd_model, "Missing oi_data"
    assert 'compute_chi2' in fwd_model, "Missing compute_chi2 function"
    assert 'compute_observables' in fwd_model, "Missing compute_observables function"

    # Check polyft_matrix shape
    n_visible = np.sum(geom.visible_mask)
    expected_shape = (oi_data.uv.shape[1], n_visible)
    actual_shape = fwd_model['polyft_matrix'].shape

    print(f"  Polyft matrix shape: {actual_shape} (expected: {expected_shape})")
    assert actual_shape == expected_shape, f"Wrong polyft_matrix shape"

    # Test forward model functions
    image = jnp.ones(geom.npix)
    chi2 = fwd_model['compute_chi2'](image)
    v2, t3amp, t3phi = fwd_model['compute_observables'](image)

    print(f"  χ² from forward model: {chi2:.4f}")
    assert chi2 >= 0, "χ² should be non-negative"

    print("  ✓ Forward model creation test passed")


def test_residuals_computation():
    """Test residuals computation."""
    print("\nTesting residuals computation...")

    # Create geometry
    tess = tessellation_healpix(n=3)
    geom = create_star(tess, radius=1.0, inc=45.0, PA=0.0, obliq=0.0)

    # Create brightness map
    image = jnp.ones(geom.npix)

    # Create mock data
    oi_data = create_mock_oi_data(nuv=15, nt3=8)

    # Compute residuals
    residuals = compute_residuals(image, geom, oi_data)

    print(f"  V² residual range: [{residuals['v2_residual'].min():.2f}, "
          f"{residuals['v2_residual'].max():.2f}]")
    print(f"  T3amp residual range: [{residuals['t3amp_residual'].min():.2f}, "
          f"{residuals['t3amp_residual'].max():.2f}]")
    print(f"  T3phi residual range: [{residuals['t3phi_residual'].min():.2f}, "
          f"{residuals['t3phi_residual'].max():.2f}]")

    # Check that all expected keys are present
    expected_keys = ['v2_residual', 't3amp_residual', 't3phi_residual',
                     'v2_model', 't3amp_model', 't3phi_model']
    for key in expected_keys:
        assert key in residuals, f"Missing key: {key}"

    # Check shapes
    assert residuals['v2_residual'].shape == (oi_data.nv2,)
    assert residuals['t3amp_residual'].shape == (oi_data.nt3amp,)
    assert residuals['t3phi_residual'].shape == (oi_data.nt3phi,)

    print("  ✓ Residuals computation test passed")


def test_reduced_chi2():
    """Test reduced chi-squared calculation."""
    print("\nTesting reduced χ² calculation...")

    chi2 = 100.0
    nv2 = 50
    nt3amp = 30
    nt3phi = 30
    n_params = 50

    chi2_reduced = compute_reduced_chi2(chi2, nv2, nt3amp, nt3phi, n_params)

    n_data = nv2 + nt3amp + nt3phi  # 110
    n_dof = n_data - n_params  # 60
    expected = chi2 / n_dof  # 100/60 = 1.667

    print(f"  χ²_reduced: {chi2_reduced:.4f} (expected: {expected:.4f})")

    assert abs(chi2_reduced - expected) < 1e-6, "Reduced χ² calculation wrong"

    print("  ✓ Reduced χ² test passed")


def test_image_perturbation_response():
    """Test that forward model responds correctly to image changes."""
    print("\nTesting forward model response to image perturbations...")

    # Create geometry
    tess = tessellation_healpix(n=3)
    geom = create_star(tess, radius=1.0, inc=45.0, PA=0.0, obliq=0.0)

    # Create mock data
    oi_data = create_mock_oi_data(nuv=20, nt3=10)

    # Create forward model
    fwd_model = create_forward_model(geom, oi_data)

    # Uniform image
    image_uniform = jnp.ones(geom.npix)
    chi2_uniform = fwd_model['compute_chi2'](image_uniform)

    # Add a bright spot
    image_spot = image_uniform.at[100].set(2.0)
    chi2_spot = fwd_model['compute_chi2'](image_spot)

    print(f"  χ² (uniform): {chi2_uniform:.4f}")
    print(f"  χ² (with spot): {chi2_spot:.4f}")
    print(f"  Δχ²: {abs(chi2_spot - chi2_uniform):.4f}")

    # Chi2 should change when image changes
    assert abs(chi2_spot - chi2_uniform) > 1e-6, \
        "χ² should change when image changes"

    print("  ✓ Image perturbation response test passed")


def run_all_tests():
    """Run all observables tests."""
    print("="*60)
    print("Running Observables (Forward Model) Tests")
    print("="*60)

    try:
        test_forward_model_uniform_disk()
        test_chi2_calculation()
        test_gradient_computation()
        test_forward_model_creation()
        test_residuals_computation()
        test_reduced_chi2()
        test_image_perturbation_response()

        print("\n" + "="*60)
        print("ALL OBSERVABLES TESTS PASSED ✓")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
