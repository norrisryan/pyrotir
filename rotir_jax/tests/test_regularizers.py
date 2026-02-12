"""Tests for regularization functions.

Tests verify:
1. Maximum entropy (MEM)
2. Total variation L1 (TV)
3. Total variation L2 (TV2)
4. Mean regularization
5. Bias regularization
6. Difference matrix construction
7. Combined regularizers
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')

from rotir_jax.regularization.regularizers import (
    maximum_entropy,
    total_variation_l1,
    total_variation_l2,
    mean_regularization,
    bias_regularization,
    build_difference_matrix,
    apply_regularizers,
)
from rotir_jax.tessellation.healpix import tessellation_healpix


def test_maximum_entropy():
    """Test maximum entropy regularizer."""
    print("Testing maximum entropy (MEM)...")

    # Uniform distribution should have low entropy penalty
    x_uniform = jnp.ones(100)
    f_uniform, g_uniform = maximum_entropy(x_uniform)

    print(f"  Uniform distribution:")
    print(f"    MEM = {f_uniform:.6f}")
    print(f"    Gradient norm = {jnp.linalg.norm(g_uniform):.6f}")

    # Peaked distribution should have high entropy penalty
    x_peaked = jnp.zeros(100)
    x_peaked = x_peaked.at[50].set(100.0)
    f_peaked, g_peaked = maximum_entropy(x_peaked)

    print(f"  Peaked distribution:")
    print(f"    MEM = {f_peaked:.6f}")
    print(f"    Gradient norm = {jnp.linalg.norm(g_peaked):.6f}")

    # Peaked should have higher penalty
    assert f_peaked > f_uniform, "Peaked should have higher MEM penalty"

    # Check gradient shape
    assert g_uniform.shape == x_uniform.shape, "Gradient shape mismatch"

    print("  ✓ Maximum entropy test passed")


def test_total_variation_l1():
    """Test TV-L1 regularizer."""
    print("\nTesting Total Variation L1...")

    # Create simple test case
    npix = 64
    tess = tessellation_healpix(n=3)
    D = build_difference_matrix(tess)

    # Smooth image (low TV)
    x_smooth = jnp.ones(npix)
    f_smooth, g_smooth = total_variation_l1(x_smooth, D)

    print(f"  Smooth image (constant):")
    print(f"    TV-L1 = {f_smooth:.6f}")

    # Should be zero for constant image
    assert jnp.abs(f_smooth) < 1e-6, "Constant image should have zero TV"

    # Step function (high TV)
    x_step = jnp.ones(npix)
    x_step = x_step.at[npix//2:].set(2.0)
    f_step, g_step = total_variation_l1(x_step, D)

    print(f"  Step function:")
    print(f"    TV-L1 = {f_step:.6f}")

    # Step should have higher TV
    assert f_step > f_smooth, "Step function should have higher TV"

    # Check gradient shape
    assert g_step.shape == x_step.shape, "Gradient shape mismatch"

    print("  ✓ Total Variation L1 test passed")


def test_total_variation_l2():
    """Test TV-L2 regularizer."""
    print("\nTesting Total Variation L2...")

    npix = 64
    tess = tessellation_healpix(n=3)
    D = build_difference_matrix(tess)

    # Smooth image
    x_smooth = jnp.ones(npix) * 5.0
    f_smooth, g_smooth = total_variation_l2(x_smooth, D)

    print(f"  Smooth image:")
    print(f"    TV-L2 = {f_smooth:.6f}")

    # Should be near zero
    assert jnp.abs(f_smooth) < 1e-6, "Constant image should have zero TV-L2"

    # Gradient (linear)
    x_grad = jnp.linspace(1.0, 2.0, npix)
    f_grad, g_grad = total_variation_l2(x_grad, D)

    print(f"  Linear gradient:")
    print(f"    TV-L2 = {f_grad:.6f}")

    # Linear should have non-zero TV-L2
    assert f_grad > 0, "Gradient should have positive TV-L2"

    # Check gradient is smooth (L2 is differentiable)
    assert jnp.all(jnp.isfinite(g_grad)), "Gradient should be finite everywhere"

    print("  ✓ Total Variation L2 test passed")


def test_mean_regularization():
    """Test mean regularization."""
    print("\nTesting mean regularization...")

    # Uniform (low penalty)
    x_uniform = jnp.ones(100) * 10.0
    f_uniform, g_uniform = mean_regularization(x_uniform)

    print(f"  Uniform:")
    print(f"    Mean reg = {f_uniform:.6f}")

    # Should be zero (all equal to mean)
    assert jnp.abs(f_uniform) < 1e-6, "Uniform should have zero mean penalty"

    # Variable (higher penalty)
    x_var = jnp.concatenate([jnp.ones(50) * 5.0, jnp.ones(50) * 15.0])
    f_var, g_var = mean_regularization(x_var)

    print(f"  Variable:")
    print(f"    Mean reg = {f_var:.6f}")

    # Should be positive
    assert f_var > 0, "Variable should have positive mean penalty"

    # Check gradient
    assert g_var.shape == x_var.shape, "Gradient shape mismatch"

    # Gradient should be -1 for low values, +1 for high values
    assert jnp.all(g_var[:50] == -1), "Low values should have gradient -1"
    assert jnp.all(g_var[50:] == 1), "High values should have gradient +1"

    print("  ✓ Mean regularization test passed")


def test_bias_regularization():
    """Test asymmetric bias regularization."""
    print("\nTesting bias regularization...")

    # Uniform (zero penalty)
    x_uniform = jnp.ones(100) * 10.0
    f_uniform, g_uniform = bias_regularization(x_uniform, bias_factor=2.0)

    print(f"  Uniform:")
    print(f"    Bias reg = {f_uniform:.6f}")

    # Should be near zero
    assert jnp.abs(f_uniform) < 1e-6, "Uniform should have zero bias penalty"

    # Spots (cool, below mean)
    x_spots = jnp.ones(100) * 10.0
    x_spots = x_spots.at[40:50].set(5.0)  # Cool spot

    f_spots, g_spots = bias_regularization(x_spots, bias_factor=2.0)

    print(f"  With cool spot:")
    print(f"    Bias reg = {f_spots:.6f}")

    # Faculae (hot, above mean)
    x_faculae = jnp.ones(100) * 10.0
    x_faculae = x_faculae.at[40:50].set(15.0)  # Hot facula

    f_faculae, g_faculae = bias_regularization(x_faculae, bias_factor=2.0)

    print(f"  With hot facula:")
    print(f"    Bias reg = {f_faculae:.6f}")

    # Hot features should have higher penalty with bias_factor > 1
    assert f_faculae > f_spots, "Hot features should have higher penalty"

    print("  ✓ Bias regularization test passed")


def test_difference_matrix():
    """Test difference matrix construction."""
    print("\nTesting difference matrix...")

    tess = tessellation_healpix(n=3)
    D = build_difference_matrix(tess)

    npix = tess.npix
    nedges = D.shape[0]

    print(f"  Tessellation: {npix} pixels")
    print(f"  Difference matrix: {nedges} edges")
    print(f"  Matrix shape: {D.shape}")

    # Each row should have +1 and -1
    row_sums = jnp.sum(jnp.abs(D), axis=1)
    print(f"  Row sums: min={row_sums.min():.1f}, max={row_sums.max():.1f}")

    # Most rows should sum to 2 (one +1, one -1)
    assert jnp.all(row_sums >= 1.9) and jnp.all(row_sums <= 2.1), \
        "Each edge should connect two pixels"

    # Test on constant image
    x_const = jnp.ones(npix) * 5.0
    Dx = D @ x_const

    print(f"  Constant image differences: max={jnp.max(jnp.abs(Dx)):.6f}")

    # Should be all zeros
    assert jnp.max(jnp.abs(Dx)) < 1e-6, "Constant image should have zero differences"

    # Test on gradient
    x_grad = jnp.arange(npix, dtype=float)
    Dx_grad = D @ x_grad

    print(f"  Gradient differences: mean={jnp.mean(jnp.abs(Dx_grad)):.2f}")

    # Should have non-zero differences
    assert jnp.mean(jnp.abs(Dx_grad)) > 0, "Gradient should have non-zero differences"

    print("  ✓ Difference matrix test passed")


def test_apply_regularizers():
    """Test combined regularizers."""
    print("\nTesting combined regularizers...")

    npix = 64
    tess = tessellation_healpix(n=3)
    D = build_difference_matrix(tess)

    x = jnp.ones(npix) * 10.0 + jnp.random.normal(size=npix) * 0.5

    # Define regularizers
    regularizers = [
        {"type": "mem", "weight": 0.1},
        {"type": "tv", "weight": 0.01},
        {"type": "mean", "weight": 0.001},
    ]

    f_reg, g_reg = apply_regularizers(x, regularizers, diff_matrix=D)

    print(f"  Combined regularization:")
    print(f"    Total value = {f_reg:.6f}")
    print(f"    Gradient norm = {jnp.linalg.norm(g_reg):.6f}")

    # Should be positive
    assert f_reg > 0, "Combined regularization should be positive"

    # Gradient should have correct shape
    assert g_reg.shape == x.shape, "Gradient shape mismatch"

    # Gradient should be finite
    assert jnp.all(jnp.isfinite(g_reg)), "Gradient should be finite"

    print("  ✓ Combined regularizers test passed")


def test_regularizer_gradients():
    """Test that regularizer gradients are correct (finite differences)."""
    print("\nTesting regularizer gradients...")

    npix = 64
    tess = tessellation_healpix(n=3)
    D = build_difference_matrix(tess)

    x = jnp.ones(npix) * 10.0 + jnp.random.normal(key=jax.random.PRNGKey(42), shape=(npix,)) * 0.5

    epsilon = 1e-5

    # Test MEM gradient
    f_mem, g_mem = maximum_entropy(x)

    # Finite difference
    g_mem_fd = jnp.zeros_like(x)
    for i in range(min(npix, 10)):  # Test first 10 pixels
        x_plus = x.at[i].add(epsilon)
        f_plus, _ = maximum_entropy(x_plus)
        g_mem_fd = g_mem_fd.at[i].set((f_plus - f_mem) / epsilon)

    # Compare
    grad_error = jnp.max(jnp.abs(g_mem[:10] - g_mem_fd[:10]))
    print(f"  MEM gradient error: {grad_error:.6e}")
    assert grad_error < 1e-4, "MEM gradient incorrect"

    # Test TV-L2 gradient (differentiable everywhere)
    f_tv2, g_tv2 = total_variation_l2(x, D)

    g_tv2_fd = jnp.zeros_like(x)
    for i in range(min(npix, 10)):
        x_plus = x.at[i].add(epsilon)
        f_plus, _ = total_variation_l2(x_plus, D)
        g_tv2_fd = g_tv2_fd.at[i].set((f_plus - f_tv2) / epsilon)

    grad_error_tv2 = jnp.max(jnp.abs(g_tv2[:10] - g_tv2_fd[:10]))
    print(f"  TV-L2 gradient error: {grad_error_tv2:.6e}")
    assert grad_error_tv2 < 1e-4, "TV-L2 gradient incorrect"

    print("  ✓ Regularizer gradients test passed")


def test_regularizer_edge_cases():
    """Test edge cases and numerical stability."""
    print("\nTesting edge cases...")

    # Zero image
    x_zero = jnp.zeros(100) + 1e-10  # Small offset to avoid log(0)
    f_mem, g_mem = maximum_entropy(x_zero, epsilon=1e-9)

    print(f"  Near-zero image:")
    print(f"    MEM = {f_mem:.6f}")

    assert jnp.isfinite(f_mem), "MEM should handle near-zero"
    assert jnp.all(jnp.isfinite(g_mem)), "MEM gradient should be finite"

    # Large values
    x_large = jnp.ones(100) * 1e6
    f_mem_large, g_mem_large = maximum_entropy(x_large)

    print(f"  Large values:")
    print(f"    MEM = {f_mem_large:.6f}")

    assert jnp.isfinite(f_mem_large), "MEM should handle large values"

    # Single pixel
    x_single = jnp.array([10.0])
    f_mean, g_mean = mean_regularization(x_single)

    print(f"  Single pixel:")
    print(f"    Mean reg = {f_mean:.6f}")

    # Should be zero (only one pixel, equals mean)
    assert jnp.abs(f_mean) < 1e-6, "Single pixel should have zero mean penalty"

    print("  ✓ Edge cases test passed")


def run_all_tests():
    """Run all regularizer tests."""
    print("="*60)
    print("Running Regularization Tests")
    print("="*60)

    try:
        test_maximum_entropy()
        test_total_variation_l1()
        test_total_variation_l2()
        test_mean_regularization()
        test_bias_regularization()
        test_difference_matrix()
        test_apply_regularizers()

        # These require JAX
        try:
            import jax
            test_regularizer_gradients()
        except ImportError:
            print("\n⚠ Skipping gradient tests (JAX required)")

        test_regularizer_edge_cases()

        print("\n" + "="*60)
        print("ALL REGULARIZATION TESTS PASSED ✓")
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
