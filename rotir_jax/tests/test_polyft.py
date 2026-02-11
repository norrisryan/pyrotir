"""Tests for polygon Fourier transform module.

Tests verify:
1. Polygon area calculation (Shoelace formula)
2. Edge Fourier transform contributions
3. Complete polygon FT
4. V2 and closure phase (T3) calculations
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')

from rotir_jax.forward_model.polyft import (
    polygon_area,
    edge_fourier_contribution,
    polygon_fourier_transform,
    setup_polyft_matrix,
    cvis_to_v2,
    cvis_to_t3,
    mod360,
)


def test_polygon_area():
    """Test Shoelace formula for polygon area."""
    print("Testing polygon area calculation...")

    # Test 1: Unit square (1x1)
    x = jnp.array([[0.0, 1.0, 1.0, 0.0]])  # (1, 4)
    y = jnp.array([[0.0, 0.0, 1.0, 1.0]])  # (1, 4)
    area = polygon_area(x, y)

    print(f"  Unit square area: {area[0]:.6f} (expected: 1.0)")
    assert jnp.abs(area[0] - 1.0) < 1e-6, f"Unit square area wrong: {area[0]}"

    # Test 2: Rectangle (2x3)
    x = jnp.array([[0.0, 2.0, 2.0, 0.0]])
    y = jnp.array([[0.0, 0.0, 3.0, 3.0]])
    area = polygon_area(x, y)

    print(f"  Rectangle (2x3) area: {area[0]:.6f} (expected: 6.0)")
    assert jnp.abs(area[0] - 6.0) < 1e-6, f"Rectangle area wrong: {area[0]}"

    # Test 3: Triangle (as degenerate quadrilateral)
    x = jnp.array([[0.0, 1.0, 0.5, 0.5]])  # Last two vertices same
    y = jnp.array([[0.0, 0.0, 1.0, 1.0]])
    area = polygon_area(x, y)

    print(f"  Triangle area: {area[0]:.6f} (expected: 0.5)")
    assert jnp.abs(area[0] - 0.5) < 1e-6, f"Triangle area wrong: {area[0]}"

    # Test 4: Multiple polygons at once
    x = jnp.array([
        [0.0, 1.0, 1.0, 0.0],  # Unit square
        [0.0, 2.0, 2.0, 0.0],  # 2x3 rectangle
    ])
    y = jnp.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 3.0, 3.0],
    ])
    areas = polygon_area(x, y)

    print(f"  Batch areas: {areas} (expected: [1.0, 6.0])")
    assert jnp.allclose(areas, jnp.array([1.0, 6.0])), "Batch areas wrong"

    print("  ✓ Polygon area test passed")


def test_edge_fourier_dc():
    """Test edge FT at zero frequency (DC component)."""
    print("\nTesting edge Fourier transform at DC (u=v=0)...")

    # Simple edge from (0,0) to (1,0)
    x1 = jnp.array([0.0])
    y1 = jnp.array([0.0])
    x2 = jnp.array([1.0])
    y2 = jnp.array([0.0])

    # Zero frequency
    kx = jnp.array([0.0])
    ky = jnp.array([0.0])

    edge_ft = edge_fourier_contribution(x1, y1, x2, y2, kx, ky)

    print(f"  Edge FT at DC: {edge_ft[0, 0]}")
    # At DC, edge contribution should be related to edge vector
    # sinc(0) = 1, cis(0) = 1, weight = ky*dx - kx*dy = 0*1 - 0*0 = 0
    # So DC component should be 0 for horizontal edge
    assert jnp.abs(edge_ft[0, 0]) < 1e-10, "DC component should be zero for edge"

    print("  ✓ Edge FT DC test passed")


def test_polygon_ft_uniform_disk():
    """Test polygon FT for a circular disk approximation."""
    print("\nTesting polygon FT for uniform disk...")

    # Create a simple square at origin
    # This approximates a uniform disk for low spatial frequencies
    size = 1.0  # mas
    x = jnp.array([[-size, size, size, -size]]) / 2
    y = jnp.array([[-size, -size, size, size]]) / 2

    # Uniform flux
    flux = jnp.array([1.0])

    # Test at several spatial frequencies
    u_test = jnp.array([0.0, 10.0, 50.0, 100.0])  # cycles/mas
    v_test = jnp.array([0.0, 0.0, 0.0, 0.0])

    cvis = polygon_fourier_transform(x, y, u_test, v_test, flux)

    print(f"  Complex visibilities at u=[0, 10, 50, 100]:")
    print(f"    Amplitudes: {jnp.abs(cvis)}")
    print(f"    Phases (deg): {jnp.angle(cvis) * 180/jnp.pi}")

    # At u=0, visibility should be 1.0 (normalized)
    assert jnp.abs(jnp.abs(cvis[0]) - 1.0) < 1e-6, "DC visibility should be 1.0"

    # At higher frequencies, visibility should decrease
    assert jnp.abs(cvis[1]) < jnp.abs(cvis[0]), "Visibility should decrease with frequency"

    print("  ✓ Polygon FT uniform disk test passed")


def test_polyft_matrix():
    """Test precomputed polygon FT matrix."""
    print("\nTesting polygon FT matrix computation...")

    # Simple geometry
    x = jnp.array([[0.0, 1.0, 1.0, 0.0]])
    y = jnp.array([[0.0, 0.0, 1.0, 1.0]])

    u = jnp.array([0.0, 10.0, 20.0])
    v = jnp.array([0.0, 10.0, 20.0])

    polyft_matrix = setup_polyft_matrix(x, y, u, v)

    # Check shape
    assert polyft_matrix.shape == (3, 1), f"Wrong shape: {polyft_matrix.shape}"

    # Check that matrix is complex
    assert jnp.iscomplexobj(polyft_matrix), "Matrix should be complex"

    # Apply to flux and check consistency with direct calculation
    flux = jnp.array([1.0])
    cvis_matrix = jnp.dot(polyft_matrix, flux) / jnp.sum(flux)
    cvis_direct = polygon_fourier_transform(x, y, u, v, flux)

    error = jnp.max(jnp.abs(cvis_matrix - cvis_direct))
    print(f"  Matrix vs direct calculation error: {error:.2e}")
    assert error < 1e-10, f"Matrix and direct should match: error={error}"

    print("  ✓ Polygon FT matrix test passed")


def test_v2_calculation():
    """Test squared visibility calculation."""
    print("\nTesting V² calculation...")

    # Test complex visibilities
    cvis = jnp.array([1.0 + 0.0j, 0.5 + 0.5j, 0.0 + 1.0j])

    v2 = cvis_to_v2(cvis)

    expected_v2 = jnp.array([1.0, 0.5, 1.0])

    print(f"  V² values: {v2}")
    print(f"  Expected: {expected_v2}")

    assert jnp.allclose(v2, expected_v2), f"V² calculation wrong"

    print("  ✓ V² calculation test passed")


def test_t3_calculation():
    """Test closure phase (T3) calculation."""
    print("\nTesting closure phase calculation...")

    # Create test visibilities with known phases
    # cvis = amplitude * exp(i*phase)
    amp1, phase1 = 0.8, 30.0 * jnp.pi / 180  # 30 degrees
    amp2, phase2 = 0.7, 45.0 * jnp.pi / 180  # 45 degrees
    amp3, phase3 = 0.6, 60.0 * jnp.pi / 180  # 60 degrees

    cvis = jnp.array([
        amp1 * jnp.exp(1j * phase1),
        amp2 * jnp.exp(1j * phase2),
        amp3 * jnp.exp(1j * phase3),
    ])

    # Single triangle using indices 0, 1, 2
    indx1 = jnp.array([0])
    indx2 = jnp.array([1])
    indx3 = jnp.array([2])

    t3, t3amp, t3phi = cvis_to_t3(cvis, indx1, indx2, indx3)

    # Expected results
    expected_amp = amp1 * amp2 * amp3
    expected_phi = (phase1 + phase2 + phase3) * 180 / jnp.pi

    print(f"  T3 amplitude: {t3amp[0]:.6f} (expected: {expected_amp:.6f})")
    print(f"  Closure phase: {t3phi[0]:.2f}° (expected: {expected_phi:.2f}°)")

    assert jnp.abs(t3amp[0] - expected_amp) < 1e-6, "T3 amplitude wrong"
    assert jnp.abs(t3phi[0] - expected_phi) < 1e-4, "Closure phase wrong"

    print("  ✓ Closure phase calculation test passed")


def test_mod360():
    """Test angle wrapping to [-180, 180]."""
    print("\nTesting angle wrapping (mod360)...")

    test_cases = [
        (0.0, 0.0),
        (90.0, 90.0),
        (180.0, 180.0),
        (270.0, -90.0),
        (360.0, 0.0),
        (450.0, 90.0),
        (-90.0, -90.0),
        (-270.0, 90.0),
    ]

    for input_angle, expected in test_cases:
        result = mod360(jnp.array([input_angle]))[0]
        print(f"  mod360({input_angle:6.1f}°) = {result:6.1f}° (expected: {expected:6.1f}°)")
        assert jnp.abs(result - expected) < 1e-6, \
            f"mod360({input_angle}) = {result}, expected {expected}"

    print("  ✓ Angle wrapping test passed")


def test_symmetry():
    """Test symmetry properties of polygon FT."""
    print("\nTesting symmetry properties...")

    # Symmetric square centered at origin
    size = 1.0
    x = jnp.array([[-size, size, size, -size]]) / 2
    y = jnp.array([[-size, -size, size, size]]) / 2
    flux = jnp.array([1.0])

    # Test at (u, v) and (-u, -v)
    u_pos = jnp.array([10.0, 20.0])
    v_pos = jnp.array([10.0, 20.0])
    u_neg = -u_pos
    v_neg = -v_pos

    cvis_pos = polygon_fourier_transform(x, y, u_pos, v_pos, flux)
    cvis_neg = polygon_fourier_transform(x, y, u_neg, v_neg, flux)

    # For real-valued image, V(-u,-v) = V*(u,v)
    error = jnp.max(jnp.abs(cvis_neg - jnp.conj(cvis_pos)))

    print(f"  Hermitian symmetry error: {error:.2e}")
    assert error < 1e-6, f"Hermitian symmetry violated: error={error}"

    print("  ✓ Symmetry test passed")


def run_all_tests():
    """Run all polygon FT tests."""
    print("="*60)
    print("Running Polygon Fourier Transform Tests")
    print("="*60)

    try:
        test_polygon_area()
        test_edge_fourier_dc()
        test_polygon_ft_uniform_disk()
        test_polyft_matrix()
        test_v2_calculation()
        test_t3_calculation()
        test_mod360()
        test_symmetry()

        print("\n" + "="*60)
        print("ALL POLYGON FT TESTS PASSED ✓")
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
