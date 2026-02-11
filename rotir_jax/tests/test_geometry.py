"""Tests for geometry/base.py module.

Tests verify:
1. Rotation matrix properties (orthogonality, determinant)
2. Visibility determination
3. Sky plane projection
4. StarGeometry creation
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')

from rotir_jax.tessellation.healpix import tessellation_healpix
from rotir_jax.geometry.base import (
    rotation_matrix,
    apply_rotation,
    visible_mask,
    sky_plane_projection,
    create_star,
    compute_limb_darkening,
    rotate_and_project,
)


def test_rotation_matrix_properties():
    """Test rotation matrix has correct mathematical properties."""
    print("Testing rotation matrix properties...")

    test_cases = [
        (0, 0, 0),       # No rotation
        (45, 0, 0),      # Inclination only
        (0, 45, 0),      # PA only
        (0, 0, 45),      # Obliquity only
        (30, 60, 90),    # All three
        (90, 180, 270),  # Edge cases
    ]

    for inc, PA, obliq in test_cases:
        R = rotation_matrix(inc, PA, obliq)

        # Test 1: Should be 3x3
        assert R.shape == (3, 3), f"Wrong shape: {R.shape}"

        # Test 2: Should be orthogonal (R.T @ R = I)
        RTR = R.T @ R
        identity = jnp.eye(3)
        orthogonality_error = jnp.max(jnp.abs(RTR - identity))

        # Test 3: Determinant should be +1 (proper rotation, not reflection)
        det = jnp.linalg.det(R)

        if orthogonality_error > 1e-6:
            print(f"  WARNING: inc={inc}, PA={PA}, obliq={obliq}")
            print(f"    Orthogonality error: {orthogonality_error:.2e}")

        if abs(det - 1.0) > 1e-6:
            print(f"  WARNING: inc={inc}, PA={PA}, obliq={obliq}")
            print(f"    Determinant: {det:.6f} (should be 1.0)")

        assert orthogonality_error < 1e-5, f"Not orthogonal: error={orthogonality_error}"
        assert abs(det - 1.0) < 1e-5, f"Wrong determinant: {det}"

    print("  ✓ All rotation matrices are valid")


def test_identity_rotation():
    """Test that zero angles give identity matrix."""
    print("\nTesting identity rotation...")

    R = rotation_matrix(0, 0, 0)
    identity = jnp.eye(3)
    error = jnp.max(jnp.abs(R - identity))

    print(f"  Max deviation from identity: {error:.2e}")
    assert error < 1e-10, f"Zero angles should give identity, error={error}"

    print("  ✓ Identity rotation test passed")


def test_pole_on_visibility():
    """Test visibility for pole-on view (inc=0)."""
    print("\nTesting pole-on visibility (inc=0)...")

    tess = tessellation_healpix(n=2)  # Small tessellation for testing
    geom = create_star(tess, radius=1.0, inc=0.0, PA=0.0, obliq=0.0)

    # For pole-on view, all pixels should be visible (northern hemisphere)
    # Actually, for a sphere viewed pole-on, we see one hemisphere
    n_visible = np.sum(geom.visible_mask)
    n_total = geom.npix

    print(f"  Visible pixels: {n_visible}/{n_total} ({100*n_visible/n_total:.1f}%)")

    # For pole-on, we should see roughly half (one hemisphere)
    assert 0.4 < n_visible/n_total < 0.6, \
        f"Pole-on should show ~50% of pixels, got {n_visible/n_total:.1%}"

    print("  ✓ Pole-on visibility test passed")


def test_edge_on_visibility():
    """Test visibility for edge-on view (inc=90)."""
    print("\nTesting edge-on visibility (inc=90)...")

    tess = tessellation_healpix(n=2)
    geom = create_star(tess, radius=1.0, inc=90.0, PA=0.0, obliq=0.0)

    n_visible = np.sum(geom.visible_mask)
    n_total = geom.npix

    print(f"  Visible pixels: {n_visible}/{n_total} ({100*n_visible/n_total:.1f}%)")

    # For edge-on, we should see roughly half (one hemisphere)
    assert 0.4 < n_visible/n_total < 0.6, \
        f"Edge-on should show ~50% of pixels, got {n_visible/n_total:.1%}"

    print("  ✓ Edge-on visibility test passed")


def test_sky_projection():
    """Test sky plane projection."""
    print("\nTesting sky plane projection...")

    tess = tessellation_healpix(n=2)
    radius = 2.5  # Solar radii

    # Pole-on view
    geom = create_star(tess, radius=radius, inc=0.0, PA=0.0, obliq=0.0)

    # Check that x_sky, y_sky have correct shape
    assert geom.x_sky.shape == (geom.npix, 5)
    assert geom.y_sky.shape == (geom.npix, 5)

    # For pole-on, z should be close to radius for visible pixels
    z_center = geom.vertices_xyz[:, 4, 2]
    z_visible = z_center[geom.visible_mask]

    # Visible pixels should have z > 0
    assert np.all(z_visible > 0), "Visible pixels should have z > 0"

    # For pole-on sphere, visible pixels should have z ≈ radius
    # (within reasonable tolerance due to tessellation)
    z_mean = np.mean(z_visible)
    print(f"  Pole-on: mean z of visible pixels = {z_mean:.3f} (radius={radius})")
    assert abs(z_mean - radius) < 0.5 * radius, "Mean z should be close to radius"

    # Check that projected radius is approximately correct
    r_sky = np.sqrt(geom.x_sky[:, 4]**2 + geom.y_sky[:, 4]**2)
    r_sky_visible = r_sky[geom.visible_mask]
    r_sky_max = np.max(r_sky_visible)

    print(f"  Max projected radius: {r_sky_max:.3f} (should be ≈ {radius})")
    assert r_sky_max < radius * 1.1, "Projected radius should not exceed stellar radius"

    print("  ✓ Sky projection test passed")


def test_foreshortening():
    """Test foreshortening factor (mu)."""
    print("\nTesting foreshortening factor μ...")

    tess = tessellation_healpix(n=3)

    # Test 1: Pole-on (inc=0)
    geom_pole = create_star(tess, radius=1.0, inc=0.0, PA=0.0, obliq=0.0)
    mu_pole = geom_pole.mu[geom_pole.visible_mask]

    # For pole-on, μ should vary from ~0 (limb) to ~1 (center)
    mu_max_pole = np.max(mu_pole)
    mu_min_pole = np.min(mu_pole)

    print(f"  Pole-on: μ range = [{mu_min_pole:.3f}, {mu_max_pole:.3f}]")
    assert mu_max_pole > 0.9, f"Max μ should be near 1 for pole-on"
    assert mu_min_pole < 0.3, f"Min μ should be near 0 at limb"

    # Test 2: Edge-on (inc=90)
    geom_edge = create_star(tess, radius=1.0, inc=90.0, PA=0.0, obliq=0.0)
    mu_edge = geom_edge.mu[geom_edge.visible_mask]

    mu_max_edge = np.max(mu_edge)
    mu_min_edge = np.min(mu_edge)

    print(f"  Edge-on: μ range = [{mu_min_edge:.3f}, {mu_max_edge:.3f}]")
    assert mu_max_edge > 0.9, f"Max μ should be near 1 at equator center"
    assert mu_min_edge < 0.3, f"Min μ should be near 0 at limb"

    print("  ✓ Foreshortening test passed")


def test_pixel_areas():
    """Test pixel area calculation."""
    print("\nTesting pixel areas...")

    tess = tessellation_healpix(n=3)
    radius = 1.0

    # Pole-on view
    geom = create_star(tess, radius=radius, inc=0.0, PA=0.0, obliq=0.0)

    # Total visible area should be approximately πr² (one hemisphere projected)
    total_area = np.sum(geom.pixel_areas[geom.visible_mask])
    expected_area = np.pi * radius**2

    print(f"  Total projected area: {total_area:.4f}")
    print(f"  Expected (πr²): {expected_area:.4f}")
    print(f"  Ratio: {total_area/expected_area:.4f}")

    # Should be close to πr² (within 10% due to discretization)
    assert 0.85 < total_area/expected_area < 1.15, \
        f"Total area mismatch: {total_area} vs {expected_area}"

    print("  ✓ Pixel area test passed")


def test_limb_darkening():
    """Test limb darkening calculation."""
    print("\nTesting limb darkening...")

    # Test uniform disk (u1=u2=0)
    mu = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    intensity_uniform = compute_limb_darkening(mu, u1=0.0, u2=0.0)

    assert jnp.allclose(intensity_uniform, 1.0), "Uniform disk should have I=1 everywhere"

    # Test linear law (u1=0.6, u2=0.0)
    intensity_linear = compute_limb_darkening(mu, u1=0.6, u2=0.0)

    # At μ=1 (center), I = 1
    # At μ=0 (limb), I = 1 - 0.6 = 0.4
    assert abs(intensity_linear[-1] - 1.0) < 1e-6, "Center should be 1.0"
    assert abs(intensity_linear[0] - 0.4) < 1e-6, "Limb should be 0.4"

    # Test quadratic law
    intensity_quad = compute_limb_darkening(mu, u1=0.5, u2=0.3)

    # At μ=1: I = 1
    # At μ=0: I = 1 - 0.5 - 0.3 = 0.2
    assert abs(intensity_quad[-1] - 1.0) < 1e-6, "Center should be 1.0"
    assert abs(intensity_quad[0] - 0.2) < 1e-6, "Limb should be 0.2"

    print("  ✓ Limb darkening test passed")


def test_rotate_and_project():
    """Test convenience function."""
    print("\nTesting rotate_and_project convenience function...")

    tess = tessellation_healpix(n=2)
    x_sky, y_sky, vis_mask, mu = rotate_and_project(
        tess, radius=1.0, inc=45.0, PA=30.0, obliq=60.0
    )

    assert x_sky.shape == (tess.npix, 5)
    assert y_sky.shape == (tess.npix, 5)
    assert vis_mask.shape == (tess.npix,)
    assert mu.shape == (tess.npix,)

    print("  ✓ Rotate and project test passed")


def run_all_tests():
    """Run all geometry tests."""
    print("="*60)
    print("Running Geometry Tests")
    print("="*60)

    try:
        test_rotation_matrix_properties()
        test_identity_rotation()
        test_pole_on_visibility()
        test_edge_on_visibility()
        test_sky_projection()
        test_foreshortening()
        test_pixel_areas()
        test_limb_darkening()
        test_rotate_and_project()

        print("\n" + "="*60)
        print("ALL GEOMETRY TESTS PASSED ✓")
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
