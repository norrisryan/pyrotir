"""Tests for HEALPix tessellation module.

Tests verify:
1. Vertex computation matches healpy.boundaries()
2. All vertices lie on unit sphere
3. Pixel centers match healpy.pix2ang()
4. Neighbor computation is correct
"""

import numpy as np
import healpy as hp
import sys
sys.path.append('..')

from rotir_jax.tessellation.healpix import (
    tessellation_healpix,
    get_neighbors,
    tv_regularization_matrices
)


def test_healpix_vertices():
    """Verify vertex computation against healpy.boundaries()."""
    print("Testing HEALPix vertex computation...")

    for n in [2, 3]:  # Test n=2 and n=3
        print(f"\n  Testing n={n} (nside={2**n}, npix={12 * (2**n)**2})")
        tess = tessellation_healpix(n)
        nside = 2**n

        # Test a sample of pixels
        n_test = min(100, tess.npix)
        test_indices = np.linspace(0, tess.npix - 1, n_test, dtype=int)

        max_error = 0.0
        for i in test_indices:
            # Get healpy boundaries
            boundaries_hp = hp.boundaries(nside, i, step=1, nest=True)
            # Shape: (3, 4) with rows = (x, y, z)

            # Get our vertices (first 4 points, excluding center at index 4)
            our_vertices = tess.unit_xyz[i, :4, :]  # Shape: (4, 3)

            # Check if all healpy vertices are present in our set
            # (order may differ)
            for hp_vert in boundaries_hp.T:  # Iterate over healpy's 4 vertices
                # Find closest match in our vertices
                distances = np.linalg.norm(our_vertices - hp_vert, axis=1)
                min_dist = np.min(distances)
                max_error = max(max_error, min_dist)

                if min_dist > 1e-6:
                    print(f"    WARNING: Pixel {i} has mismatched vertex")
                    print(f"      healpy vertex: {hp_vert}")
                    print(f"      min distance to our vertices: {min_dist}")

        print(f"    Max vertex mismatch: {max_error:.2e} (should be < 1e-6)")
        assert max_error < 1e-5, f"Vertex mismatch too large: {max_error}"

    print("✓ Vertex computation test passed")


def test_unit_sphere():
    """Verify all vertices lie on unit sphere."""
    print("\nTesting that all vertices lie on unit sphere...")

    for n in [2, 3, 4]:
        tess = tessellation_healpix(n)

        # Check all vertices (including center)
        radii = np.linalg.norm(tess.unit_xyz, axis=2)  # Shape: (npix, 5)
        max_dev = np.max(np.abs(radii - 1.0))

        print(f"  n={n}: max deviation from r=1: {max_dev:.2e}")
        assert max_dev < 1e-6, f"Vertices not on unit sphere for n={n}"

    print("✓ Unit sphere test passed")


def test_pixel_centers():
    """Verify pixel centers match healpy.pix2ang()."""
    print("\nTesting pixel centers...")

    for n in [2, 3]:
        tess = tessellation_healpix(n)
        nside = 2**n

        # Get healpy centers
        theta_hp, phi_hp = hp.pix2ang(nside, np.arange(tess.npix), nest=True)

        # Get our centers (index 4)
        theta_ours = tess.unit_spherical[:, 4, 1]
        phi_ours = tess.unit_spherical[:, 4, 2]

        # Check match
        theta_error = np.max(np.abs(theta_hp - theta_ours))
        phi_error = np.max(np.abs(phi_hp - phi_ours))

        print(f"  n={n}: theta error={theta_error:.2e}, phi error={phi_error:.2e}")
        assert theta_error < 1e-6, f"Theta mismatch for n={n}"
        assert phi_error < 1e-6, f"Phi mismatch for n={n}"

    print("✓ Pixel center test passed")


def test_spherical_to_cartesian():
    """Verify spherical <-> Cartesian conversion consistency."""
    print("\nTesting spherical-Cartesian consistency...")

    for n in [2, 3]:
        tess = tessellation_healpix(n)

        # Recompute xyz from spherical
        theta = tess.unit_spherical[:, :, 1]
        phi = tess.unit_spherical[:, :, 2]

        x_recomp = np.sin(theta) * np.cos(phi)
        y_recomp = np.sin(theta) * np.sin(phi)
        z_recomp = np.cos(theta)

        # Compare with stored xyz
        error_x = np.max(np.abs(tess.unit_xyz[:, :, 0] - x_recomp))
        error_y = np.max(np.abs(tess.unit_xyz[:, :, 1] - y_recomp))
        error_z = np.max(np.abs(tess.unit_xyz[:, :, 2] - z_recomp))

        max_error = max(error_x, error_y, error_z)
        print(f"  n={n}: max xyz reconstruction error={max_error:.2e}")
        assert max_error < 1e-6, f"Spherical-Cartesian inconsistency for n={n}"

    print("✓ Spherical-Cartesian consistency test passed")


def test_neighbors():
    """Test neighbor computation."""
    print("\nTesting neighbor computation...")

    for n in [2, 3]:
        neighbors = get_neighbors(n)
        nside = 2**n
        npix = hp.nside2npix(nside)

        # Check that we have the right number of pixels
        assert len(neighbors) == npix

        # Most pixels should have 8 neighbors
        neighbor_counts = [len(nb) for nb in neighbors]
        avg_neighbors = np.mean(neighbor_counts)
        min_neighbors = np.min(neighbor_counts)
        max_neighbors = np.max(neighbor_counts)

        print(f"  n={n}: avg neighbors={avg_neighbors:.1f}, "
              f"min={min_neighbors}, max={max_neighbors}")

        # Check reasonableness: should be between 7 and 8
        assert min_neighbors >= 7, f"Too few neighbors for n={n}"
        assert max_neighbors <= 8, f"Too many neighbors for n={n}"

    print("✓ Neighbor computation test passed")


def test_tv_matrices():
    """Test TV regularization matrix construction."""
    print("\nTesting TV regularization matrices...")

    for n in [2, 3]:
        grad_matrix, laplacian, neighbors = tv_regularization_matrices(n)
        npix = 12 * (2**n)**2

        # Check shapes
        assert grad_matrix.shape == (npix, npix)
        assert laplacian.shape == (npix, npix)

        # Check that gradient matrix has the right structure
        # Diagonal should equal number of neighbors
        for k in range(min(100, npix)):
            n_neighbors = len(neighbors[k])
            diag_val = grad_matrix[k, k]
            assert abs(diag_val - n_neighbors) < 1e-10, \
                f"Diagonal mismatch at pixel {k}"

        print(f"  n={n}: grad_matrix sparsity={grad_matrix.nnz / npix**2:.2%}")

    print("✓ TV regularization matrix test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running HEALPix Tessellation Tests")
    print("="*60)

    try:
        test_healpix_vertices()
        test_unit_sphere()
        test_pixel_centers()
        test_spherical_to_cartesian()
        test_neighbors()
        test_tv_matrices()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
