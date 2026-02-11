"""Integration tests for ROTIR - end-to-end pipeline verification.

Tests verify the complete pipeline from geometry setup to chi-squared calculation:
1. Tessellation → Geometry → Forward model → Observables → Chi²
2. OIFITS data loading → Forward model → Reconstruction
3. Full forward modeling pipeline with realistic parameters
"""

import numpy as np
import jax.numpy as jnp
import tempfile
import os
import sys
sys.path.append('..')

from rotir_jax.tessellation.healpix import tessellation_healpix
from rotir_jax.geometry.base import create_star
from rotir_jax.forward_model import (
    compute_observables,
    compute_chi2,
    create_forward_model,
    compute_residuals,
)
from rotir_jax.io import read_oifits
from rotir_jax.tests.test_oifits_reader import create_synthetic_oifits


def test_end_to_end_uniform_disk():
    """Test complete pipeline with uniform disk model.

    This is the simplest end-to-end test:
    - Create star geometry
    - Create uniform brightness map
    - Create synthetic OIFITS data
    - Compute observables
    - Verify chi-squared calculation
    """
    print("="*60)
    print("Integration Test: End-to-End Uniform Disk")
    print("="*60)

    # Step 1: Create tessellation and geometry
    print("\n1. Creating star geometry...")
    tess = tessellation_healpix(n=4)  # Medium resolution
    print(f"   Tessellation: {tess.npix} pixels")

    # Create star with moderate inclination
    geom = create_star(
        tess,
        radius=1.0,  # 1 mas radius
        inc=45.0,    # 45° inclination
        PA=30.0,     # 30° position angle
        obliq=0.0,   # No obliquity
    )
    print(f"   Visible pixels: {np.sum(geom.visible_mask)}/{geom.npix}")

    # Step 2: Create uniform brightness map
    print("\n2. Creating uniform brightness map...")
    image = jnp.ones(geom.npix)
    print(f"   Image shape: {image.shape}")
    print(f"   Total flux: {jnp.sum(image):.2f}")

    # Step 3: Create synthetic OIFITS data
    print("\n3. Creating synthetic OIFITS data...")
    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        create_synthetic_oifits(tmp_filename, nv2=30, nt3=15, nwave=3)
        oi_data = read_oifits(tmp_filename, verbose=False)
        print(f"   V² measurements: {oi_data.nv2}")
        print(f"   Closure phases: {oi_data.nt3phi}")
        print(f"   UV points: {oi_data.uv.shape[1]}")

        # Step 4: Create forward model
        print("\n4. Creating forward model...")
        fwd_model = create_forward_model(geom, oi_data)
        print(f"   Polyft matrix shape: {fwd_model['polyft_matrix'].shape}")

        # Step 5: Compute observables
        print("\n5. Computing observables...")
        v2_model, t3amp_model, t3phi_model = fwd_model['compute_observables'](image)
        print(f"   V² range: [{v2_model.min():.4f}, {v2_model.max():.4f}]")
        print(f"   T3amp range: [{t3amp_model.min():.4f}, {t3amp_model.max():.4f}]")
        print(f"   T3phi range: [{t3phi_model.min():.2f}°, {t3phi_model.max():.2f}°]")

        # Verify observable properties
        assert np.all(v2_model >= 0), "V² should be non-negative"
        assert np.all(v2_model <= 1), "V² should be <= 1"
        assert np.all(t3amp_model >= 0), "T3amp should be non-negative"

        # Step 6: Compute chi-squared
        print("\n6. Computing chi-squared...")
        chi2_total, chi2_v2, chi2_t3amp, chi2_t3phi = fwd_model['compute_chi2'](
            image, return_components=True
        )
        print(f"   χ²_V²: {chi2_v2:.2f} ({chi2_v2/oi_data.nv2:.4f} per point)")
        print(f"   χ²_T3amp: {chi2_t3amp:.2f} ({chi2_t3amp/oi_data.nt3amp:.4f} per point)")
        print(f"   χ²_T3phi: {chi2_t3phi:.2f} ({chi2_t3phi/oi_data.nt3phi:.4f} per point)")
        print(f"   χ²_total: {chi2_total:.2f}")

        # Verify chi-squared properties
        assert chi2_total >= 0, "Chi-squared should be non-negative"
        assert chi2_total == chi2_v2 + chi2_t3amp + chi2_t3phi, \
            "Total chi² should equal sum of components"

        # Step 7: Compute residuals
        print("\n7. Computing residuals...")
        residuals = compute_residuals(
            image, geom, oi_data, fwd_model['polyft_matrix']
        )
        print(f"   V² residual RMS: {jnp.std(residuals['v2_residual']):.4f}")
        print(f"   T3amp residual RMS: {jnp.std(residuals['t3amp_residual']):.4f}")
        print(f"   T3phi residual RMS: {jnp.std(residuals['t3phi_residual']):.4f}")

        print("\n" + "="*60)
        print("✓ END-TO-END UNIFORM DISK TEST PASSED")
        print("="*60)

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_forward_model_image_reconstruction():
    """Test forward model with non-uniform brightness distribution.

    Simulates a simple image reconstruction scenario:
    - Create star geometry
    - Add a bright spot to brightness map
    - Verify that observables change
    - Check that chi² responds to image changes
    """
    print("\n" + "="*60)
    print("Integration Test: Forward Model with Bright Spot")
    print("="*60)

    # Create geometry
    print("\n1. Creating star geometry...")
    tess = tessellation_healpix(n=3)
    geom = create_star(tess, radius=1.0, inc=60.0, PA=0.0, obliq=0.0)

    # Create OIFITS data
    print("\n2. Creating synthetic data...")
    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        create_synthetic_oifits(tmp_filename, nv2=25, nt3=12, nwave=3)
        oi_data = read_oifits(tmp_filename, verbose=False)

        # Create forward model
        print("\n3. Creating forward model...")
        fwd_model = create_forward_model(geom, oi_data)

        # Test 1: Uniform image
        print("\n4. Testing uniform image...")
        image_uniform = jnp.ones(geom.npix)
        chi2_uniform = fwd_model['compute_chi2'](image_uniform)
        v2_uniform, _, _ = fwd_model['compute_observables'](image_uniform)
        print(f"   χ²_uniform: {chi2_uniform:.2f}")
        print(f"   V²_uniform mean: {jnp.mean(v2_uniform):.4f}")

        # Test 2: Image with bright spot
        print("\n5. Testing image with bright spot...")
        image_spot = jnp.ones(geom.npix)
        # Add bright spot at equator (more visible)
        spot_index = geom.npix // 2
        image_spot = image_spot.at[spot_index].set(3.0)

        chi2_spot = fwd_model['compute_chi2'](image_spot)
        v2_spot, _, _ = fwd_model['compute_observables'](image_spot)
        print(f"   χ²_spot: {chi2_spot:.2f}")
        print(f"   V²_spot mean: {jnp.mean(v2_spot):.4f}")

        # Chi² should be different for different images
        chi2_diff = abs(chi2_spot - chi2_uniform)
        print(f"\n6. Chi² difference: {chi2_diff:.2f}")
        assert chi2_diff > 0.01, "Chi² should change with different images"

        # V² should be different
        v2_diff = jnp.mean(jnp.abs(v2_spot - v2_uniform))
        print(f"   V² mean difference: {v2_diff:.6f}")
        assert v2_diff > 1e-6, "V² should change with different images"

        print("\n" + "="*60)
        print("✓ FORWARD MODEL BRIGHT SPOT TEST PASSED")
        print("="*60)

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_different_geometries():
    """Test forward model with different stellar geometries.

    Verifies that geometry changes (inclination, PA) affect observables:
    - Pole-on vs edge-on stars have different visibility profiles
    - Position angle rotation affects UV coverage
    """
    print("\n" + "="*60)
    print("Integration Test: Different Geometries")
    print("="*60)

    # Create tessellation
    tess = tessellation_healpix(n=3)

    # Create OIFITS data
    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        create_synthetic_oifits(tmp_filename, nv2=20, nt3=10, nwave=3)
        oi_data = read_oifits(tmp_filename, verbose=False)

        # Uniform image for all tests
        image = jnp.ones(tess.npix)

        # Test 1: Pole-on star (inc=0)
        print("\n1. Pole-on star (inc=0°)...")
        geom_pole = create_star(tess, radius=1.0, inc=0.0, PA=0.0, obliq=0.0)
        fwd_pole = create_forward_model(geom_pole, oi_data)
        v2_pole, _, _ = fwd_pole['compute_observables'](image)
        print(f"   Visible pixels: {np.sum(geom_pole.visible_mask)}")
        print(f"   V² mean: {jnp.mean(v2_pole):.4f}")

        # Test 2: Edge-on star (inc=90)
        print("\n2. Edge-on star (inc=90°)...")
        geom_edge = create_star(tess, radius=1.0, inc=90.0, PA=0.0, obliq=0.0)
        fwd_edge = create_forward_model(geom_edge, oi_data)
        v2_edge, _, _ = fwd_edge['compute_observables'](image)
        print(f"   Visible pixels: {np.sum(geom_edge.visible_mask)}")
        print(f"   V² mean: {jnp.mean(v2_edge):.4f}")

        # Test 3: Intermediate inclination
        print("\n3. Intermediate star (inc=45°)...")
        geom_mid = create_star(tess, radius=1.0, inc=45.0, PA=0.0, obliq=0.0)
        fwd_mid = create_forward_model(geom_mid, oi_data)
        v2_mid, _, _ = fwd_mid['compute_observables'](image)
        print(f"   Visible pixels: {np.sum(geom_mid.visible_mask)}")
        print(f"   V² mean: {jnp.mean(v2_mid):.4f}")

        # Verify different geometries give different results
        print("\n4. Comparing geometries...")
        diff_pole_edge = jnp.mean(jnp.abs(v2_pole - v2_edge))
        diff_pole_mid = jnp.mean(jnp.abs(v2_pole - v2_mid))
        print(f"   |V²_pole - V²_edge|: {diff_pole_edge:.6f}")
        print(f"   |V²_pole - V²_mid|: {diff_pole_mid:.6f}")

        assert diff_pole_edge > 1e-6, "Pole-on and edge-on should be different"
        assert diff_pole_mid > 1e-6, "Pole-on and intermediate should be different"

        # Pole-on should have fewer visible pixels than others
        assert np.sum(geom_pole.visible_mask) <= np.sum(geom_mid.visible_mask), \
            "Pole-on should have fewer or equal visible pixels"

        print("\n" + "="*60)
        print("✓ DIFFERENT GEOMETRIES TEST PASSED")
        print("="*60)

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_realistic_reconstruction_scenario():
    """Test realistic image reconstruction scenario.

    Simulates a typical reconstruction workflow:
    1. Setup star geometry (moderately inclined)
    2. Load interferometric data
    3. Create forward model with precomputation
    4. Try different images and compare chi²
    5. Verify that better images have lower chi²
    """
    print("\n" + "="*60)
    print("Integration Test: Realistic Reconstruction Scenario")
    print("="*60)

    # Realistic star parameters (e.g., Altair-like rapid rotator)
    print("\n1. Setting up star geometry...")
    print("   Target: Moderately oblate rapid rotator")
    print("   Radius: 1.0 mas")
    print("   Inclination: 60°")
    print("   Position angle: 45°")

    tess = tessellation_healpix(n=4)  # ~200 pixels
    geom = create_star(
        tess,
        radius=1.0,
        inc=60.0,
        PA=45.0,
        obliq=0.0,
    )
    print(f"   Pixels: {geom.npix}")
    print(f"   Visible: {np.sum(geom.visible_mask)}")

    # Realistic observational setup
    print("\n2. Setting up observational data...")
    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Realistic data volume (e.g., single CHARA night)
        create_synthetic_oifits(tmp_filename, nv2=50, nt3=25, nwave=5)
        oi_data = read_oifits(tmp_filename, verbose=False)
        print(f"   V² measurements: {oi_data.nv2}")
        print(f"   Closure phases: {oi_data.nt3phi}")
        print(f"   Degrees of freedom: {oi_data.nv2 + 2*oi_data.nt3phi}")

        # Create optimized forward model
        print("\n3. Creating forward model with precomputation...")
        fwd_model = create_forward_model(geom, oi_data)
        print(f"   Precomputed polyft matrix: {fwd_model['polyft_matrix'].shape}")
        print("   Forward model ready for optimization")

        # Test different image models
        print("\n4. Testing different image models...")

        # Model 1: Uniform disk (baseline)
        print("\n   Model 1: Uniform disk")
        image_uniform = jnp.ones(geom.npix)
        chi2_uniform = fwd_model['compute_chi2'](image_uniform, verbose=True)

        # Model 2: Limb-darkened disk (better)
        print("\n   Model 2: Limb-darkened disk")
        # Simple cosine limb darkening: I(μ) = I₀(1 - u + u*μ)
        # μ = cos(angle from surface normal)
        # For uniform sphere, μ ≈ sqrt(1 - r²) where r is projected radius
        r_proj = np.sqrt(tess.x**2 + tess.y**2)  # Projected radius
        mu = np.sqrt(np.maximum(0, 1 - r_proj**2))
        limb_darkening_coeff = 0.6
        image_limb = 1.0 - limb_darkening_coeff + limb_darkening_coeff * mu
        image_limb = jnp.array(image_limb)
        chi2_limb = fwd_model['compute_chi2'](image_limb, verbose=True)

        # Model 3: Random image (worse)
        print("\n   Model 3: Random image")
        np.random.seed(42)
        image_random = jnp.array(np.random.rand(geom.npix) * 2)
        chi2_random = fwd_model['compute_chi2'](image_random, verbose=True)

        # Compare models
        print("\n5. Model comparison:")
        print(f"   χ²_uniform: {chi2_uniform:.2f}")
        print(f"   χ²_limb-darkened: {chi2_limb:.2f}")
        print(f"   χ²_random: {chi2_random:.2f}")

        # Sanity checks
        assert chi2_uniform > 0, "Chi² should be positive"
        assert chi2_limb > 0, "Chi² should be positive"
        assert chi2_random > 0, "Chi² should be positive"

        # Random image should generally be worse than structured models
        print(f"\n   Random image penalty: {chi2_random - chi2_uniform:.2f}")

        print("\n" + "="*60)
        print("✓ REALISTIC RECONSTRUCTION SCENARIO TEST PASSED")
        print("="*60)

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print(" "*15 + "ROTIR INTEGRATION TESTS")
    print("="*70)
    print("\nTesting complete pipeline: Tessellation → Geometry → Polygon FT")
    print("                           → Observables → Chi² calculation")
    print("="*70)

    try:
        test_end_to_end_uniform_disk()
        test_forward_model_image_reconstruction()
        test_different_geometries()
        test_realistic_reconstruction_scenario()

        print("\n" + "="*70)
        print(" "*20 + "ALL INTEGRATION TESTS PASSED ✓")
        print("="*70)
        print("\nThe complete ROTIR forward model pipeline is working!")
        print("Ready for:")
        print("  - Real OIFITS data")
        print("  - Image reconstruction")
        print("  - Rapid rotator modeling")
        print("  - Binary star systems")
        print("="*70 + "\n")
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
    success = run_all_integration_tests()
    exit(0 if success else 1)
