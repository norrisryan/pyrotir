"""Tests for binary star orbital mechanics and Roche geometry.

Tests verify:
1. Kepler's equation solver
2. Orbital position calculations
3. Roche lobe approximations
4. Roche potential calculations
5. Binary system configurations
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')

from rotir_jax.tessellation.healpix import tessellation_healpix
from rotir_jax.geometry.orbits import (
    compute_E_NR,
    compute_eccentric_anomaly,
    compute_true_anomaly,
    compute_orbital_coefficients,
    compute_separation,
    binary_orbit_relative,
    binary_orbit_absolute,
    compute_orbital_phase,
    compute_radial_velocity,
)
from rotir_jax.geometry.roche import (
    eggleton_roche_radius,
    roche_radius_pathania,
    compute_potential_primary,
    compute_potential_secondary,
    compute_L1_distance,
    is_contact_binary,
)


def test_kepler_equation():
    """Test Kepler's equation solver."""
    print("Testing Kepler's equation solver...")

    # Test circular orbit (e=0): E = M
    M = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
    E = compute_E_NR(M, e=0.0)
    print(f"  Circular orbit (e=0): E ≈ M")
    assert jnp.allclose(E, M, atol=1e-6), "For circular orbit, E should equal M"

    # Test eccentric orbit
    e = 0.5
    M_test = jnp.array([0.0, jnp.pi/4, jnp.pi/2, jnp.pi])
    E_test = compute_E_NR(M_test, e=e)

    # Verify Kepler's equation: M = E - e*sin(E)
    M_check = E_test - e * jnp.sin(E_test)
    print(f"  Eccentric orbit (e={e}): max error = {jnp.max(jnp.abs(M_test - M_check)):.2e}")
    assert jnp.allclose(M_test, M_check, atol=1e-6), \
        "Kepler's equation not satisfied"

    # Test high eccentricity
    e_high = 0.9
    E_high = compute_E_NR(M_test, e=e_high)
    M_high_check = E_high - e_high * jnp.sin(E_high)
    print(f"  High eccentricity (e={e_high}): max error = {jnp.max(jnp.abs(M_test - M_high_check)):.2e}")
    assert jnp.allclose(M_test, M_high_check, atol=1e-6), \
        "High eccentricity case failed"

    print("  ✓ Kepler's equation solver test passed")


def test_orbital_anomalies():
    """Test eccentric and true anomaly calculations."""
    print("\nTesting orbital anomalies...")

    P = 365.25  # 1 year
    e = 0.1
    T0 = 0.0

    # Test at periastron (t = T0)
    tepoch = jnp.array([T0])
    E = compute_eccentric_anomaly(P, e, T0, tepoch)
    υ = compute_true_anomaly(P, e, T0, tepoch)

    print(f"  At periastron:")
    print(f"    E = {E[0]:.6f} rad (expected ≈ 0)")
    print(f"    υ = {υ[0]:.6f} rad (expected ≈ 0)")

    assert jnp.abs(E[0]) < 1e-6, "Eccentric anomaly should be ~0 at periastron"
    assert jnp.abs(υ[0]) < 1e-6, "True anomaly should be ~0 at periastron"

    # Test at apastron (t = T0 + P/2)
    tepoch_apa = jnp.array([T0 + P / 2])
    E_apa = compute_eccentric_anomaly(P, e, T0, tepoch_apa)
    υ_apa = compute_true_anomaly(P, e, T0, tepoch_apa)

    print(f"  At apastron:")
    print(f"    E = {E_apa[0]:.6f} rad (expected ≈ π = {jnp.pi:.6f})")
    print(f"    υ = {υ_apa[0]:.6f} rad (expected ≈ π)")

    assert jnp.abs(E_apa[0] - jnp.pi) < 1e-3, "E should be ~π at apastron"
    assert jnp.abs(υ_apa[0] - jnp.pi) < 1e-3, "υ should be ~π at apastron"

    # Test full orbit
    tepochs = jnp.linspace(0, P, 100)
    υ_orbit = compute_true_anomaly(P, e, T0, tepochs)

    print(f"  Full orbit: υ range = [{υ_orbit.min():.3f}, {υ_orbit.max():.3f}]")
    assert υ_orbit.min() >= 0, "True anomaly should be non-negative"
    assert υ_orbit.max() <= 2*jnp.pi, "True anomaly should be < 2π"

    print("  ✓ Orbital anomalies test passed")


def test_binary_separation():
    """Test binary separation calculations."""
    print("\nTesting binary separation...")

    a = 100.0  # mas
    e = 0.3
    P = 10.0   # days
    T0 = 0.0

    # At periastron
    D_peri = compute_separation(a, e, P, T0, jnp.array([T0]))[0]
    D_peri_expected = a * (1 - e)

    print(f"  Periastron:")
    print(f"    D = {D_peri:.2f} mas")
    print(f"    Expected = {D_peri_expected:.2f} mas (= a(1-e))")

    assert jnp.abs(D_peri - D_peri_expected) < 0.01, \
        "Periastron separation wrong"

    # At apastron
    D_apa = compute_separation(a, e, P, T0, jnp.array([T0 + P/2]))[0]
    D_apa_expected = a * (1 + e)

    print(f"  Apastron:")
    print(f"    D = {D_apa:.2f} mas")
    print(f"    Expected = {D_apa_expected:.2f} mas (= a(1+e))")

    assert jnp.abs(D_apa - D_apa_expected) < 0.01, \
        "Apastron separation wrong"

    print("  ✓ Binary separation test passed")


def test_binary_orbit_relative():
    """Test relative binary orbit."""
    print("\nTesting relative binary orbit...")

    # Simple face-on orbit
    a = 100.0  # mas
    e = 0.0    # circular
    P = 10.0   # days
    T0 = 0.0
    Omega = 0.0   # deg
    i = 0.0       # face-on
    omega = 0.0   # deg

    # At phase 0.0 (zero longitude)
    x1, y1, z1, x2, y2, z2 = binary_orbit_relative(
        a, e, P, T0, Omega, i, omega, T0
    )

    print(f"  Face-on circular orbit at t=T0:")
    print(f"    Primary: ({x1:.2f}, {y1:.2f}, {z1:.2f}) mas")
    print(f"    Secondary: ({x2:.2f}, {y2:.2f}, {z2:.2f}) mas")

    # Primary should be at origin
    assert x1 == 0 and y1 == 0 and z1 == 0, "Primary should be at origin"

    # Secondary should be at distance a
    r2 = jnp.sqrt(x2**2 + y2**2 + z2**2)
    print(f"    Secondary distance: {r2:.2f} mas (expected {a:.2f})")
    assert jnp.abs(r2 - a) < 0.01, "Secondary distance wrong"

    # For face-on, z should be zero
    assert jnp.abs(z2) < 1e-10, "Face-on orbit should have z=0"

    print("  ✓ Relative binary orbit test passed")


def test_binary_orbit_absolute():
    """Test absolute binary orbit (center of mass)."""
    print("\nTesting absolute binary orbit...")

    # Equal mass binary
    a = 100.0
    e = 0.0
    P = 10.0
    T0 = 0.0
    q = 1.0  # Equal masses
    Omega = 0.0
    i = 0.0  # Face-on
    omega = 0.0

    x1, y1, z1, x2, y2, z2 = binary_orbit_absolute(
        a, e, P, T0, q, Omega, i, omega, T0
    )

    print(f"  Equal mass binary:")
    print(f"    Primary: ({x1:.2f}, {y1:.2f}, {z1:.2f}) mas")
    print(f"    Secondary: ({x2:.2f}, {y2:.2f}, {z2:.2f}) mas")

    # For equal masses, should be equidistant from origin
    r1 = jnp.sqrt(x1**2 + y1**2 + z1**2)
    r2 = jnp.sqrt(x2**2 + y2**2 + z2**2)

    print(f"    r1 = {r1:.2f}, r2 = {r2:.2f} mas")
    assert jnp.abs(r1 - r2) < 0.01, "Equal masses should have equal distances"

    # Sum should equal separation
    total_sep = r1 + r2
    print(f"    Total separation: {total_sep:.2f} mas (expected {a:.2f})")
    assert jnp.abs(total_sep - a) < 0.1, "Total separation wrong"

    # Center of mass at origin: M1*r1 + M2*r2 = 0
    # For equal masses: r1 + r2 = 0
    com_x = (x1 + x2)  # q=1 means equal masses
    com_y = (y1 + y2)
    com_z = (z1 + z2)

    print(f"    Center of mass: ({com_x:.6f}, {com_y:.6f}, {com_z:.6f})")
    assert jnp.abs(com_x) < 1e-6, "COM should be at origin"
    assert jnp.abs(com_y) < 1e-6, "COM should be at origin"

    print("  ✓ Absolute binary orbit test passed")


def test_orbital_phase():
    """Test orbital phase calculation."""
    print("\nTesting orbital phase...")

    P = 10.0
    T0 = 0.0

    # Test full orbit
    tepochs = jnp.array([0, 2.5, 5.0, 7.5, 10.0, 12.5])
    phases = compute_orbital_phase(P, T0, tepochs)

    expected_phases = jnp.array([0.0, 0.25, 0.5, 0.75, 0.0, 0.25])

    print(f"  Epochs: {tepochs}")
    print(f"  Phases: {phases}")
    print(f"  Expected: {expected_phases}")

    assert jnp.allclose(phases, expected_phases, atol=1e-6), "Phases wrong"

    print("  ✓ Orbital phase test passed")


def test_radial_velocity():
    """Test radial velocity calculation."""
    print("\nTesting radial velocity...")

    K = 50.0  # km/s
    e = 0.1
    P = 10.0  # days
    T0 = 0.0
    omega = 0.0  # deg
    gamma = 10.0  # km/s systemic velocity

    # At periastron
    RV_peri = compute_radial_velocity(K, e, P, T0, omega, jnp.array([T0]), gamma)[0]

    print(f"  At periastron:")
    print(f"    RV = {RV_peri:.2f} km/s")
    print(f"    Expected ≈ {gamma + K * (1 + e):.2f} km/s")

    # Should be near maximum (approaching)
    assert RV_peri > gamma, "RV should be > systemic at periastron"

    # Full orbit
    tepochs = jnp.linspace(0, P, 100)
    RVs = compute_radial_velocity(K, e, P, T0, omega, tepochs, gamma)

    print(f"  Full orbit RV range: [{RVs.min():.2f}, {RVs.max():.2f}] km/s")
    print(f"  Amplitude: {(RVs.max() - RVs.min())/2:.2f} km/s (expected ~{K:.2f})")

    # Check amplitude
    amplitude = (RVs.max() - RVs.min()) / 2
    assert jnp.abs(amplitude - K) < 5.0, "RV amplitude should be ~K"

    print("  ✓ Radial velocity test passed")


def test_eggleton_roche_radius():
    """Test Eggleton Roche lobe approximation."""
    print("\nTesting Eggleton Roche radius...")

    # Equal mass case (q=1)
    q = 1.0
    R_L = eggleton_roche_radius(q)

    print(f"  Equal masses (q=1): R_L/a = {R_L:.4f}")
    # For equal masses, R_L ≈ 0.38-0.39
    assert 0.35 < R_L < 0.45, "Equal mass Roche radius out of range"

    # Small mass ratio (q=0.1)
    q_small = 0.1
    R_L_small = eggleton_roche_radius(q_small)

    print(f"  Small mass (q=0.1): R_L/a = {R_L_small:.4f}")
    # For small q, R_L ≈ 0.46*q^(2/3)
    assert R_L_small < R_L, "Small mass should have smaller Roche lobe"

    # Large mass ratio (q=10)
    q_large = 10.0
    R_L_large = eggleton_roche_radius(q_large)

    print(f"  Large mass (q=10): R_L/a = {R_L_large:.4f}")
    assert R_L_large > R_L, "Large mass should have larger Roche lobe"

    # Test range of mass ratios
    q_range = jnp.array([0.1, 0.5, 1.0, 2.0, 10.0])
    for q_test in q_range:
        R_L_test = eggleton_roche_radius(q_test)
        print(f"    q={q_test:.1f}: R_L/a = {R_L_test:.4f}")

    print("  ✓ Eggleton Roche radius test passed")


def test_roche_potential():
    """Test Roche potential calculation."""
    print("\nTesting Roche potential...")

    # Test at pole (theta=0)
    r = 0.3  # Dimensionless
    D = 1.0  # Separation
    q = 0.5  # Mass ratio
    async_ratio = 1.0  # Synchronous

    Ω_pole, dΩ_pole, ddΩ_pole = compute_potential_primary(
        r, D, 0.0, 0.0, q, async_ratio
    )

    print(f"  At pole (r={r}, q={q}):")
    print(f"    Ω = {Ω_pole:.6f}")
    print(f"    dΩ/dr = {dΩ_pole:.6f}")
    print(f"    d²Ω/dr² = {ddΩ_pole:.6f}")

    # Potential should be real and finite
    assert jnp.isfinite(Ω_pole), "Potential should be finite"
    assert jnp.isfinite(dΩ_pole), "Derivative should be finite"

    # Test at equator (theta=π/2)
    Ω_eq, dΩ_eq, ddΩ_eq = compute_potential_primary(
        r, D, jnp.pi/2, 0.0, q, async_ratio
    )

    print(f"  At equator:")
    print(f"    Ω = {Ω_eq:.6f}")

    # Equator should have different potential than pole (centrifugal)
    assert jnp.abs(Ω_eq - Ω_pole) > 1e-6, \
        "Equator and pole should have different potentials"

    print("  ✓ Roche potential test passed")


def test_L1_distance():
    """Test L1 Lagrange point distance."""
    print("\nTesting L1 distance...")

    # Equal masses: L1 at midpoint
    q = 1.0
    r_L1 = compute_L1_distance(q)

    print(f"  Equal masses (q=1): r_L1/a = {r_L1:.4f}")
    # Should be near 0.5 for equal masses
    assert 0.3 < r_L1 < 0.5, "L1 distance for equal masses wrong"

    # Small companion
    q_small = 0.1
    r_L1_small = compute_L1_distance(q_small)

    print(f"  Small companion (q=0.1): r_L1/a = {r_L1_small:.4f}")
    # Should be near primary (r_L1 → 1)
    assert r_L1_small > r_L1, "L1 should be closer to small companion"

    print("  ✓ L1 distance test passed")


def test_contact_binary():
    """Test contact binary detection."""
    print("\nTesting contact binary detection...")

    a = 100.0  # mas
    q = 0.5

    # Detached: both stars small
    rpole1 = 20.0
    rpole2 = 15.0

    contact = is_contact_binary(rpole1, rpole2, a, q)
    print(f"  Detached (R1={rpole1}, R2={rpole2}, a={a}):")
    print(f"    Contact: {contact}")
    assert not contact, "Small stars should not be in contact"

    # Semi-detached: primary fills Roche lobe
    R_L1 = eggleton_roche_radius(q) * a
    rpole1_fill = R_L1 * 1.0  # Fills Roche lobe
    rpole2_small = 10.0

    contact_semi = is_contact_binary(rpole1_fill, rpole2_small, a, q)
    print(f"  Semi-detached (R1={rpole1_fill:.1f}, R2={rpole2_small}):")
    print(f"    Contact: {contact_semi}")
    # Note: is_contact_binary requires BOTH to fill for contact

    # Contact: both fill
    R_L2 = eggleton_roche_radius(1/q) * a
    rpole2_fill = R_L2 * 1.0

    contact_both = is_contact_binary(rpole1_fill, rpole2_fill, a, q)
    print(f"  Contact (both fill Roche lobes):")
    print(f"    Contact: {contact_both}")
    assert contact_both, "Both filling should be contact"

    print("  ✓ Contact binary detection test passed")


def run_all_tests():
    """Run all binary star tests."""
    print("="*60)
    print("Running Binary Star Tests (Orbits + Roche)")
    print("="*60)

    try:
        # Orbital mechanics
        test_kepler_equation()
        test_orbital_anomalies()
        test_binary_separation()
        test_binary_orbit_relative()
        test_binary_orbit_absolute()
        test_orbital_phase()
        test_radial_velocity()

        # Roche geometry
        test_eggleton_roche_radius()
        test_roche_potential()
        test_L1_distance()
        test_contact_binary()

        print("\n" + "="*60)
        print("ALL BINARY STAR TESTS PASSED ✓")
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
