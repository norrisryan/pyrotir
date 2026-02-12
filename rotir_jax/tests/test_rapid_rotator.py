"""Tests for rapid rotator geometry module.

Tests verify:
1. Roche model radius function
2. Oblate spheroid approximation
3. Angular velocity calculations
4. Von Zeipel gravity darkening
5. Temperature-to-intensity conversion
6. Complete rapid rotator star creation
"""

import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('..')

from rotir_jax.tessellation.healpix import tessellation_healpix
from rotir_jax.geometry.base import create_star
from rotir_jax.geometry.rapid_rotator import (
    f_rapid_rot,
    oblate_const,
    calc_omega,
    calc_rotspin,
    temperature_map_vonZeipel,
    compute_intensity_from_temperature,
    create_rapid_rotator_star,
    compute_teff_vonzeipel,
)


def test_roche_radius_function():
    """Test Roche model radius function."""
    print("Testing Roche radius function...")

    # Test at poles (x→0): r/r_pole → 1
    x_pole = jnp.array([1e-12, 1e-10, 1e-8])
    r_ratio_pole = f_rapid_rot(x_pole)
    print(f"  At poles (x→0): r/r_pole = {r_ratio_pole}")
    assert jnp.allclose(r_ratio_pole, 1.0, atol=1e-6), \
        "Radius at poles should be r_pole"

    # Test at moderate rotation
    x_mid = jnp.array([0.3, 0.5, 0.7])
    r_ratio_mid = f_rapid_rot(x_mid)
    print(f"  At x={x_mid}: r/r_pole = {r_ratio_mid}")
    assert jnp.all(r_ratio_mid > 1.0), \
        "Radius should increase from poles due to centrifugal force"

    # Test near critical rotation (x→1)
    x_crit = jnp.array([0.9, 0.95, 0.99])
    r_ratio_crit = f_rapid_rot(x_crit)
    print(f"  At x={x_crit}: r/r_pole = {r_ratio_crit}")
    assert jnp.all(r_ratio_crit > r_ratio_mid.max()), \
        "Radius should be largest near critical rotation"

    print("  ✓ Roche radius function test passed")


def test_oblate_approximation():
    """Test oblate spheroid approximation."""
    print("\nTesting oblate spheroid approximation...")

    rpole = 1.0  # mas

    # Test slow rotation
    omega_slow = 0.1
    a, b, c = oblate_const(rpole, omega_slow)
    print(f"  Slow rotation (ω={omega_slow}): a={a:.4f}, b={b:.4f}, c={c:.4f}")
    assert c == rpole, "Polar radius should be unchanged"
    assert a > rpole, "Equatorial radius should be larger"
    assert b == a, "Should be axisymmetric"
    assert (a - rpole) / rpole < 0.2, "Small oblateness for slow rotation"

    # Test moderate rotation (Altair-like)
    omega_altair = 0.9
    a, b, c = oblate_const(rpole, omega_altair)
    oblateness = (a - c) / c
    print(f"  Altair-like (ω={omega_altair}): oblateness = {oblateness:.3f}")
    print(f"    R_eq/R_pole = {a/c:.3f}")
    assert 1.2 < a/c < 1.4, "Altair has ~25% oblateness"

    # Test zero rotation
    omega_zero = 0.0
    a, b, c = oblate_const(rpole, omega_zero)
    print(f"  No rotation (ω={omega_zero}): a={a:.4f}, b={b:.4f}, c={c:.4f}")
    assert a == b == c == rpole, "Should be spherical"

    print("  ✓ Oblate approximation test passed")


def test_omega_calculation():
    """Test angular velocity from oblateness."""
    print("\nTesting omega calculation...")

    rpole = 1.0

    # Test round-trip: oblateness → omega → oblateness
    test_oblates = [0.1, 0.2, 0.3]
    for oblate_in in test_oblates:
        omega, _, req = calc_omega(rpole, oblate_in)
        oblate_out = (req - rpole) / rpole

        print(f"  Input oblateness: {oblate_in:.3f}")
        print(f"  Computed omega: {omega:.4f}")
        print(f"  Output oblateness: {oblate_out:.3f}")

        assert abs(oblate_out - oblate_in) < 1e-6, \
            f"Round-trip failed: {oblate_in} → {oblate_out}"

    print("  ✓ Omega calculation test passed")


def test_rotation_period():
    """Test rotation period calculation."""
    print("\nTesting rotation period calculation...")

    # Altair parameters
    rpole_altair = 1.63  # R_sun
    req_altair = 2.03    # R_sun
    omega_c = 0.9
    mass_altair = 1.79   # M_sun

    rot_vel, period = calc_rotspin(rpole_altair, req_altair, omega_c, mass_altair)

    print(f"  Altair:")
    print(f"    Rotation period: {period:.2f} days")
    print(f"    Rotation rate: {rot_vel:.4f} rot/day")

    # Altair's observed period is ~8-10 hours ≈ 0.35-0.42 days
    assert 0.3 < period < 0.5, \
        f"Altair period should be ~0.4 days, got {period:.2f}"

    # Sun (slow rotator)
    rpole_sun = 1.0
    req_sun = 1.0  # Essentially spherical
    omega_sun = 0.01  # Very slow
    mass_sun = 1.0

    rot_vel_sun, period_sun = calc_rotspin(rpole_sun, req_sun, omega_sun, mass_sun)

    print(f"\n  Sun (slow rotator):")
    print(f"    Rotation period: {period_sun:.1f} days")

    # Sun's period is ~25-35 days at equator
    # Our simplified model won't match exactly, but should be >>1 day
    assert period_sun > 10, "Sun should rotate slowly"

    print("  ✓ Rotation period test passed")


def test_vonzeipel_temperature_map():
    """Test von Zeipel gravity darkening."""
    print("\nTesting von Zeipel temperature map...")

    # Create tessellation and geometry
    tess = tessellation_healpix(n=3)
    geom = create_star(tess, radius=1.0, inc=0.0, PA=0.0, obliq=0.0)

    # Altair-like parameters
    tpole = 8450.0  # K
    rpole = 1.0
    omega = 0.9
    beta = 0.22  # Altair's measured value

    # Compute temperature map
    temp_map = temperature_map_vonZeipel(
        tess, geom, tpole, rpole, omega, beta
    )

    print(f"  Temperature range: [{temp_map.min():.0f}, {temp_map.max():.0f}] K")
    print(f"  T_pole: {temp_map.max():.0f} K")
    print(f"  T_min (near equator): {temp_map.min():.0f} K")
    print(f"  ΔT: {temp_map.max() - temp_map.min():.0f} K")

    # Sanity checks
    assert temp_map.max() <= tpole * 1.01, \
        "Maximum temperature should be at pole"
    assert temp_map.min() < tpole * 0.85, \
        "Equator should be significantly cooler"

    # For Altair with β=0.22 and ω=0.9:
    # T_eq/T_pole ≈ 0.81-0.82 (observed)
    temp_ratio = temp_map.min() / temp_map.max()
    print(f"  T_eq/T_pole: {temp_ratio:.3f}")
    assert 0.75 < temp_ratio < 0.90, \
        "Temperature ratio should match observations"

    # Test β dependence
    print("\n  Testing gravity darkening exponent β:")
    for beta_test in [0.08, 0.25, 0.40]:
        temp_test = temperature_map_vonZeipel(
            tess, geom, tpole, rpole, omega, beta_test
        )
        ratio_test = temp_test.min() / temp_test.max()
        print(f"    β={beta_test:.2f}: T_eq/T_pole = {ratio_test:.3f}")

    # Larger β → stronger gravity darkening → larger T contrast
    temp_beta_low = temperature_map_vonZeipel(tess, geom, tpole, rpole, omega, 0.08)
    temp_beta_high = temperature_map_vonZeipel(tess, geom, tpole, rpole, omega, 0.40)

    ratio_low = temp_beta_low.min() / temp_beta_low.max()
    ratio_high = temp_beta_high.min() / temp_beta_high.max()

    assert ratio_high < ratio_low, \
        "Higher β should give stronger temperature contrast"

    print("  ✓ von Zeipel temperature map test passed")


def test_temperature_to_intensity():
    """Test temperature to intensity conversion."""
    print("\nTesting temperature to intensity conversion...")

    # Test blackbody at different temperatures
    temps = jnp.array([8000., 7000., 6000., 5000.])  # K
    wavelength = 1.65e-6  # H-band

    intensities = compute_intensity_from_temperature(temps, wavelength)

    print(f"  Temperatures: {temps} K")
    print(f"  Relative intensities: {intensities / intensities[0]}")

    # Hotter should be brighter
    assert jnp.all(jnp.diff(intensities) < 0), \
        "Intensity should decrease with decreasing temperature"

    # Test wavelength dependence
    wavelengths = jnp.array([1.25e-6, 1.65e-6, 2.2e-6])  # J, H, K bands
    for wl in wavelengths:
        intensity = compute_intensity_from_temperature(temps[0], wl)
        print(f"  λ = {wl*1e6:.2f} μm: I = {intensity:.4e}")

    print("  ✓ Temperature to intensity conversion test passed")


def test_create_rapid_rotator_star():
    """Test complete rapid rotator star creation."""
    print("\nTesting rapid rotator star creation...")

    # Create Altair model
    tess = tessellation_healpix(n=4)

    print("  Creating Altair model:")
    print("    R_pole = 1.63 R_sun")
    print("    ω/ω_crit = 0.9")
    print("    T_pole = 8450 K")
    print("    β = 0.22")
    print("    inc = 60°")

    geom, intensity = create_rapid_rotator_star(
        tess,
        rpole=1.63,
        frac_escapevel=0.9,
        tpole=8450.0,
        beta=0.22,
        inc=60.0,
        PA=45.0,
        wavelength=1.65e-6,
    )

    print(f"  Geometry: {geom.npix} pixels, {np.sum(geom.visible_mask)} visible")
    print(f"  Intensity range: [{intensity.min():.4e}, {intensity.max():.4e}]")
    print(f"  I_max/I_min: {intensity.max() / intensity.min():.2f}")

    # Sanity checks
    assert intensity.shape == (geom.npix,), "Intensity shape mismatch"
    assert jnp.all(intensity > 0), "Intensity should be positive"
    assert intensity.max() / intensity.min() > 1.5, \
        "Should have significant intensity variation"

    # Test different rotation rates
    print("\n  Testing different rotation rates:")
    for omega in [0.3, 0.6, 0.9]:
        _, intensity_omega = create_rapid_rotator_star(
            tess, rpole=1.0, frac_escapevel=omega,
            tpole=8000.0, beta=0.25, inc=60.0
        )
        contrast = intensity_omega.max() / intensity_omega.min()
        print(f"    ω={omega:.1f}: I_max/I_min = {contrast:.2f}")

    print("  ✓ Rapid rotator star creation test passed")


def test_simplified_vonzeipel():
    """Test simplified von Zeipel for oblate spheroid."""
    print("\nTesting simplified von Zeipel...")

    rpole = 1.0
    req = 1.25  # 25% oblate
    tpole = 8000.0
    beta = 0.25

    # Test at different latitudes
    colatitudes = jnp.array([0.0, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4, jnp.pi])
    latitudes_deg = 90 - np.degrees(colatitudes)

    temps = compute_teff_vonzeipel(rpole, req, tpole, beta, colatitudes)

    print("  Latitude (deg) | Temperature (K)")
    print("  " + "-"*35)
    for lat, temp in zip(latitudes_deg, temps):
        print(f"  {lat:14.1f} | {temp:15.0f}")

    # Pole should be hottest
    assert temps[0] == temps.max(), "Pole should be hottest"
    # Equator should be coolest
    assert temps[2] == temps.min(), "Equator should be coolest"

    print("  ✓ Simplified von Zeipel test passed")


def test_altair_realistic_model():
    """Test realistic Altair model."""
    print("\nTesting realistic Altair model...")

    print("  Altair observed properties:")
    print("    Spectral type: A7V")
    print("    T_eff: 6900-8500 K (pole-to-equator)")
    print("    v sin i: ~240 km/s")
    print("    Period: ~10 hours")
    print("    R_eq/R_pole: ~1.25")
    print("    Inclination: ~60°")

    # Create model
    tess = tessellation_healpix(n=5)
    geom, intensity = create_rapid_rotator_star(
        tess,
        rpole=1.63,  # R_sun
        frac_escapevel=0.9,
        tpole=8450.0,  # K
        beta=0.22,  # Measured from interferometry
        inc=60.0,  # degrees
        PA=45.0,
        wavelength=1.65e-6,  # H-band
    )

    print(f"\n  Model results:")
    print(f"    Pixels: {geom.npix}")
    print(f"    Visible pixels: {np.sum(geom.visible_mask)}")
    print(f"    Intensity contrast: {intensity.max()/intensity.min():.2f}")

    # Verify model is reasonable
    assert geom.npix > 100, "Should have sufficient resolution"
    assert np.sum(geom.visible_mask) > 50, "Should have many visible pixels"
    assert 1.5 < intensity.max()/intensity.min() < 3.0, \
        "Intensity contrast should be moderate"

    print("  ✓ Altair realistic model test passed")


def run_all_tests():
    """Run all rapid rotator tests."""
    print("="*60)
    print("Running Rapid Rotator Geometry Tests")
    print("="*60)

    try:
        test_roche_radius_function()
        test_oblate_approximation()
        test_omega_calculation()
        test_rotation_period()
        test_vonzeipel_temperature_map()
        test_temperature_to_intensity()
        test_create_rapid_rotator_star()
        test_simplified_vonzeipel()
        test_altair_realistic_model()

        print("\n" + "="*60)
        print("ALL RAPID ROTATOR TESTS PASSED ✓")
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
