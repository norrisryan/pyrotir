"""Tests for OIFITS reader module.

Tests verify:
1. Reading synthetic OIFITS files
2. Wavelength filtering
3. Multi-epoch reading
4. Data structure integrity
"""

import numpy as np
from astropy.io import fits
import tempfile
import os
import sys
sys.path.append('..')

from rotir_jax.io import read_oifits, read_oifits_multiepoch, summarize_oifits


def create_synthetic_oifits(filename, nv2=10, nt3=5, nwave=3):
    """Create a synthetic OIFITS file for testing.

    This creates a minimal OIFITS v2 file with:
    - OI_WAVELENGTH table
    - OI_VIS2 table (squared visibilities)
    - OI_T3 table (closure phases)
    """
    # Create primary HDU
    primary_hdu = fits.PrimaryHDU()

    # OI_WAVELENGTH table
    wl_data = np.array([(1.65e-6, 0.05e-6)] * nwave,
                       dtype=[('EFF_WAVE', 'f8'), ('EFF_BAND', 'f8')])
    wl_hdu = fits.BinTableHDU(wl_data, name='OI_WAVELENGTH')
    wl_hdu.header['INSNAME'] = 'TEST_INSTRUMENT'

    # OI_VIS2 table
    v2_dtype = [
        ('TARGET_ID', 'i4'),
        ('TIME', 'f8'),
        ('MJD', 'f8'),
        ('INT_TIME', 'f8'),
        ('VIS2DATA', 'f8', (nwave,)),
        ('VIS2ERR', 'f8', (nwave,)),
        ('UCOORD', 'f8'),
        ('VCOORD', 'f8'),
        ('STA_INDEX', 'i4', (2,)),
        ('FLAG', 'bool', (nwave,)),
    ]

    v2_data = []
    for i in range(nv2):
        # Random UV coordinates (in meters, ~50m baseline at 1.65 micron)
        u = np.random.randn() * 50
        v = np.random.randn() * 50

        # Synthetic V² data (uniform disk-like)
        baseline = np.sqrt(u**2 + v**2)
        wavelength = 1.65e-6
        spatial_freq = baseline / wavelength  # cycles/radian
        # Uniform disk: V² = [2*J1(x)/x]²
        # For simplicity, just use exponential decay
        v2_val = 0.5 * np.exp(-spatial_freq / 1e8)

        v2_row = (
            1,  # TARGET_ID
            0.0,  # TIME
            57000.0,  # MJD
            100.0,  # INT_TIME
            np.ones(nwave) * v2_val,  # VIS2DATA
            np.ones(nwave) * 0.05,  # VIS2ERR
            u,  # UCOORD
            v,  # VCOORD
            [1, 2],  # STA_INDEX
            np.zeros(nwave, dtype=bool),  # FLAG (False = good)
        )
        v2_data.append(v2_row)

    v2_array = np.array(v2_data, dtype=v2_dtype)
    v2_hdu = fits.BinTableHDU(v2_array, name='OI_VIS2')
    v2_hdu.header['INSNAME'] = 'TEST_INSTRUMENT'
    v2_hdu.header['ARRNAME'] = 'TEST_ARRAY'

    # OI_T3 table
    t3_dtype = [
        ('TARGET_ID', 'i4'),
        ('TIME', 'f8'),
        ('MJD', 'f8'),
        ('INT_TIME', 'f8'),
        ('T3AMP', 'f8', (nwave,)),
        ('T3AMPERR', 'f8', (nwave,)),
        ('T3PHI', 'f8', (nwave,)),
        ('T3PHIERR', 'f8', (nwave,)),
        ('U1COORD', 'f8'),
        ('V1COORD', 'f8'),
        ('U2COORD', 'f8'),
        ('V2COORD', 'f8'),
        ('STA_INDEX', 'i4', (3,)),
        ('FLAG', 'bool', (nwave,)),
    ]

    t3_data = []
    for i in range(nt3):
        # Random UV coordinates for 3 baselines
        u1 = np.random.randn() * 40
        v1 = np.random.randn() * 40
        u2 = np.random.randn() * 40
        v2 = np.random.randn() * 40

        # Synthetic T3 data
        t3amp = 0.1 + 0.1 * np.random.rand()
        t3phi = np.random.randn() * 10  # degrees

        t3_row = (
            1,  # TARGET_ID
            0.0,  # TIME
            57000.0,  # MJD
            100.0,  # INT_TIME
            np.ones(nwave) * t3amp,  # T3AMP
            np.ones(nwave) * 0.02,  # T3AMPERR
            np.ones(nwave) * t3phi,  # T3PHI
            np.ones(nwave) * 5.0,  # T3PHIERR
            u1,  # U1COORD
            v1,  # V1COORD
            u2,  # U2COORD
            v2,  # V2COORD
            [1, 2, 3],  # STA_INDEX
            np.zeros(nwave, dtype=bool),  # FLAG
        )
        t3_data.append(t3_row)

    t3_array = np.array(t3_data, dtype=t3_dtype)
    t3_hdu = fits.BinTableHDU(t3_array, name='OI_T3')
    t3_hdu.header['INSNAME'] = 'TEST_INSTRUMENT'
    t3_hdu.header['ARRNAME'] = 'TEST_ARRAY'

    # Create HDU list and write
    hdul = fits.HDUList([primary_hdu, wl_hdu, v2_hdu, t3_hdu])
    hdul.writeto(filename, overwrite=True)
    hdul.close()


def test_read_synthetic_oifits():
    """Test reading a synthetic OIFITS file."""
    print("Testing OIFITS reader with synthetic file...")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Create synthetic OIFITS file
        nv2 = 15
        nt3 = 8
        create_synthetic_oifits(tmp_filename, nv2=nv2, nt3=nt3, nwave=3)

        # Read it
        oi_data = read_oifits(tmp_filename, verbose=True)

        # Check data structure
        assert oi_data.nv2 == nv2, f"Wrong number of V²: {oi_data.nv2} vs {nv2}"
        assert oi_data.nt3amp == nt3, f"Wrong number of T3: {oi_data.nt3amp} vs {nt3}"
        assert oi_data.nt3phi == nt3, f"Wrong number of T3phi: {oi_data.nt3phi} vs {nt3}"

        # Check UV array shape
        # Should have: nv2 + 3*nt3 UV points (3 baselines per triangle)
        expected_nuv = nv2 + 3 * nt3
        actual_nuv = oi_data.uv.shape[1]
        assert actual_nuv == expected_nuv, \
            f"Wrong UV array size: {actual_nuv} vs {expected_nuv}"

        # Check data ranges
        assert np.all(oi_data.v2 >= 0), "V² should be non-negative"
        assert np.all(oi_data.v2 <= 1), "V² should be <= 1"
        assert np.all(oi_data.v2_err > 0), "V² errors should be positive"

        assert np.all(oi_data.t3amp >= 0), "T3amp should be non-negative"
        assert np.all(oi_data.t3amp_err > 0), "T3amp errors should be positive"
        assert np.all(oi_data.t3phi_err > 0), "T3phi errors should be positive"

        # Check indices
        assert len(oi_data.indx_v2) == nv2, "Wrong V² index length"
        assert len(oi_data.indx_t3_1) == nt3, "Wrong T3 index length"
        assert len(oi_data.indx_t3_2) == nt3, "Wrong T3 index length"
        assert len(oi_data.indx_t3_3) == nt3, "Wrong T3 index length"

        # Check index ranges
        assert np.all(oi_data.indx_v2 < actual_nuv), "V² indices out of range"
        assert np.all(oi_data.indx_t3_1 < actual_nuv), "T3_1 indices out of range"
        assert np.all(oi_data.indx_t3_2 < actual_nuv), "T3_2 indices out of range"
        assert np.all(oi_data.indx_t3_3 < actual_nuv), "T3_3 indices out of range"

        print("  ✓ Synthetic OIFITS reading test passed")

    finally:
        # Clean up
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_wavelength_filtering():
    """Test wavelength range filtering."""
    print("\nTesting wavelength filtering...")

    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Create OIFITS with specific wavelength range
        create_synthetic_oifits(tmp_filename, nv2=10, nt3=5, nwave=5)

        # Read without filtering
        oi_data_all = read_oifits(tmp_filename, verbose=False)

        # Read with filtering (should still get data since wavelengths are ~1.65 micron)
        oi_data_filtered = read_oifits(
            tmp_filename,
            wavelength_range=(1.6e-6, 1.7e-6),
            verbose=False
        )

        # Should have same number of points (wavelength averaging)
        assert oi_data_filtered.nv2 == oi_data_all.nv2, \
            "Wavelength filtering changed number of V² points"

        print("  ✓ Wavelength filtering test passed")

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_multiepoch_reading():
    """Test reading multiple epochs."""
    print("\nTesting multi-epoch reading...")

    tmp_files = []

    try:
        # Create 3 synthetic OIFITS files
        for i in range(3):
            tmp_file = tempfile.NamedTemporaryFile(suffix='.oifits', delete=False)
            tmp_files.append(tmp_file.name)
            tmp_file.close()

            create_synthetic_oifits(
                tmp_file.name,
                nv2=10 + i*2,  # Different sizes
                nt3=5 + i,
                nwave=3
            )

        # Read all epochs
        data_list = read_oifits_multiepoch(tmp_files, verbose=False)

        assert len(data_list) == 3, f"Wrong number of epochs: {len(data_list)}"

        # Check that each epoch has different data sizes
        assert data_list[0].nv2 == 10, "Epoch 1 wrong V² count"
        assert data_list[1].nv2 == 12, "Epoch 2 wrong V² count"
        assert data_list[2].nv2 == 14, "Epoch 3 wrong V² count"

        print("  ✓ Multi-epoch reading test passed")

    finally:
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)


def test_summarize_oifits():
    """Test OIFITS data summary."""
    print("\nTesting OIFITS summary...")

    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        create_synthetic_oifits(tmp_filename, nv2=20, nt3=10, nwave=3)
        oi_data = read_oifits(tmp_filename, verbose=False)

        summary = summarize_oifits(oi_data)

        print(f"  Summary:")
        print(f"    nv2: {summary['nv2']}")
        print(f"    nt3: {summary['nt3']}")
        print(f"    nuv: {summary['nuv']}")
        print(f"    V² range: {summary['v2_range']}")
        print(f"    T3phi range: {summary['t3phi_range']}")
        print(f"    Max baseline: {summary['uv_max_baseline']:.2e} cycles/rad")

        assert summary['nv2'] == 20, "Summary V² count wrong"
        assert summary['nt3'] == 10, "Summary T3 count wrong"
        assert summary['nuv'] == 20 + 3*10, "Summary UV count wrong"

        print("  ✓ OIFITS summary test passed")

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_flagged_data_filtering():
    """Test that flagged data is properly excluded."""
    print("\nTesting flagged data filtering...")

    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        # Create file with some flagged data
        create_synthetic_oifits(tmp_filename, nv2=10, nt3=5, nwave=3)

        # Read and manually set some flags
        hdul = fits.open(tmp_filename, mode='update')

        # Flag first V² point
        v2_table = hdul['OI_VIS2']
        v2_table.data['FLAG'][0, :] = True

        # Flag first T3 point
        t3_table = hdul['OI_T3']
        t3_table.data['FLAG'][0, :] = True

        hdul.flush()
        hdul.close()

        # Read the modified file
        oi_data = read_oifits(tmp_filename, verbose=False)

        # Should have fewer points now (9 V², 4 T3)
        assert oi_data.nv2 == 9, f"Flagged V² not excluded: {oi_data.nv2}"
        assert oi_data.nt3amp == 4, f"Flagged T3 not excluded: {oi_data.nt3amp}"

        print("  ✓ Flagged data filtering test passed")

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def test_uv_coordinates():
    """Test UV coordinate conversion."""
    print("\nTesting UV coordinate conversion...")

    with tempfile.NamedTemporaryFile(suffix='.oifits', delete=False) as tmp:
        tmp_filename = tmp.name

    try:
        create_synthetic_oifits(tmp_filename, nv2=5, nt3=3, nwave=3)
        oi_data = read_oifits(tmp_filename, verbose=False)

        # Check UV coordinates are in correct units (cycles/radian)
        # For baselines ~50m and wavelength ~1.65 micron:
        # spatial freq ~ 50 / 1.65e-6 ~ 3e7 cycles/radian
        uv_mag = np.sqrt(oi_data.uv[0]**2 + oi_data.uv[1]**2)

        print(f"  UV magnitude range: [{uv_mag.min():.2e}, {uv_mag.max():.2e}] cycles/rad")

        # Should be on order of 1e7 to 1e8
        assert np.all(uv_mag > 1e6), "UV coordinates too small"
        assert np.all(uv_mag < 1e9), "UV coordinates too large"

        print("  ✓ UV coordinate conversion test passed")

    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def run_all_tests():
    """Run all OIFITS reader tests."""
    print("="*60)
    print("Running OIFITS Reader Tests")
    print("="*60)

    try:
        test_read_synthetic_oifits()
        test_wavelength_filtering()
        test_multiepoch_reading()
        test_summarize_oifits()
        test_flagged_data_filtering()
        test_uv_coordinates()

        print("\n" + "="*60)
        print("ALL OIFITS READER TESTS PASSED ✓")
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
