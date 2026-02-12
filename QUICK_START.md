# ROTIR Quick Start Guide 🚀

Get started with ROTIR in 5 minutes!

---

## 📋 Prerequisites

```bash
# Install dependencies
pip install numpy scipy jax jaxlib matplotlib astropy healpy

# Or with conda
conda install numpy scipy matplotlib astropy healpy
conda install -c conda-forge jax jaxlib
```

---

## 🎯 Quick Reconstruction (3 Steps)

### Step 1: Prepare Your Data

You need an **OIFITS file** from interferometry (CHARA, VLTI, NPOI, etc.)

```
your_star.fits  <-- OIFITS file with visibilities & closure phases
```

### Step 2: Edit the Notebook

Open `rotir_reconstruction_demo.ipynb` and edit these lines:

```python
# Line ~35: Your OIFITS file
OIFITS_FILE = "path/to/your/red_supergiant.fits"  # <-- CHANGE THIS

# Line ~40: Stellar parameters
DIAMETER = 44.0    # mas (your star's diameter)
T_EFF = 3500.0     # K (effective temperature)
INCLINATION = 90.0 # degrees (90 = edge-on)
```

### Step 3: Run!

```bash
jupyter notebook rotir_reconstruction_demo.ipynb
```

Then: **Run All Cells** (Cell → Run All)

Done! Results saved to:
- `surface_map_rsg.png` - Your reconstructed surface
- `reconstruction_convergence_rsg.png` - Convergence plots
- `temperature_map_rsg.npy` - Data for analysis

---

## 📊 Your Data Types

### Red Supergiants (Betelgeuse, Antares, Aldebaran)
- **Use:** Part 1 of notebook
- **Features:** Convection cells, limb darkening
- **Parameters:**
  - `DIAMETER`: 20-50 mas (large!)
  - `T_EFF`: 3000-4000 K (cool)
  - `NSIDE`: 4 (192 pixels) or 8 (768 pixels)

### Symbiotic Stars (R Aquarii, Mira AB)
- **Use:** Part 2 of notebook
- **Features:** Binary system, Roche geometry, mass transfer
- **Parameters:**
  - `SEPARATION`: 50-200 mas
  - `MASS_RATIO`: 0.3-0.8 (WD/giant)
  - `PERIOD`: 100-1000 days
  - `R_GIANT`: 30-60 mas

---

## ⚙️ Configuration Tips

### Regularization Weights

**For smooth targets** (red supergiants):
```python
regularizers = [
    {"type": "mem", "weight": 0.05},  # Smooth
    {"type": "tv", "weight": 0.01},   # Preserve edges
]
```

**For spotted stars:**
```python
regularizers = [
    {"type": "mem", "weight": 0.03},     # Less smoothing
    {"type": "tv", "weight": 0.02},      # More edge preservation
    {"type": "bias", "weight": 0.001,    # Prefer cool spots
     "params": {"bias_factor": 2.0}},
]
```

### Resolution (NSIDE)

| NSIDE | Pixels | Use Case | Speed |
|-------|--------|----------|-------|
| 2 | 48 | Quick tests | Fast |
| 3 | 108 | Binaries | Medium |
| 4 | 192 | **Standard** | Good |
| 8 | 768 | High-res | Slow |

**Recommendation:** Start with NSIDE=4, increase if needed.

### Temperature Bounds

| Star Type | T_min (K) | T_max (K) |
|-----------|-----------|-----------|
| Red supergiant | 2500 | 4500 |
| Orange giant | 3500 | 5500 |
| Solar-type | 4500 | 6500 |
| A-type | 7000 | 10000 |

---

## 🔍 Interpreting Results

### Good Reconstruction
```
✓ Success: True
✓ χ²_red ≈ 1.0 (0.8 - 1.5 acceptable)
✓ Convergence plot: smooth decrease
✓ Temperature range: physical
```

### Issues & Solutions

**χ²_red >> 1 (poor fit):**
- Decrease regularization weights (0.05 → 0.02)
- Increase resolution (NSIDE 4 → 8)
- Check data quality

**χ²_red << 1 (overfitting):**
- Increase regularization weights (0.05 → 0.1)
- Decrease resolution (NSIDE 8 → 4)
- Add more regularizers

**Unrealistic temperatures:**
- Adjust bounds (T_min, T_max)
- Check initial guess
- Verify stellar parameters

**Slow convergence:**
- Reduce MAXITER temporarily
- Check regularizer weights
- Try different initial guess

---

## 📁 File Outputs

After reconstruction, you'll have:

```
surface_map_rsg.png             - Surface visualization
reconstruction_convergence_rsg.png  - Diagnostics
temperature_map_rsg.npy         - Temperature array
tessellation_theta.npy          - θ coordinates
tessellation_phi.npy            - φ coordinates
reconstruction_report.txt       - Summary statistics
```

### Load Results Later

```python
import numpy as np
import matplotlib.pyplot as plt

# Load temperature map
T_map = np.load('temperature_map_rsg.npy')
theta = np.load('tessellation_theta.npy')
phi = np.load('tessellation_phi.npy')

# Analyze
print(f"Temperature: {T_map.min():.0f} - {T_map.max():.0f} K")
print(f"Contrast: {100*(T_map.max()-T_map.min())/T_map.mean():.1f}%")
```

---

## 🎓 Example Workflow

### Your Red Supergiant

1. **Edit notebook:**
   ```python
   OIFITS_FILE = "betelgeuse_chara_2023.fits"
   DIAMETER = 44.0  # mas
   T_EFF = 3500.0   # K
   NSIDE = 4
   ```

2. **Run reconstruction** → 5-15 minutes

3. **Check results:**
   - χ²_red ≈ 1.0? ✓
   - Temperature range reasonable? ✓
   - Surface features visible? ✓

4. **Refine if needed:**
   - Adjust regularization
   - Try higher resolution (NSIDE=8)
   - Add temporal evolution (multi-epoch)

### Your Symbiotic Star

1. **Compute orbital position:**
   ```python
   # In notebook, Part 2
   SEPARATION = 100.0  # mas
   MASS_RATIO = 0.5
   PERIOD = 640.0      # days
   ```

2. **Check Roche lobe:**
   - Fillout < 0.8: Detached ✓
   - Fillout > 0.95: Mass transfer! ⚠️

3. **Reconstruct giant component**

4. **Analyze tidal distortion**

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'rotir_jax'"
```python
# In notebook, add:
import sys
sys.path.insert(0, '/home/user/pyrotir')
```

### "OIFITS file not found"
Check file path is absolute:
```python
OIFITS_FILE = "/full/path/to/your/star.fits"
```

### "Reconstruction fails to converge"
- Increase `MAXITER` (200 → 400)
- Decrease regularization weights
- Try different initial guess
- Check data quality (are there outliers?)

### "Out of memory"
- Decrease `NSIDE` (8 → 4)
- Close other applications
- Use CPU instead of GPU

### "Takes too long"
- Start with `MAXITER=50` for testing
- Use `NSIDE=3` initially
- Increase once you confirm it works

---

## 📚 Next Steps

After successful reconstruction:

1. **Compare with literature**
   - Published diameters
   - Known features
   - Previous reconstructions

2. **Multi-wavelength**
   - Run at different wavelengths
   - Look for chromatic effects

3. **Time series**
   - Multiple epochs → spot evolution
   - Differential rotation

4. **Publish!** 📝
   - Export high-res figures
   - Write up results
   - Share with community

---

## 💡 Pro Tips

- **Start simple:** Uniform disk → basic reconstruction → high-res
- **Check χ²_red:** Should be ~1.0 for good fit
- **Regularization:** Start high, decrease gradually
- **Resolution:** NSIDE=4 usually sufficient
- **Save everything:** Results are expensive to recompute!
- **Visualize:** Convergence plots reveal issues
- **Compare:** Uniform disk vs. limb-darkened vs. reconstruction

---

## 📞 Need Help?

1. **Check convergence plots** - smooth decrease?
2. **Verify data** - reasonable number of points?
3. **Try defaults** - notebook has working examples
4. **Adjust gradually** - change one parameter at a time

---

## 🎉 Ready to Go!

```bash
cd /home/user/pyrotir
jupyter notebook rotir_reconstruction_demo.ipynb
```

**Edit → Run → Analyze → Publish!**

From photons to pixels: Mapping the surfaces of distant stars ✨

---

*For detailed documentation, see `ROTIR_PYTHON_COMPLETE.md`*
