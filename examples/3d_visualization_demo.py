"""Demo: 3D Stellar Surface Visualization

Shows how to create 3D plots of reconstructed stellar surfaces.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../rotir_jax')

from rotir_jax.tessellation.healpix import tessellation_healpix
from rotir_jax.geometry.base import create_star
from rotir_jax.visualization import (
    plot_star_3d,
    plot_star_3d_smooth,
    plot_star_3d_interactive,
    create_rotation_movie,
    plot_comparison_3d,
)

# ============================================================
# Create a mock stellar surface with features
# ============================================================

print("Creating mock stellar surface...")

# Create tessellation
tess = tessellation_healpix(n=4)  # 192 pixels
npix = tess.npix

# Stellar parameters
DIAMETER = 44.0  # mas (Betelgeuse-like)
T_EFF = 3500.0   # K

# Create star
star = create_star(
    tess=tess,
    inclination=60.0,  # degrees
    orientation=0.0,
    intensities=jnp.ones(npix),
    diameter=DIAMETER,
)

# Create mock temperature map with features
theta = np.array(star.theta)  # colatitude
phi = np.array(star.phi)      # longitude

# Add features:
# 1. Hot spot (convection cell)
hot_spot = np.exp(-((theta - 1.0)**2 + (phi - 2.0)**2) / 0.3)

# 2. Cool region
cool_spot = np.exp(-((theta - 2.0)**2 + (phi - 4.5)**2) / 0.5)

# 3. Limb darkening
mu = np.clip(star.z / (DIAMETER/2), 0, 1)
limb_darkening = 1.0 - 0.6 * (1 - mu)

# Combine features
temperature_map = T_EFF * (
    limb_darkening +
    0.15 * hot_spot -
    0.10 * cool_spot +
    0.05 * np.random.randn(npix)  # Small-scale noise
)

print(f"  Temperature range: {temperature_map.min():.0f} - {temperature_map.max():.0f} K")
print(f"  Contrast: {100*(temperature_map.max() - temperature_map.min())/T_EFF:.1f}%")

# ============================================================
# 1. Basic 3D plot
# ============================================================

print("\n1. Creating basic 3D surface plot...")
fig1 = plot_star_3d(
    star=star,
    temperature_map=temperature_map,
    cmap='hot',
    title="Red Supergiant Surface (Polygonal)",
    elev=20,
    azim=45,
)
plt.savefig('star_3d_basic.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: star_3d_basic.png")

# ============================================================
# 2. Smooth interpolated plot
# ============================================================

print("\n2. Creating smooth interpolated 3D plot...")
fig2 = plot_star_3d_smooth(
    star=star,
    temperature_map=temperature_map,
    resolution=100,
    cmap='hot',
    title="Red Supergiant Surface (Smooth)",
    elev=20,
    azim=45,
)
plt.savefig('star_3d_smooth.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: star_3d_smooth.png")

# ============================================================
# 3. Multiple viewing angles
# ============================================================

print("\n3. Creating multi-angle view...")
fig3, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

for i, azim in enumerate([0, 90, 180]):
    ax = axes[i]

    # Plot on existing axis (reuse logic)
    from rotir_jax.visualization.surface_plot import _plot_star_on_axis
    _plot_star_on_axis(
        ax, star, temperature_map,
        cmap='hot',
        vmin=temperature_map.min(),
        vmax=temperature_map.max(),
        title=f"View: {azim}°"
    )
    ax.view_init(elev=20, azim=azim)

plt.tight_layout()
plt.savefig('star_3d_multiview.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: star_3d_multiview.png")

# ============================================================
# 4. Interactive plot (if plotly available)
# ============================================================

print("\n4. Creating interactive 3D plot (Plotly)...")
try:
    fig_interactive = plot_star_3d_interactive(
        star=star,
        temperature_map=temperature_map,
        resolution=60,
        cmap='hot',
        title="Interactive Stellar Surface (drag to rotate)"
    )
    fig_interactive.write_html('star_3d_interactive.html')
    print("   ✓ Saved: star_3d_interactive.html")
    print("     Open in browser to interact!")
except ImportError:
    print("   ⚠️  Plotly not installed. Run: pip install plotly")

# ============================================================
# 5. Comparison plot
# ============================================================

print("\n5. Creating comparison plot...")

# Create variations
temp_uniform = jnp.ones(npix) * T_EFF
temp_limb_only = T_EFF * limb_darkening
temp_with_spots = temperature_map

fig5 = plot_comparison_3d(
    star=star,
    temp_maps=[temp_uniform, temp_limb_only, temp_with_spots],
    labels=["Uniform", "Limb Darkening", "With Convection"],
    cmap='hot',
)
plt.savefig('star_3d_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: star_3d_comparison.png")

# ============================================================
# 6. Rotation animation (optional - can be slow)
# ============================================================

print("\n6. Creating rotation animation...")
CREATE_ANIMATION = False  # Set to True to create GIF

if CREATE_ANIMATION:
    create_rotation_movie(
        star=star,
        temperature_map=temperature_map,
        n_frames=36,
        filename='star_rotation.gif',
        cmap='hot',
        fps=10,
    )
    print("   ✓ Saved: star_rotation.gif")
else:
    print("   ⚠️  Skipped (set CREATE_ANIMATION=True to enable)")

# ============================================================
# Summary
# ============================================================

print("\n" + "="*60)
print("3D Visualization Demo Complete!")
print("="*60)
print("\nGenerated files:")
print("  - star_3d_basic.png       : Polygonal 3D surface")
print("  - star_3d_smooth.png      : Smooth interpolated surface")
print("  - star_3d_multiview.png   : Multiple viewing angles")
print("  - star_3d_comparison.png  : Side-by-side comparison")
print("  - star_3d_interactive.html: Interactive (if plotly available)")
print("\nUsage in reconstruction:")
print("  >>> from rotir_jax.visualization import plot_star_3d")
print("  >>> fig = plot_star_3d(star, result.x_solution)")
print("  >>> plt.show()")
print("\n✨ Beautiful stellar surfaces! ✨")
