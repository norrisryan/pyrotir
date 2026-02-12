"""3D surface visualization for ROTIR stellar maps.

Renders temperature maps on 3D spherical surfaces using matplotlib or plotly.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple

import sys
sys.path.append('..')
from rotir_jax.datatypes import Star, Tessellation


def plot_star_3d(
    star: Star,
    temperature_map: jnp.ndarray,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'hot',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Stellar Surface Map",
    show_invisible: bool = False,
    alpha: float = 1.0,
    elev: float = 30,
    azim: float = 45,
) -> plt.Figure:
    """Plot stellar surface in 3D with temperature map.

    Args:
        star: Star object with geometry
        temperature_map: (npix,) temperature values for each pixel
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum temperature for colormap
        vmax: Maximum temperature for colormap
        title: Plot title
        show_invisible: If True, show backside pixels in gray
        alpha: Transparency (1.0 = opaque)
        elev: Elevation angle for view
        azim: Azimuth angle for view

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_star_3d(star, temperature_map, cmap='hot')
        >>> plt.show()
    """
    if vmin is None:
        vmin = float(temperature_map.min())
    if vmax is None:
        vmax = float(temperature_map.max())

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Get vertex coordinates from tessellation
    radius = star.diameter / 2.0
    vertices = radius * np.array(star.tess.unit_xyz)  # (npix, 5, 3)

    # Normalize colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    # Plot each pixel as a polygon
    for i in range(star.tess.npix):
        if not show_invisible and not star.visible[i]:
            continue

        # Get 4 corner vertices (skip center point at index 4)
        x = vertices[i, :4, 0]
        y = vertices[i, :4, 1]
        z = vertices[i, :4, 2]

        # Close the polygon
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])

        if star.visible[i]:
            # Visible pixel - color by temperature
            color = colormap(norm(temperature_map[i]))
            ax.plot_surface(
                x.reshape(1, -1),
                y.reshape(1, -1),
                z.reshape(1, -1),
                color=color,
                alpha=alpha,
                linewidth=0.5,
                edgecolor='black',
                antialiased=True,
            )
        else:
            # Invisible pixel - gray
            ax.plot_surface(
                x.reshape(1, -1),
                y.reshape(1, -1),
                z.reshape(1, -1),
                color='lightgray',
                alpha=0.3,
                linewidth=0.1,
                edgecolor='gray',
            )

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Temperature (K)', fontsize=12)

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Labels
    ax.set_xlabel('X (mas)', fontsize=10)
    ax.set_ylabel('Y (mas)', fontsize=10)
    ax.set_zlabel('Z (mas)', fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)

    # Equal aspect ratio
    max_range = radius
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    return fig


def plot_star_3d_smooth(
    star: Star,
    temperature_map: jnp.ndarray,
    resolution: int = 100,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'hot',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Stellar Surface Map (Interpolated)",
    elev: float = 30,
    azim: float = 45,
) -> plt.Figure:
    """Plot stellar surface in 3D with smooth interpolation.

    Uses spherical interpolation for smoother appearance.

    Args:
        star: Star object
        temperature_map: (npix,) temperature values
        resolution: Grid resolution for interpolation
        figsize: Figure size
        cmap: Colormap
        vmin: Min temperature
        vmax: Max temperature
        title: Plot title
        elev: Elevation angle
        azim: Azimuth angle

    Returns:
        Matplotlib figure
    """
    from scipy.interpolate import griddata

    if vmin is None:
        vmin = float(temperature_map.min())
    if vmax is None:
        vmax = float(temperature_map.max())

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Get pixel centers in spherical coordinates
    theta = np.array(star.theta)  # colatitude
    phi = np.array(star.phi)      # longitude
    temps = np.array(temperature_map)

    # Create regular grid
    theta_grid = np.linspace(0, np.pi, resolution)
    phi_grid = np.linspace(0, 2*np.pi, resolution)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)

    # Interpolate temperature onto grid
    points = np.column_stack([theta, phi])
    temps_grid = griddata(
        points, temps,
        (theta_mesh, phi_mesh),
        method='linear',
        fill_value=temps.mean()
    )

    # Convert to Cartesian
    radius = star.diameter / 2.0
    x_mesh = radius * np.sin(theta_mesh) * np.cos(phi_mesh)
    y_mesh = radius * np.sin(theta_mesh) * np.sin(phi_mesh)
    z_mesh = radius * np.cos(theta_mesh)

    # Apply visibility mask (only front hemisphere)
    # Use star's rotation to determine visibility
    from rotir_jax.geometry.base import rotation_matrix, apply_rotation
    rot_mat = rotation_matrix(star.inclination, 0.0, star.orientation)

    # Rotate grid points
    xyz = np.stack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()], axis=-1)
    xyz_rot = (rot_mat @ xyz.T).T
    visible_mask = xyz_rot[:, 2] > 0  # z > 0 is visible

    # Mask invisible points
    temps_masked = temps_grid.ravel()
    temps_masked[~visible_mask] = np.nan

    # Plot surface
    surf = ax.plot_surface(
        x_mesh, y_mesh, z_mesh,
        facecolors=cm.get_cmap(cmap)(plt.Normalize(vmin, vmax)(temps_grid)),
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    # Colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Temperature (K)', fontsize=12)

    # Set view
    ax.view_init(elev=elev, azim=azim)

    # Labels
    ax.set_xlabel('X (mas)', fontsize=10)
    ax.set_ylabel('Y (mas)', fontsize=10)
    ax.set_zlabel('Z (mas)', fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)

    # Equal aspect
    max_range = radius
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    return fig


def plot_star_3d_interactive(
    star: Star,
    temperature_map: jnp.ndarray,
    resolution: int = 50,
    cmap: str = 'hot',
    title: str = "Interactive Stellar Surface",
):
    """Create interactive 3D plot using Plotly.

    Args:
        star: Star object
        temperature_map: (npix,) temperature values
        resolution: Grid resolution
        cmap: Colormap name
        title: Plot title

    Returns:
        Plotly figure (call fig.show() to display)

    Example:
        >>> fig = plot_star_3d_interactive(star, temps)
        >>> fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly not installed. Run: pip install plotly")

    from scipy.interpolate import griddata

    # Get data
    theta = np.array(star.theta)
    phi = np.array(star.phi)
    temps = np.array(temperature_map)

    # Create grid
    theta_grid = np.linspace(0, np.pi, resolution)
    phi_grid = np.linspace(0, 2*np.pi, resolution)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)

    # Interpolate
    points = np.column_stack([theta, phi])
    temps_grid = griddata(
        points, temps,
        (theta_mesh, phi_mesh),
        method='linear',
        fill_value=temps.mean()
    )

    # Convert to Cartesian
    radius = star.diameter / 2.0
    x = radius * np.sin(theta_mesh) * np.cos(phi_mesh)
    y = radius * np.sin(theta_mesh) * np.sin(phi_mesh)
    z = radius * np.cos(theta_mesh)

    # Create surface
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        surfacecolor=temps_grid,
        colorscale=cmap,
        colorbar=dict(title="Temperature (K)"),
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5,
        ),
        lightposition=dict(x=100, y=100, z=100),
    )])

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mas)',
            yaxis_title='Y (mas)',
            zaxis_title='Z (mas)',
            aspectmode='cube',
        ),
        width=900,
        height=700,
    )

    return fig


def create_rotation_movie(
    star: Star,
    temperature_map: jnp.ndarray,
    n_frames: int = 36,
    filename: str = "star_rotation.gif",
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'hot',
    fps: int = 10,
):
    """Create animated rotation movie of stellar surface.

    Args:
        star: Star object
        temperature_map: (npix,) temperature values
        n_frames: Number of frames (360/n_frames = degrees per frame)
        filename: Output filename (.gif or .mp4)
        figsize: Figure size
        cmap: Colormap
        fps: Frames per second

    Example:
        >>> create_rotation_movie(star, temps, filename="betelgeuse.gif")
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        raise ImportError("Animation requires pillow: pip install pillow")

    frames = []
    for i in range(n_frames):
        azim = i * (360 / n_frames)
        fig = plot_star_3d(
            star, temperature_map,
            figsize=figsize,
            cmap=cmap,
            azim=azim,
            elev=30,
            title=f"Stellar Surface (azim={azim:.0f}°)",
        )
        frames.append(fig)

    # Save animation
    print(f"Creating {n_frames}-frame animation...")

    def update(frame):
        plt.close('all')
        return frames[frame]

    anim = FuncAnimation(
        frames[0], update,
        frames=n_frames,
        interval=1000/fps,
    )

    if filename.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    else:
        anim.save(filename, fps=fps)

    print(f"✓ Saved: {filename}")


# ============================================================
# Comparison plots
# ============================================================

def plot_comparison_3d(
    star: Star,
    temp_maps: list,
    labels: list,
    figsize: Tuple[int, int] = (18, 6),
    cmap: str = 'hot',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot multiple temperature maps side-by-side in 3D.

    Args:
        star: Star object
        temp_maps: List of (npix,) temperature arrays
        labels: List of labels for each map
        figsize: Figure size
        cmap: Colormap
        vmin: Min temperature (applies to all)
        vmax: Max temperature (applies to all)

    Returns:
        Matplotlib figure

    Example:
        >>> fig = plot_comparison_3d(
        ...     star,
        ...     [initial, reconstructed, model],
        ...     ["Initial", "Reconstructed", "Model"]
        ... )
    """
    n_maps = len(temp_maps)
    fig = plt.figure(figsize=figsize)

    # Global vmin/vmax if not specified
    if vmin is None:
        vmin = min(tm.min() for tm in temp_maps)
    if vmax is None:
        vmax = max(tm.max() for tm in temp_maps)

    for i, (temp_map, label) in enumerate(zip(temp_maps, labels)):
        ax = fig.add_subplot(1, n_maps, i+1, projection='3d')

        # Reuse plotting logic
        _plot_star_on_axis(
            ax, star, temp_map,
            cmap=cmap, vmin=vmin, vmax=vmax,
            title=label,
        )

    plt.tight_layout()
    return fig


def _plot_star_on_axis(ax, star, temperature_map, cmap, vmin, vmax, title):
    """Helper to plot star on existing axis."""
    radius = star.diameter / 2.0
    vertices = radius * np.array(star.tess.unit_xyz)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    for i in range(star.tess.npix):
        if not star.visible[i]:
            continue

        x = vertices[i, :4, 0]
        y = vertices[i, :4, 1]
        z = vertices[i, :4, 2]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])

        color = colormap(norm(temperature_map[i]))
        ax.plot_surface(
            x.reshape(1, -1),
            y.reshape(1, -1),
            z.reshape(1, -1),
            color=color,
            linewidth=0.3,
            edgecolor='black',
            antialiased=True,
        )

    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('X (mas)', fontsize=8)
    ax.set_ylabel('Y (mas)', fontsize=8)
    ax.set_zlabel('Z (mas)', fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
