"""Visualization tools for ROTIR.

3D surface plots, animations, and comparison visualizations.
"""

from .surface_plot import (
    plot_star_3d,
    plot_star_3d_smooth,
    plot_star_3d_interactive,
    create_rotation_movie,
    plot_comparison_3d,
)

__all__ = [
    'plot_star_3d',
    'plot_star_3d_smooth',
    'plot_star_3d_interactive',
    'create_rotation_movie',
    'plot_comparison_3d',
]
