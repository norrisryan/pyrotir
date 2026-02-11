"""Geometry module for ROTIR - stellar surface coordinate transformations."""

from .base import (
    rotation_matrix,
    apply_rotation,
    visible_mask,
    sky_plane_projection,
    create_star,
    compute_limb_darkening,
    rotate_and_project,
)

__all__ = [
    "rotation_matrix",
    "apply_rotation",
    "visible_mask",
    "sky_plane_projection",
    "create_star",
    "compute_limb_darkening",
    "rotate_and_project",
]
