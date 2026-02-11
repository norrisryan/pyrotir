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

from .rapid_rotator import (
    f_rapid_rot,
    oblate_const,
    calc_omega,
    calc_rotspin,
    temperature_map_vonZeipel,
    compute_intensity_from_temperature,
    create_rapid_rotator_star,
    compute_teff_vonzeipel,
)

__all__ = [
    # Base geometry
    "rotation_matrix",
    "apply_rotation",
    "visible_mask",
    "sky_plane_projection",
    "create_star",
    "compute_limb_darkening",
    "rotate_and_project",
    # Rapid rotators
    "f_rapid_rot",
    "oblate_const",
    "calc_omega",
    "calc_rotspin",
    "temperature_map_vonZeipel",
    "compute_intensity_from_temperature",
    "create_rapid_rotator_star",
    "compute_teff_vonzeipel",
]
