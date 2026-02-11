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

from .orbits import (
    compute_E_NR,
    compute_eccentric_anomaly,
    compute_true_anomaly,
    compute_orbital_coefficients,
    compute_separation,
    binary_orbit_relative,
    binary_orbit_absolute,
    compute_orbital_phase,
    compute_radial_velocity,
    compute_masses_from_orbit,
)

from .roche import (
    eggleton_roche_radius,
    roche_radius_pathania,
    compute_potential_primary,
    compute_potential_secondary,
    solve_roche_radius,
    compute_fillout_factor,
    compute_roche_shape,
    compute_L1_distance,
    is_contact_binary,
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
    # Orbital mechanics
    "compute_E_NR",
    "compute_eccentric_anomaly",
    "compute_true_anomaly",
    "compute_orbital_coefficients",
    "compute_separation",
    "binary_orbit_relative",
    "binary_orbit_absolute",
    "compute_orbital_phase",
    "compute_radial_velocity",
    "compute_masses_from_orbit",
    # Roche geometry
    "eggleton_roche_radius",
    "roche_radius_pathania",
    "compute_potential_primary",
    "compute_potential_secondary",
    "solve_roche_radius",
    "compute_fillout_factor",
    "compute_roche_shape",
    "compute_L1_distance",
    "is_contact_binary",
]
