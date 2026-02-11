"""Forward model module for ROTIR - interferometric observables."""

from .polyft import (
    polygon_area,
    edge_fourier_contribution,
    polygon_fourier_transform,
    setup_polyft_matrix,
    compute_polyflux,
    cvis_to_v2,
    cvis_to_t3,
    mod360,
)

from .observables import (
    compute_observables,
    compute_chi2,
    compute_chi2_gradient,
    create_forward_model,
    compute_residuals,
    compute_reduced_chi2,
    compute_chi2_multiepoch,
)

__all__ = [
    # Polygon FT
    "polygon_area",
    "edge_fourier_contribution",
    "polygon_fourier_transform",
    "setup_polyft_matrix",
    "compute_polyflux",
    "cvis_to_v2",
    "cvis_to_t3",
    "mod360",
    # Observables and chi2
    "compute_observables",
    "compute_chi2",
    "compute_chi2_gradient",
    "create_forward_model",
    "compute_residuals",
    "compute_reduced_chi2",
    "compute_chi2_multiepoch",
]
