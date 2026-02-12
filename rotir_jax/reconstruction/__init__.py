"""Reconstruction module for ROTIR - image reconstruction from interferometric data."""

from .optimizer import (
    OptimizationResult,
    StellarImageReconstructor,
    reconstruct_stellar_surface,
    compute_reduced_chi2,
    estimate_optimal_regularization_weight,
)

from .multi_epoch import (
    Epoch,
    MultiEpochReconstructor,
    reconstruct_multi_epoch,
    compute_rotation_phase,
    extract_epoch_maps,
)

__all__ = [
    # Single-epoch reconstruction
    "OptimizationResult",
    "StellarImageReconstructor",
    "reconstruct_stellar_surface",
    "compute_reduced_chi2",
    "estimate_optimal_regularization_weight",
    # Multi-epoch reconstruction
    "Epoch",
    "MultiEpochReconstructor",
    "reconstruct_multi_epoch",
    "compute_rotation_phase",
    "extract_epoch_maps",
]
