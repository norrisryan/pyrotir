"""Regularization module for ROTIR - mathematical priors for image reconstruction."""

from .regularizers import (
    maximum_entropy,
    total_variation_l1,
    total_variation_l2,
    mean_regularization,
    bias_regularization,
    build_difference_matrix,
    build_healpix_difference_matrix,
    apply_regularizers,
)

__all__ = [
    "maximum_entropy",
    "total_variation_l1",
    "total_variation_l2",
    "mean_regularization",
    "bias_regularization",
    "build_difference_matrix",
    "build_healpix_difference_matrix",
    "apply_regularizers",
]
