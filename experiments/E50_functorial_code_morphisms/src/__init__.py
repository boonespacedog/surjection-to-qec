"""
E50: Functorial Code Morphisms
Continuous families of quantum error correcting codes

This module implements continuous interpolation between
surjection-based quantum codes, tracking kernel dimension
evolution and validating functorial properties.
"""

from .core import SurjectionFamily
from .compute import (
    construct_endpoint_surjection,
    interpolate_surjections,
    compute_kernel_dimension_curve,
    compute_code_distance_numerical,
    verify_smoothness_criterion
)
from .validate import (
    validate_kernel_evolution,
    validate_endpoint_consistency,
    validate_functorial_composition,
    create_validation_summary
)
from .visualize import (
    plot_kernel_evolution,
    plot_code_parameters,
    plot_singular_values,
    create_all_visualizations
)

__all__ = [
    'SurjectionFamily',
    'construct_endpoint_surjection',
    'interpolate_surjections',
    'compute_kernel_dimension_curve',
    'compute_code_distance_numerical',
    'verify_smoothness_criterion',
    'validate_kernel_evolution',
    'validate_endpoint_consistency',
    'validate_functorial_composition',
    'create_validation_summary',
    'plot_kernel_evolution',
    'plot_code_parameters',
    'plot_singular_values',
    'create_all_visualizations'
]