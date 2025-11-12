"""
E49: Kernel Geometry Visualization
Surjection-based QEC code analysis
"""

from .core import SurjectionCode
from .compute import (
    compute_kernel_via_svd,
    compute_quotient_metric,
    create_standard_surjection
)
from .validate import run_full_validation
from .visualize import save_all_figures

__all__ = [
    'SurjectionCode',
    'compute_kernel_via_svd',
    'compute_quotient_metric',
    'create_standard_surjection',
    'run_full_validation',
    'save_all_figures'
]

__version__ = '1.0.0'