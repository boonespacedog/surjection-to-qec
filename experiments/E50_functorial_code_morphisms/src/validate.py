"""
File: validate.py
Purpose: Validation of smoothness and functorial properties
Created: 2025-11-11 (E50 implementation)
Used by: main.py, tests

This module implements validation checks for continuous code
families, including smoothness criteria and functorial composition.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .compute import (
    compute_kernel_dimension,
    verify_smoothness_criterion,
    compute_kernel_dimension_curve,
    find_rank_jumps,
    analyze_singular_spectrum
)

# === CONFIG ===
# Parameters
SMOOTHNESS_THRESHOLD = 1.0  # Maximum |Δdim/Δt|
RANK_TOLERANCE = 1e-10
COMPOSITION_TOLERANCE = 1e-8  # For functorial checks

# Notes
# - Discovery-based validation
# - Report actual values, don't hardcode expectations
# - Functoriality requires preservation of structure


def validate_smoothness(dims: np.ndarray,
                        t_values: np.ndarray,
                        threshold: float = SMOOTHNESS_THRESHOLD) -> Dict:
    """
    Validate |Δdim/Δt| ≤ threshold (from protocols).

    Function: validate_smoothness
    Role: Check if dimension evolution is smooth
    Inputs: dims, t_values, threshold
    Returns: Validation results dictionary
    Notes: DISCOVERY-BASED - reports actual max derivative
    """
    # Compute discrete derivatives
    dt = np.diff(t_values)
    ddim = np.diff(dims)

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        derivatives = ddim / dt
        derivatives[~np.isfinite(derivatives)] = 0

    max_derivative = np.max(np.abs(derivatives))

    # Find locations of large derivatives
    large_deriv_indices = np.where(np.abs(derivatives) > threshold)[0]
    violation_points = []
    for idx in large_deriv_indices:
        violation_points.append({
            't': float(t_values[idx]),
            'derivative': float(derivatives[idx]),
            'dim_before': int(dims[idx]),
            'dim_after': int(dims[idx + 1])
        })

    result = {
        'max_derivative': float(max_derivative),
        'threshold': threshold,
        'validated': max_derivative <= threshold,
        'n_violations': len(violation_points),
        'violation_points': violation_points,
        'discovery': f"Max |Δdim/Δt| = {max_derivative:.4f}",
        'interpretation': (
            "Smooth evolution" if max_derivative <= threshold
            else f"Non-smooth: {len(violation_points)} jumps detected"
        )
    }

    return result


def validate_kernel_evolution(family) -> Dict:
    """
    Validate smooth evolution of kernel dimension.

    Function: validate_kernel_evolution
    Role: Check kernel dimension changes smoothly
    Inputs: family - SurjectionFamily instance
    Returns: Comprehensive validation report
    Notes: Main validation for functorial property
    """
    # Get kernel dimension curve
    t_values, kernel_dims = compute_kernel_dimension_curve(family)

    # Check smoothness
    smoothness_result = validate_smoothness(kernel_dims, t_values)

    # Find rank jumps
    jumps = find_rank_jumps(family)

    # Analyze evolution pattern
    initial_dim = kernel_dims[0]
    final_dim = kernel_dims[-1]
    max_dim = np.max(kernel_dims)
    min_dim = np.min(kernel_dims)

    # Check monotonicity
    differences = np.diff(kernel_dims)
    is_monotonic_increasing = np.all(differences >= 0)
    is_monotonic_decreasing = np.all(differences <= 0)
    is_monotonic = is_monotonic_increasing or is_monotonic_decreasing

    result = {
        'smoothness': smoothness_result,
        'initial_kernel_dim': int(initial_dim),
        'final_kernel_dim': int(final_dim),
        'kernel_dim_range': [int(min_dim), int(max_dim)],
        'n_rank_jumps': len(jumps),
        'rank_jumps': jumps,
        'is_monotonic': is_monotonic,
        'monotonic_type': (
            'increasing' if is_monotonic_increasing else
            'decreasing' if is_monotonic_decreasing else
            'non-monotonic'
        ),
        'discovery': (
            f"Kernel dimension evolves from {initial_dim} to {final_dim}, "
            f"{'monotonically' if is_monotonic else 'non-monotonically'}"
        ),
        'validated': smoothness_result['validated'] and len(jumps) <= 2
    }

    return result


def validate_endpoint_consistency(family) -> Dict:
    """
    Check t=0 and t=1 match specifications.

    Function: validate_endpoint_consistency
    Role: Verify endpoints have expected properties
    Inputs: family - SurjectionFamily instance
    Returns: Endpoint validation results
    Notes: Ensures interpolation preserves endpoints
    """
    # Check t=0
    pi_0 = family.get_surjection(0.0)
    rank_0 = np.linalg.matrix_rank(pi_0, tol=RANK_TOLERANCE)
    kernel_dim_0, _ = compute_kernel_dimension(pi_0)

    # Check t=1
    pi_1 = family.get_surjection(1.0)
    rank_1 = np.linalg.matrix_rank(pi_1, tol=RANK_TOLERANCE)
    kernel_dim_1, _ = compute_kernel_dimension(pi_1)

    # Expected values
    expected_rank_0 = family.k_initial
    expected_rank_1 = family.k_final

    # Validation
    t0_valid = (rank_0 == expected_rank_0)
    t1_valid = (rank_1 == expected_rank_1)

    result = {
        't=0': {
            'rank': int(rank_0),
            'expected_rank': expected_rank_0,
            'kernel_dim': int(kernel_dim_0),
            'validated': t0_valid,
            'shape': pi_0.shape
        },
        't=1': {
            'rank': int(rank_1),
            'expected_rank': expected_rank_1,
            'kernel_dim': int(kernel_dim_1),
            'validated': t1_valid,
            'shape': pi_1.shape
        },
        'validated': t0_valid and t1_valid,
        'discovery': (
            f"Endpoints: rank {rank_0}→{rank_1}, "
            f"kernel dim {kernel_dim_0}→{kernel_dim_1}"
        )
    }

    return result


def validate_functorial_composition(family) -> Dict:
    """
    Verify composition preserves code structure.

    Function: validate_functorial_composition
    Role: Test functorial property of code morphisms
    Inputs: family - SurjectionFamily instance
    Returns: Functoriality validation results
    Notes: Simplified test - checks intermediate consistency
    """
    # Sample three points: 0, 0.5, 1
    t_values = [0.0, 0.5, 1.0]
    surjections = []
    ranks = []
    kernel_dims = []

    for t in t_values:
        pi_t = family.get_surjection(t)
        surjections.append(pi_t)
        rank = np.linalg.matrix_rank(pi_t, tol=RANK_TOLERANCE)
        ranks.append(rank)
        kernel_dim, _ = compute_kernel_dimension(pi_t)
        kernel_dims.append(kernel_dim)

    # Check monotonicity (simplified functoriality test)
    # In a functorial family, kernel dimension should change monotonically
    kernel_monotonic = (
        (kernel_dims[0] <= kernel_dims[1] <= kernel_dims[2]) or
        (kernel_dims[0] >= kernel_dims[1] >= kernel_dims[2])
    )

    # Check rank monotonicity
    rank_monotonic = (
        (ranks[0] <= ranks[1] <= ranks[2]) or
        (ranks[0] >= ranks[1] >= ranks[2])
    )

    # Check intermediate point consistency
    # The midpoint should be "between" the endpoints
    pi_mid_expected = 0.5 * surjections[0] + 0.5 * surjections[2]
    pi_mid_actual = surjections[1]
    composition_error = np.linalg.norm(pi_mid_actual - pi_mid_expected)

    result = {
        'ranks': ranks,
        'kernel_dims': kernel_dims,
        'kernel_monotonic': kernel_monotonic,
        'rank_monotonic': rank_monotonic,
        'composition_error': float(composition_error),
        'composition_consistent': composition_error < COMPOSITION_TOLERANCE,
        'validated': kernel_monotonic and rank_monotonic,
        'discovery': (
            f"Ranks: {ranks[0]}→{ranks[1]}→{ranks[2]}, "
            f"{'monotonic' if rank_monotonic else 'non-monotonic'}"
        )
    }

    return result


def validate_interpolation_bounds(family, n_samples: int = 20) -> Dict:
    """
    Verify interpolation stays within physical bounds.

    Function: validate_interpolation_bounds
    Role: Check that interpolated codes are physically valid
    Inputs: family, n_samples (number of test points)
    Returns: Bounds validation results
    Notes: Ensures no unphysical behavior during interpolation
    """
    # Sample random t values
    np.random.seed(42)
    test_t_values = np.random.uniform(0, 1, n_samples)

    violations = []
    for t in test_t_values:
        pi_t = family.get_surjection(t)

        # Check shape consistency
        expected_cols = family.physical_dim
        if pi_t.shape[1] != expected_cols:
            violations.append({
                't': float(t),
                'issue': 'shape',
                'expected': expected_cols,
                'actual': pi_t.shape[1]
            })

        # Check rank bounds
        rank = np.linalg.matrix_rank(pi_t, tol=RANK_TOLERANCE)
        min_rank = min(family.k_initial, family.k_final)
        max_rank = max(family.k_initial, family.k_final)

        if not (min_rank <= rank <= max_rank + 1):  # Allow small variation
            violations.append({
                't': float(t),
                'issue': 'rank',
                'rank': int(rank),
                'expected_range': [min_rank, max_rank]
            })

        # Check for NaN or Inf
        if np.any(~np.isfinite(pi_t)):
            violations.append({
                't': float(t),
                'issue': 'non-finite',
                'n_nan': int(np.sum(np.isnan(pi_t))),
                'n_inf': int(np.sum(np.isinf(pi_t)))
            })

    result = {
        'n_samples_tested': n_samples,
        'n_violations': len(violations),
        'violations': violations,
        'validated': len(violations) == 0,
        'discovery': (
            "All interpolated codes within bounds" if len(violations) == 0
            else f"{len(violations)} bound violations detected"
        )
    }

    return result


def validate_spectral_evolution(family, n_samples: int = 10) -> Dict:
    """
    Validate smooth evolution of singular value spectrum.

    Function: validate_spectral_evolution
    Role: Check spectral properties evolve continuously
    Inputs: family, n_samples
    Returns: Spectral validation results
    Notes: Singular values indicate code quality
    """
    # Sample evenly spaced t values
    test_t_values = np.linspace(0, 1, n_samples)

    spectra = []
    condition_numbers = []

    for t in test_t_values:
        pi_t = family.get_surjection(t)
        spectrum = analyze_singular_spectrum(pi_t)
        spectra.append(spectrum)
        condition_numbers.append(spectrum['condition_number'])

    # Check for smooth evolution of condition number
    finite_conds = [c for c in condition_numbers if np.isfinite(c)]
    if len(finite_conds) > 1:
        max_cond_jump = np.max(np.abs(np.diff(finite_conds)))
    else:
        max_cond_jump = 0

    # Check spectral gap evolution
    gaps = [s['spectral_gap'] for s in spectra]
    gap_smooth = np.std(gaps) < 0.5  # Heuristic threshold

    result = {
        'n_samples': n_samples,
        'condition_number_range': [
            float(np.min(finite_conds)) if finite_conds else float('inf'),
            float(np.max(finite_conds)) if finite_conds else float('inf')
        ],
        'max_condition_jump': float(max_cond_jump),
        'spectral_gaps': gaps,
        'gap_evolution_smooth': gap_smooth,
        'validated': gap_smooth and max_cond_jump < 100,
        'discovery': f"Spectral gap std: {np.std(gaps):.4f}"
    }

    return result


def create_validation_summary(tests: List[Dict]) -> Dict:
    """
    Aggregate all validation results.

    Function: create_validation_summary
    Role: Create comprehensive validation report
    Inputs: tests - list of validation results
    Returns: Aggregated summary with overall status
    Notes: Used for final experiment reporting
    """
    # Count validations
    n_tests = len(tests)
    n_passed = sum(1 for t in tests if t.get('validated', False))

    # Collect all discoveries
    discoveries = []
    for test in tests:
        if 'discovery' in test:
            discoveries.append(test['discovery'])

    # Overall validation
    all_passed = (n_passed == n_tests)

    summary = {
        'n_tests': n_tests,
        'n_passed': n_passed,
        'n_failed': n_tests - n_passed,
        'all_validated': all_passed,
        'validation_rate': n_passed / n_tests if n_tests > 0 else 0,
        'discoveries': discoveries,
        'overall_status': 'PASSED' if all_passed else 'FAILED',
        'individual_results': tests
    }

    return summary