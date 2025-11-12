"""
File: compute.py
Purpose: Computational algorithms for code interpolation and analysis
Created: 2025-11-11 (E50 implementation)
Used by: core.py, validate.py, main.py

This module implements the interpolation algorithms and
numerical computations for continuous code families.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.linalg import null_space

# === CONFIG ===
# Parameters
RANK_TOLERANCE = 1e-10
WEIGHT_TOLERANCE = 1e-10
DEFAULT_SEED = 42

# Notes
# - Linear interpolation preserves convexity
# - Singular value decomposition for kernel analysis
# - Numerical stability critical for rank jumps


def construct_endpoint_surjection(n: int, k: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """
    Construct random surjection with specified dimensions.

    Function: construct_endpoint_surjection
    Role: Create endpoint codes for interpolation family
    Inputs: n (qubits), k (logical dim), seed (reproducibility)
    Returns: Complex matrix with rank k
    Notes: Uses Gaussian random + SVD to ensure exact rank
    """
    np.random.seed(seed)

    # Physical and logical dimensions
    physical_dim = 2**n
    logical_dim = k

    # Create random complex matrix
    # Real and imaginary parts from standard normal
    real_part = np.random.randn(logical_dim, physical_dim)
    imag_part = np.random.randn(logical_dim, physical_dim)
    pi = real_part + 1j * imag_part

    # Normalize rows for numerical stability
    for i in range(logical_dim):
        pi[i] = pi[i] / np.linalg.norm(pi[i])

    # Verify rank
    rank = np.linalg.matrix_rank(pi, tol=RANK_TOLERANCE)
    if rank != k:
        # Force exact rank using SVD truncation
        U, s, Vh = np.linalg.svd(pi, full_matrices=False)
        # Keep only k largest singular values
        s_truncated = np.zeros_like(s)
        s_truncated[:k] = s[:k]
        pi = U @ np.diag(s_truncated) @ Vh

    return pi


def interpolate_surjections(pi_0: np.ndarray, pi_1: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation: π_t = (1-t)π_0 + t·π_1.

    Function: interpolate_surjections
    Role: Create continuous path between endpoint codes
    Inputs: pi_0 (t=0), pi_1 (t=1), t (parameter)
    Returns: Interpolated surjection matrix
    Notes: Linear interpolation may not preserve rank
    """
    if not (0 <= t <= 1):
        raise ValueError(f"Parameter t={t} must be in [0,1]")

    # Ensure matrices have same shape
    if pi_0.shape != pi_1.shape:
        raise ValueError(f"Shape mismatch: {pi_0.shape} vs {pi_1.shape}")

    # Linear interpolation
    pi_t = (1 - t) * pi_0 + t * pi_1

    return pi_t


def compute_kernel_dimension(pi: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Compute kernel dimension and basis.

    Function: compute_kernel_dimension
    Role: Find dimension and basis of ker(π)
    Inputs: pi - surjection matrix
    Returns: (kernel_dim, kernel_basis)
    Notes: Uses SVD for numerical stability
    """
    # Get dimensions
    m, n = pi.shape  # m = logical, n = physical

    # Compute rank
    rank = np.linalg.matrix_rank(pi, tol=RANK_TOLERANCE)

    # Kernel dimension by rank-nullity theorem
    kernel_dim = n - rank

    # Find kernel basis using SVD
    U, s, Vh = np.linalg.svd(pi, full_matrices=True)

    # Kernel vectors are last (n - rank) columns of V
    V = Vh.conj().T
    kernel_basis = V[:, rank:]

    return kernel_dim, kernel_basis


def compute_kernel_dimension_curve(family) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kernel dimension at all t values.

    Function: compute_kernel_dimension_curve
    Role: Track kernel dimension evolution through family
    Inputs: family - SurjectionFamily instance
    Returns: (t_values, kernel_dims)
    Notes: Main observable for smoothness analysis
    """
    t_values = family.t_values
    kernel_dims = []

    for t in t_values:
        pi_t = family.get_surjection(t)
        kernel_dim, _ = compute_kernel_dimension(pi_t)
        kernel_dims.append(kernel_dim)

    return t_values, np.array(kernel_dims)


def compute_code_distance_numerical(pi: np.ndarray) -> int:
    """
    Compute code distance from kernel weight.

    Function: compute_code_distance_numerical
    Role: Find minimum Hamming weight of kernel vectors
    Inputs: pi - surjection matrix
    Returns: Code distance d
    Notes: Expensive computation, scales with kernel dimension
    """
    # Get kernel basis
    kernel_dim, kernel_basis = compute_kernel_dimension(pi)

    if kernel_dim == 0:
        # No kernel means no codewords
        return float('inf')

    # For numerical stability, use threshold for "zero"
    min_weight = float('inf')

    # Check weight of each kernel vector
    for i in range(kernel_basis.shape[1]):
        v = kernel_basis[:, i]

        # Compute Hamming weight (non-zero components)
        weight = np.sum(np.abs(v) > WEIGHT_TOLERANCE)

        if weight > 0 and weight < min_weight:
            min_weight = weight

    # If all vectors are zero (shouldn't happen), return inf
    if min_weight == float('inf'):
        return 0

    return int(min_weight)


def verify_smoothness_criterion(t_values: np.ndarray,
                               kernel_dims: np.ndarray,
                               threshold: float = 1.0) -> Tuple[bool, float, np.ndarray]:
    """
    Check |Δdim/Δt| ≤ threshold (discrete derivative).

    Function: verify_smoothness_criterion
    Role: Validate smoothness of dimension evolution
    Inputs: t_values, kernel_dims, threshold
    Returns: (is_smooth, max_derivative, derivatives)
    Notes: DISCOVERY-BASED - reports actual max derivative
    """
    # Compute discrete derivatives
    dt = np.diff(t_values)
    ddim = np.diff(kernel_dims)

    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        derivatives = ddim / dt
        derivatives[~np.isfinite(derivatives)] = 0

    # Find maximum absolute derivative
    max_derivative = np.max(np.abs(derivatives))

    # Check smoothness criterion
    is_smooth = max_derivative <= threshold

    return is_smooth, max_derivative, derivatives


def find_rank_jumps(family) -> List[Tuple[float, int, int]]:
    """
    Find all points where rank changes.

    Function: find_rank_jumps
    Role: Identify critical points in code family
    Inputs: family - SurjectionFamily instance
    Returns: List of (t, rank_before, rank_after)
    Notes: These are potential obstructions to functoriality
    """
    jumps = []
    dims = family.compute_dimension_evolution()

    for i in range(1, len(dims)):
        if dims[i] != dims[i-1]:
            t_jump = family.t_values[i]
            rank_before = int(dims[i-1])
            rank_after = int(dims[i])
            jumps.append((t_jump, rank_before, rank_after))

    return jumps


def compute_distance_evolution(family) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track code distance d(t) through family.

    Function: compute_distance_evolution
    Role: Compute how error correction capability changes
    Inputs: family - SurjectionFamily instance
    Returns: (t_values, distances)
    Notes: Computationally expensive, may be slow
    """
    t_values = family.t_values
    distances = []

    for t in t_values:
        pi_t = family.get_surjection(t)
        d = compute_code_distance_numerical(pi_t)
        distances.append(d)

    return t_values, np.array(distances)


def analyze_singular_spectrum(pi: np.ndarray) -> dict:
    """
    Analyze singular value spectrum of surjection.

    Function: analyze_singular_spectrum
    Role: Extract spectral properties of code
    Inputs: pi - surjection matrix
    Returns: Dictionary with spectral analysis
    Notes: Singular values indicate code quality
    """
    # Compute SVD
    U, s, Vh = np.linalg.svd(pi, full_matrices=False)

    # Find effective rank (number of significant singular values)
    effective_rank = np.sum(s > RANK_TOLERANCE)

    # Compute condition number
    if s[-1] > RANK_TOLERANCE:
        condition_number = s[0] / s[-1]
    else:
        condition_number = float('inf')

    # Spectral gap (if exists)
    if effective_rank > 0 and effective_rank < len(s):
        spectral_gap = s[effective_rank - 1] - s[effective_rank]
    else:
        spectral_gap = 0

    return {
        'singular_values': s,
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'spectral_gap': spectral_gap,
        'max_singular_value': float(s[0]),
        'min_singular_value': float(s[-1])
    }


def create_test_family(n: int = 8,
                      k_initial: int = 2,
                      k_final: int = 4) -> 'SurjectionFamily':
    """
    Create test family with known properties.

    Function: create_test_family
    Role: Generate family for testing/validation
    Inputs: n (qubits), k_initial, k_final
    Returns: SurjectionFamily instance
    Notes: Used in tests to verify theory predictions
    """
    # Import here to avoid circular dependency
    from .core import SurjectionFamily

    # Create family with linear interpolation
    family = SurjectionFamily(
        n=n,
        k_initial=k_initial,
        k_final=k_final,
        n_points=100
    )

    return family