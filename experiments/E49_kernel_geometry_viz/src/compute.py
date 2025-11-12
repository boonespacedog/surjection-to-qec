"""
📄 File: compute.py
Purpose: Main computation functions for kernel geometry analysis
Created: November 11, 2025
Used by: main.py, tests/test_theory.py
"""

import numpy as np
from typing import Tuple, Dict, Any
import logging


# === CONFIG ===
# 🛠️ Numerical Tolerances (from EXPERIMENTAL_DESIGN_PROTOCOLS)
SVD_TOLERANCE = 1e-14      # For identifying zero singular values
ORTHO_TOLERANCE = 1e-10    # For checking orthogonality
ISOMETRY_TOLERANCE = 1e-10 # For verifying quotient isometry

# 🧠 Notes
# - These tolerances are from paper specifications
# - SVD tolerance is tighter than orthogonality (14 vs 10 decimal places)
# - All comparisons must use explicit tolerances


def compute_kernel_via_svd(pi: np.ndarray,
                          tol: float = SVD_TOLERANCE) -> Tuple[np.ndarray, np.ndarray]:
    """
    🧠 Function: compute_kernel_via_svd
    Role: Compute kernel basis using SVD with specified tolerance
    Inputs: pi - surjection matrix (logical_dim × physical_dim)
            tol - threshold for zero singular values
    Returns: (kernel_basis, singular_values)
            kernel_basis: Orthonormal basis for ker(π)
            singular_values: All singular values for diagnostics
    Notes: Discovery-based - doesn't assume kernel dimension
    """
    # Input validation
    if not isinstance(pi, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(pi)}")

    if len(pi.shape) != 2:
        raise ValueError(f"Expected 2D matrix, got shape {pi.shape}")

    # Log computation start
    logging.info(f"Computing kernel via SVD for {pi.shape} matrix")

    # Compute SVD: π = U Σ V^H
    try:
        U, s, Vh = np.linalg.svd(pi, full_matrices=True)
    except np.linalg.LinAlgError as e:
        logging.error(f"SVD failed: {e}")
        raise

    # Log singular values for diagnostics
    logging.info(f"Singular values: {s}")
    logging.info(f"Using tolerance: {tol}")

    # Identify rank (number of non-zero singular values)
    rank = np.sum(s > tol)
    kernel_dim = pi.shape[1] - rank

    logging.info(f"Discovered: rank={rank}, kernel_dim={kernel_dim}")

    # Extract kernel basis from right singular vectors
    if kernel_dim > 0:
        # The last kernel_dim rows of Vh (transposed) form kernel basis
        kernel_basis = Vh[rank:, :].T.conj()
    else:
        # Empty kernel (full rank surjection)
        kernel_basis = np.zeros((pi.shape[1], 0), dtype=np.complex128)

    # Verify kernel property: π @ v ≈ 0 for all v in kernel
    if kernel_dim > 0:
        residual = pi @ kernel_basis
        max_residual = np.max(np.abs(residual))
        logging.info(f"Kernel verification: max|π·v| = {max_residual:.2e}")

        if max_residual > tol * 10:
            logging.warning(f"Large kernel residual: {max_residual}")

    return kernel_basis, s


def compute_quotient_metric(pi: np.ndarray) -> np.ndarray:
    """
    🧠 Function: compute_quotient_metric
    Role: Compute Q = π π^† quotient metric
    Inputs: pi - surjection matrix
    Returns: Q - quotient metric (logical_dim × logical_dim)
    Notes: For isometry, Q should equal identity
    """
    if not isinstance(pi, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(pi)}")

    # Compute Q = π π^†
    Q = pi @ pi.conj().T

    # Log properties
    logging.info(f"Quotient metric shape: {Q.shape}")
    logging.info(f"Quotient metric trace: {np.trace(Q):.6f}")

    # Check hermiticity
    hermiticity_error = np.max(np.abs(Q - Q.conj().T))
    if hermiticity_error > 1e-14:
        logging.warning(f"Quotient metric not Hermitian: error={hermiticity_error:.2e}")

    return Q


def compute_kernel_angles(kernel_basis: np.ndarray) -> np.ndarray:
    """
    🧠 Function: compute_kernel_angles
    Role: Compute inner product matrix between kernel vectors
    Inputs: kernel_basis - matrix with kernel vectors as columns
    Returns: Inner product matrix (kernel_dim × kernel_dim)
    Notes: Diagonal should be 1, off-diagonal should be ~0 for orthonormal
    """
    if kernel_basis.shape[1] == 0:
        # Empty kernel
        return np.array([[]], dtype=np.complex128)

    # Compute Gram matrix: G = K^† K
    gram = kernel_basis.conj().T @ kernel_basis

    logging.info(f"Kernel Gram matrix shape: {gram.shape}")

    # Check if basis is normalized
    diag_elements = np.diag(gram)
    normalization_error = np.max(np.abs(diag_elements - 1.0))

    if normalization_error > 1e-10:
        logging.warning(f"Kernel basis not normalized: max error={normalization_error:.2e}")

    return gram


def verify_orthogonality(basis: np.ndarray, tol: float = ORTHO_TOLERANCE) -> bool:
    """
    🧠 Function: verify_orthogonality
    Role: Check if basis vectors are orthonormal
    Inputs: basis - matrix with basis vectors as columns
            tol - tolerance for orthogonality check
    Returns: True if orthonormal within tolerance
    Notes: Checks both normalization and orthogonality
    """
    if basis.shape[1] <= 1:
        # Single vector or empty - trivially orthogonal
        return True

    # Compute Gram matrix
    gram = basis.conj().T @ basis

    # Check normalization (diagonal elements)
    diag_error = np.max(np.abs(np.diag(gram) - 1.0))

    # Check orthogonality (off-diagonal elements)
    gram_copy = gram.copy()
    np.fill_diagonal(gram_copy, 0.0)
    off_diag_error = np.max(np.abs(gram_copy))

    logging.info(f"Orthogonality check: diag_error={diag_error:.2e}, "
                f"off_diag_error={off_diag_error:.2e}")

    return diag_error < tol and off_diag_error < tol


def analyze_singular_spectrum(s: np.ndarray, tol: float = SVD_TOLERANCE) -> Dict[str, Any]:
    """
    🧠 Function: analyze_singular_spectrum
    Role: Detailed analysis of singular value spectrum
    Inputs: s - array of singular values
            tol - threshold for zero
    Returns: Dictionary with spectrum analysis
    Notes: Helps diagnose numerical issues
    """
    analysis = {
        'n_total': len(s),
        'n_nonzero': int(np.sum(s > tol)),
        'n_zero': int(np.sum(s <= tol)),
        'largest': float(np.max(s)) if len(s) > 0 else 0.0,
        'smallest_nonzero': float(np.min(s[s > tol])) if np.any(s > tol) else 0.0,
        'condition_number': float(np.max(s) / np.min(s[s > tol]))
                           if np.any(s > tol) else float('inf'),
        'gap_ratio': float(s[-1] / s[0]) if len(s) > 1 and s[0] > 0 else 0.0,
        'threshold_used': float(tol)
    }

    # Log spectrum details
    logging.info("Singular value spectrum analysis:")
    for key, value in analysis.items():
        logging.info(f"  {key}: {value}")

    return analysis


def compute_projection_error(pi: np.ndarray, kernel_basis: np.ndarray) -> float:
    """
    🧠 Function: compute_projection_error
    Role: Verify kernel vectors are in null space of π
    Inputs: pi - surjection matrix
            kernel_basis - proposed kernel basis
    Returns: Maximum norm of π·v for v in kernel
    Notes: Should be < tolerance for valid kernel
    """
    if kernel_basis.shape[1] == 0:
        return 0.0

    # Compute π·K where K is kernel basis
    projection = pi @ kernel_basis

    # Find maximum projection error
    max_error = np.max(np.abs(projection))

    logging.info(f"Kernel projection error: max|π·v| = {max_error:.2e}")

    return float(max_error)


def create_standard_surjection() -> np.ndarray:
    """
    🧠 Function: create_standard_surjection
    Role: Create the standard 4→2 surjection from paper
    Inputs: None
    Returns: π matrix from Example 4.1
    Notes: This is the canonical example from paper lines 270-274
    """
    # From paper Example 4.1:
    # π: C^4 → C^2 defined by:
    pi = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=np.complex128)

    logging.info("Created standard 4→2 surjection from paper Example 4.1")

    return pi