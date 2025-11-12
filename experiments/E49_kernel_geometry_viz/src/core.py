"""
📄 File: core.py
Purpose: Core mathematical objects for surjection-based QEC codes
Created: November 11, 2025
Used by: compute.py, validate.py, main.py
"""

import numpy as np
from typing import Optional, Tuple
import warnings


class SurjectionCode:
    """
    Quantum error correction code from bounded surjection.

    This class represents a QEC code constructed from a surjection π: B → H
    where B is the physical Hilbert space and H is the logical space.
    The kernel ker(π) encodes the error correction properties.

    Theory reference: Surjection→QEC v3, Section 4.1, lines 263-304
    """

    def __init__(self, pi_matrix: np.ndarray, tolerance: float = 1e-14):
        """
        Initialize code from surjection matrix.

        Args:
            pi_matrix: Surjection π: B → H (must be full row rank)
                      Shape: (logical_dim, physical_dim)
            tolerance: Numerical tolerance for rank computation

        Raises:
            ValueError: If pi_matrix not full row rank (not surjective)
            TypeError: If pi_matrix not complex-valued
        """
        # === Input Validation ===
        if not isinstance(pi_matrix, np.ndarray):
            raise TypeError(f"pi_matrix must be numpy array, got {type(pi_matrix)}")

        # Ensure complex dtype for quantum computation
        if not np.issubdtype(pi_matrix.dtype, np.complexfloating):
            warnings.warn("Converting pi_matrix to complex dtype", UserWarning)
            pi_matrix = pi_matrix.astype(np.complex128)

        # Store surjection
        self.pi = pi_matrix.copy()  # Defensive copy
        self.tolerance = tolerance

        # Extract dimensions
        self.logical_dim = pi_matrix.shape[0]   # Target space dimension
        self.physical_dim = pi_matrix.shape[1]  # Source space dimension

        # Validate surjectivity (full row rank)
        if not self._validate_surjection():
            raise ValueError(
                f"Surjection not full row rank. "
                f"Rank={np.linalg.matrix_rank(self.pi, tol=tolerance)} "
                f"< logical_dim={self.logical_dim}"
            )

        # Lazy-computed properties (cached after first access)
        self._kernel = None
        self._singular_values = None
        self._quotient_metric = None
        self._kernel_dim = None

    def _validate_surjection(self) -> bool:
        """
        🧠 Function: _validate_surjection
        Role: Check if π is surjective (full row rank)
        Inputs: None (uses self.pi)
        Returns: True if surjective, False otherwise
        Notes: Uses SVD to compute numerical rank
        """
        rank = np.linalg.matrix_rank(self.pi, tol=self.tolerance)
        return rank >= self.logical_dim

    @property
    def kernel(self) -> np.ndarray:
        """
        Compute kernel basis via SVD (cached).

        The kernel ker(π) = {v ∈ B : π(v) = 0} encodes the code subspace.
        We compute an orthonormal basis using SVD.

        Returns:
            kernel_basis: Orthonormal basis for ker(π)
                         Shape: (physical_dim, kernel_dim)
        """
        if self._kernel is None:
            self._kernel, self._singular_values = self._compute_kernel()
        return self._kernel

    @property
    def singular_values(self) -> np.ndarray:
        """Get singular values from SVD (cached)."""
        if self._singular_values is None:
            self._kernel, self._singular_values = self._compute_kernel()
        return self._singular_values

    @property
    def kernel_dim(self) -> int:
        """Dimension of kernel (codimension)."""
        if self._kernel_dim is None:
            # Rank-nullity theorem: dim(ker) = dim(B) - rank(π)
            rank = np.linalg.matrix_rank(self.pi, tol=self.tolerance)
            self._kernel_dim = self.physical_dim - rank
        return self._kernel_dim

    def _compute_kernel(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        🧠 Function: _compute_kernel
        Role: SVD-based kernel computation with numerical tolerance
        Inputs: None (uses self.pi)
        Returns: (kernel_basis, singular_values)
        Notes: Uses full_matrices=True to get complete null space
        """
        # Compute SVD: π = U Σ V^H
        # U: logical_dim × logical_dim (left singular vectors)
        # Σ: diagonal with singular values
        # V^H: physical_dim × physical_dim (right singular vectors, conjugate transposed)
        U, s, Vh = np.linalg.svd(self.pi, full_matrices=True)

        # Identify kernel vectors (corresponding to zero singular values)
        # The columns of V corresponding to σ_i ≈ 0 span ker(π)
        null_mask = s < self.tolerance

        # For rectangular matrices, we need to check dimensions
        n_singular = len(s)

        # Extract kernel basis from right singular vectors
        # Vh has shape (physical_dim, physical_dim) when full_matrices=True
        # We want the last (physical_dim - rank) rows of Vh transposed
        rank = np.sum(~null_mask)  # Number of non-zero singular values

        if rank < self.physical_dim:
            # Extract null space basis vectors
            kernel_basis = Vh[rank:, :].T.conj()  # Shape: (physical_dim, kernel_dim)
        else:
            # No kernel (full rank mapping)
            kernel_basis = np.zeros((self.physical_dim, 0), dtype=np.complex128)

        # Pad singular values to physical dimension if needed
        singular_values = np.zeros(self.physical_dim)
        singular_values[:n_singular] = s

        return kernel_basis, singular_values

    @property
    def quotient_metric(self) -> np.ndarray:
        """
        Compute quotient metric Q = π π^†.

        This metric represents the inner product structure on the quotient
        space B/ker(π) ≅ H. For an isometry, Q should equal the identity.

        Returns:
            Q: Quotient metric (logical_dim × logical_dim)
        """
        if self._quotient_metric is None:
            self._quotient_metric = self.pi @ self.pi.conj().T
        return self._quotient_metric

    def compute_code_distance(self) -> int:
        """
        🧠 Function: compute_code_distance
        Role: Compute minimum Hamming weight of kernel vectors
        Inputs: None (uses self.kernel)
        Returns: Code distance d (minimum weight)
        Notes: For this pedagogical example, d=1 (no error correction)
        """
        kernel_basis = self.kernel

        if kernel_basis.shape[1] == 0:
            # No kernel means infinite distance (perfect code)
            return float('inf')

        # Compute weight of each kernel basis vector
        # Weight = number of non-zero components
        weights = []
        for i in range(kernel_basis.shape[1]):
            v = kernel_basis[:, i]
            weight = np.sum(np.abs(v) > self.tolerance)
            weights.append(weight)

        # Code distance is minimum weight
        return int(min(weights)) if weights else 0

    def verify_orthogonality(self) -> float:
        """
        🧠 Function: verify_orthogonality
        Role: Check orthogonality of kernel basis vectors
        Inputs: None (uses self.kernel)
        Returns: Maximum off-diagonal inner product magnitude
        Notes: Should be < 1e-10 for orthonormal basis
        """
        kernel_basis = self.kernel

        if kernel_basis.shape[1] <= 1:
            # Single vector or empty kernel is trivially orthogonal
            return 0.0

        # Compute Gram matrix G = K^† K
        gram = kernel_basis.conj().T @ kernel_basis

        # Extract maximum off-diagonal element
        # Set diagonal to zero for this check
        np.fill_diagonal(gram, 0.0)
        max_off_diagonal = np.max(np.abs(gram))

        return float(max_off_diagonal)

    def get_state_dict(self) -> dict:
        """
        Return complete state for serialization.

        Returns:
            Dictionary containing all computed properties
        """
        return {
            'surjection_matrix': self.pi.tolist(),
            'physical_dim': self.physical_dim,
            'logical_dim': self.logical_dim,
            'kernel_dim': self.kernel_dim,
            'kernel_basis': self.kernel.tolist() if self._kernel is not None else None,
            'singular_values': self.singular_values.tolist(),
            'quotient_metric': self.quotient_metric.tolist(),
            'code_distance': self.compute_code_distance(),
            'orthogonality_error': self.verify_orthogonality(),
            'tolerance': self.tolerance
        }