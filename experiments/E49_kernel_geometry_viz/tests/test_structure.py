"""
📄 File: test_structure.py
Purpose: Code correctness tests for kernel geometry computation
Created: November 11, 2025
Used by: pytest test suite
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import SurjectionCode
from compute import (
    compute_kernel_via_svd,
    compute_quotient_metric,
    compute_kernel_angles,
    verify_orthogonality,
    create_standard_surjection
)


class TestSurjectionCodeStructure:
    """Test SurjectionCode class structure and methods."""

    def test_initialization_valid(self):
        """Test initialization with valid surjection."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        assert code.physical_dim == 4
        assert code.logical_dim == 2
        assert code.tolerance == 1e-14

    def test_initialization_complex_conversion(self):
        """Test automatic conversion to complex dtype."""
        # Create real-valued matrix
        pi = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)

        with pytest.warns(UserWarning, match="Converting pi_matrix to complex"):
            code = SurjectionCode(pi)

        assert np.issubdtype(code.pi.dtype, np.complexfloating)

    def test_initialization_not_surjective(self):
        """Test rejection of non-surjective matrix."""
        # Rank-deficient matrix (rows linearly dependent)
        pi = np.array([
            [1, 0, 1, 0],
            [2, 0, 2, 0]  # Second row is 2× first row
        ], dtype=np.complex128)

        with pytest.raises(ValueError, match="not full row rank"):
            SurjectionCode(pi)

    def test_initialization_invalid_input(self):
        """Test rejection of invalid input types."""
        with pytest.raises(TypeError, match="must be numpy array"):
            SurjectionCode([[1, 0], [0, 1]])  # List instead of array

    def test_kernel_caching(self):
        """Test that kernel is computed once and cached."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        # First access computes kernel
        kernel1 = code.kernel
        # Second access returns cached value
        kernel2 = code.kernel

        # Should be same object (cached)
        assert kernel1 is kernel2

    def test_singular_values_property(self):
        """Test singular values are computed and cached."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        s = code.singular_values
        assert isinstance(s, np.ndarray)
        assert len(s) == 4  # Physical dimension

        # Check caching
        s2 = code.singular_values
        assert s is s2

    def test_quotient_metric_property(self):
        """Test quotient metric computation."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        Q = code.quotient_metric
        assert Q.shape == (2, 2)  # logical_dim × logical_dim

        # Check Hermitian
        assert np.allclose(Q, Q.conj().T)

    def test_kernel_dimension_property(self):
        """Test kernel dimension calculation."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        # Should discover dimension = 2
        kernel_dim = code.kernel_dim
        assert isinstance(kernel_dim, int)
        assert kernel_dim >= 0
        assert kernel_dim <= 4  # Can't exceed physical dimension

    def test_get_state_dict(self):
        """Test state dictionary serialization."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        state = code.get_state_dict()

        assert 'surjection_matrix' in state
        assert 'physical_dim' in state
        assert 'logical_dim' in state
        assert 'kernel_dim' in state
        assert 'singular_values' in state
        assert 'quotient_metric' in state
        assert 'code_distance' in state
        assert 'orthogonality_error' in state


class TestComputeFunctions:
    """Test standalone computation functions."""

    def test_compute_kernel_via_svd_standard(self):
        """Test kernel computation for standard example."""
        pi = create_standard_surjection()
        kernel_basis, s = compute_kernel_via_svd(pi)

        # Check shapes
        assert kernel_basis.shape[0] == 4  # Physical dimension
        # Don't hardcode kernel dimension!
        discovered_kernel_dim = kernel_basis.shape[1]
        print(f"Discovered kernel dimension: {discovered_kernel_dim}")

        assert len(s) == 2  # Number of singular values = min(m,n)

    def test_compute_kernel_via_svd_full_rank(self):
        """Test kernel computation for full rank matrix."""
        # Full rank 2×4 matrix (no kernel)
        pi = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.complex128)

        kernel_basis, s = compute_kernel_via_svd(pi)

        # For full rank with more columns than rows
        # kernel dimension = 4 - 2 = 2
        discovered_dim = kernel_basis.shape[1]
        print(f"Full rank case - discovered kernel dim: {discovered_dim}")
        assert discovered_dim == 2

    def test_compute_kernel_via_svd_invalid_input(self):
        """Test kernel computation with invalid input."""
        with pytest.raises(TypeError):
            compute_kernel_via_svd("not an array")

        with pytest.raises(ValueError):
            compute_kernel_via_svd(np.array([1, 2, 3]))  # 1D array

    def test_compute_quotient_metric(self):
        """Test quotient metric computation."""
        pi = create_standard_surjection()
        Q = compute_quotient_metric(pi)

        assert Q.shape == (2, 2)
        assert np.issubdtype(Q.dtype, np.complexfloating)

        # Check Hermiticity
        assert np.allclose(Q, Q.conj().T, atol=1e-14)

    def test_compute_kernel_angles_orthonormal(self):
        """Test kernel angle computation for orthonormal basis."""
        # Create orthonormal basis
        basis = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ], dtype=np.complex128) / np.sqrt(1)

        gram = compute_kernel_angles(basis)

        assert gram.shape == (2, 2)
        # Diagonal should be ~1
        assert np.allclose(np.diag(gram), 1.0, atol=1e-10)
        # Off-diagonal should be ~0
        assert np.abs(gram[0, 1]) < 1e-10
        assert np.abs(gram[1, 0]) < 1e-10

    def test_compute_kernel_angles_empty(self):
        """Test kernel angle computation for empty kernel."""
        empty_basis = np.zeros((4, 0), dtype=np.complex128)
        gram = compute_kernel_angles(empty_basis)

        assert gram.shape == (0, 0)

    def test_verify_orthogonality_true(self):
        """Test orthogonality verification for orthonormal basis."""
        # Create exactly orthonormal basis
        basis = np.eye(4, 2, dtype=np.complex128)

        assert verify_orthogonality(basis) is True

    def test_verify_orthogonality_false(self):
        """Test orthogonality verification for non-orthogonal basis."""
        # Create non-orthogonal basis
        basis = np.array([
            [1, 1],
            [0, 1],
            [0, 0],
            [0, 0]
        ], dtype=np.complex128)
        basis = basis / np.linalg.norm(basis, axis=0)  # Normalize

        # Vectors are not orthogonal (overlap exists)
        assert verify_orthogonality(basis, tol=1e-10) is False

    def test_verify_orthogonality_single_vector(self):
        """Test orthogonality for single vector (trivially true)."""
        single = np.array([[1], [0], [0], [0]], dtype=np.complex128)
        assert verify_orthogonality(single) is True

    def test_create_standard_surjection(self):
        """Test creation of standard surjection from paper."""
        pi = create_standard_surjection()

        assert pi.shape == (2, 4)
        assert np.issubdtype(pi.dtype, np.complexfloating)

        # Verify exact structure from paper
        expected = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ], dtype=np.complex128)

        assert np.allclose(pi, expected)