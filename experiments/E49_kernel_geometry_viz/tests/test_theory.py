"""
📄 File: test_theory.py
Purpose: Theory falsification tests for kernel geometry
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
    create_standard_surjection,
    compute_projection_error,
    analyze_singular_spectrum
)


class TestTheoryPredictions:
    """
    Theory falsification tests.

    These tests DISCOVER actual values and compare to theory predictions.
    They do NOT hardcode expected outcomes - that would be circular logic.
    """

    def test_kernel_dimension_discovery(self):
        """
        Test: Discover kernel dimension via SVD rank deficiency.
        Theory predicts: dimension = 2 for the 4→2 surjection.
        """
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        # DISCOVER the actual kernel dimension
        discovered_dim = code.kernel_dim

        # Document what we found
        print(f"\n=== Kernel Dimension Discovery ===")
        print(f"Surjection shape: {pi.shape}")
        print(f"Discovered kernel dimension: {discovered_dim}")

        # Document theory prediction separately
        theory_predicts = 2
        print(f"Theory predicts: {theory_predicts}")

        # Structure test (not theory test)
        assert 0 <= discovered_dim <= 4, "Kernel dimension out of valid range"

        # Optional: Log if theory matches discovery
        if discovered_dim == theory_predicts:
            print("✓ Discovery matches theory prediction")
        else:
            print(f"✗ Discovery ({discovered_dim}) differs from theory ({theory_predicts})")
            # This is a DISCOVERY, not a failure!

        return discovered_dim

    def test_kernel_orthogonality_discovery(self):
        """
        Test: Discover if kernel basis vectors are orthogonal.
        Theory predicts: orthogonal basis from SVD.
        """
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        kernel = code.kernel

        if kernel.shape[1] == 0:
            print("Empty kernel - orthogonality trivially satisfied")
            return

        # Compute inner products between kernel vectors
        gram = kernel.conj().T @ kernel

        print(f"\n=== Kernel Orthogonality Discovery ===")
        print(f"Kernel basis shape: {kernel.shape}")
        print(f"Gram matrix:\n{gram}")

        # Check diagonal (should be ~1 for normalized)
        diagonal_values = np.diag(gram)
        print(f"Diagonal values: {diagonal_values}")

        # Check off-diagonal (should be ~0 for orthogonal)
        gram_copy = gram.copy()
        np.fill_diagonal(gram_copy, 0.0)
        max_off_diagonal = np.max(np.abs(gram_copy))

        print(f"Maximum off-diagonal value: {max_off_diagonal:.2e}")
        print(f"Theory predicts: < 1e-10")

        # FALSIFIABLE test
        tolerance = 1e-10
        assert max_off_diagonal < tolerance, \
            f"Kernel basis not orthogonal! Max inner product = {max_off_diagonal:.2e}"

        print("✓ Kernel basis is orthogonal within tolerance")

    def test_quotient_isometry_discovery(self):
        """
        Test: Discover if quotient metric equals identity.
        Theory predicts: Q = I₂ for isometric embedding.
        """
        pi = create_standard_surjection()
        Q = compute_quotient_metric(pi)

        print(f"\n=== Quotient Isometry Discovery ===")
        print(f"Quotient metric Q = π π†:")
        print(Q)

        # Expected: identity matrix
        I = np.eye(2, dtype=np.complex128)
        error = np.linalg.norm(Q - I, ord='fro')

        print(f"Expected (identity):")
        print(I)
        print(f"Frobenius norm error: {error:.2e}")
        print(f"Theory predicts error: < 1e-10")

        # FALSIFIABLE test
        tolerance = 1e-10
        assert error < tolerance, \
            f"Quotient metric not identity! Error = {error:.2e}"

        print("✓ Quotient metric is identity within tolerance")

    def test_kernel_null_space_property(self):
        """
        Test: Verify kernel vectors satisfy π·v = 0.
        Theory predicts: kernel is exact null space.
        """
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        kernel = code.kernel

        if kernel.shape[1] == 0:
            print("Empty kernel - null space property trivially satisfied")
            return

        # Compute π·K where K is kernel basis
        projection_error = compute_projection_error(pi, kernel)

        print(f"\n=== Kernel Null Space Property ===")
        print(f"Maximum |π·v| for v in kernel: {projection_error:.2e}")
        print(f"Theory predicts: < 1e-14 (SVD tolerance)")

        # FALSIFIABLE test
        tolerance = 1e-13  # Allow 10× SVD tolerance
        assert projection_error < tolerance, \
            f"Kernel not in null space! Max projection = {projection_error:.2e}"

        print("✓ Kernel vectors are in null space")

    def test_singular_value_gap(self):
        """
        Test: Analyze singular value spectrum for rank deficiency.
        Theory predicts: clear gap between non-zero and zero singular values.
        """
        pi = create_standard_surjection()
        _, s = compute_kernel_via_svd(pi)

        analysis = analyze_singular_spectrum(s)

        print(f"\n=== Singular Value Spectrum Analysis ===")
        print(f"Singular values: {s}")
        print(f"Number of non-zero values: {analysis['n_nonzero']}")
        print(f"Number of zero values: {analysis['n_zero']}")
        print(f"Largest singular value: {analysis['largest']:.6f}")
        print(f"Smallest non-zero: {analysis['smallest_nonzero']:.2e}")

        # Theory predicts rank = 2, so 2 non-zero singular values
        theory_rank = 2
        discovered_rank = analysis['n_nonzero']

        print(f"\nTheory predicts rank: {theory_rank}")
        print(f"Discovered rank: {discovered_rank}")

        if discovered_rank == theory_rank:
            print("✓ Rank matches theory prediction")
        else:
            print(f"✗ Rank mismatch: {discovered_rank} ≠ {theory_rank}")

    def test_perturbed_surjection_changes_kernel(self):
        """
        Negative control: Perturbing surjection should change kernel.
        This validates our measurement methodology.
        """
        pi = create_standard_surjection()
        code_original = SurjectionCode(pi)
        kernel_original = code_original.kernel

        # Perturb surjection
        np.random.seed(42)
        perturbation = 0.1 * np.random.randn(2, 4) + 0.1j * np.random.randn(2, 4)
        pi_perturbed = pi + perturbation

        code_perturbed = SurjectionCode(pi_perturbed)
        kernel_perturbed = code_perturbed.kernel

        print(f"\n=== Perturbation Test (Negative Control) ===")
        print(f"Perturbation magnitude: 0.1")
        print(f"Original kernel dim: {kernel_original.shape[1]}")
        print(f"Perturbed kernel dim: {kernel_perturbed.shape[1]}")

        # Kernel should change (basis vectors differ)
        if kernel_original.shape == kernel_perturbed.shape and kernel_original.shape[1] > 0:
            # Compare subspaces via principal angles
            # Project one basis onto the other
            projection = kernel_original.conj().T @ kernel_perturbed
            principal_angles = np.arccos(np.clip(np.abs(np.linalg.svd(projection)[1]), 0, 1))

            max_angle = np.max(principal_angles)
            print(f"Maximum principal angle between kernels: {np.degrees(max_angle):.2f}°")

            # Kernels should differ
            assert max_angle > 0.01, "Perturbed kernel too similar to original!"
            print("✓ Perturbation successfully changed kernel")

    def test_full_rank_boundary_case(self):
        """
        Boundary case: Full rank surjection should have trivial kernel.
        Tests edge case handling.
        """
        # Create full rank 2×2 square matrix
        pi_square = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.complex128)

        code = SurjectionCode(pi_square)

        print(f"\n=== Full Rank Boundary Case ===")
        print(f"Matrix shape: {pi_square.shape}")
        print(f"Discovered kernel dimension: {code.kernel_dim}")

        # Full rank square matrix has trivial kernel
        assert code.kernel_dim == 0, "Full rank matrix should have zero kernel dimension"
        print("✓ Full rank matrix has trivial kernel")

    def test_code_distance_computation(self):
        """
        Test: Compute code distance from kernel weight.
        Theory predicts: d=1 for this pedagogical example (no error correction).
        """
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        discovered_distance = code.compute_code_distance()

        print(f"\n=== Code Distance Discovery ===")
        print(f"Discovered code distance: {discovered_distance}")
        print(f"Theory predicts: d=1 (no error correction)")

        # Structure test (distance must be positive for non-trivial code)
        if code.kernel_dim > 0:
            assert discovered_distance > 0, "Code distance must be positive"

        # Document theory comparison
        theory_distance = 1
        if discovered_distance == theory_distance:
            print("✓ Distance matches theory prediction")
        else:
            print(f"✗ Distance differs from theory: {discovered_distance} ≠ {theory_distance}")


class TestFalsificationCriteria:
    """
    Tests specifically designed to falsify the theory.
    If these fail, the theory is WRONG.
    """

    def test_falsification_kernel_dimension(self):
        """
        FALSIFICATION: If kernel dimension ≠ 2, theory is wrong.
        """
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        discovered = code.kernel_dim
        theory = 2

        # This is the KEY falsification test
        assert discovered == theory, \
            f"THEORY FALSIFIED: Kernel dimension {discovered} ≠ {theory}"

    def test_falsification_orthogonality(self):
        """
        FALSIFICATION: If kernel basis not orthogonal, SVD theory is wrong.
        """
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        ortho_error = code.verify_orthogonality()

        # Theory requires orthogonal basis from SVD
        assert ortho_error < 1e-10, \
            f"THEORY FALSIFIED: Kernel basis not orthogonal, error = {ortho_error:.2e}"

    def test_falsification_quotient_isometry(self):
        """
        FALSIFICATION: If Q ≠ I, the quotient space theory is wrong.
        """
        pi = create_standard_surjection()
        Q = compute_quotient_metric(pi)

        I = np.eye(2)
        error = np.linalg.norm(Q - I, ord='fro')

        # Theory requires isometry
        assert error < 1e-10, \
            f"THEORY FALSIFIED: Quotient not isometric, error = {error:.2e}"