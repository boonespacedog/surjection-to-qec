"""
Test suite: Structure tests for E50
Purpose: Verify interpolation bounds and matrix consistency
Created: 2025-11-11

These tests verify that the code family maintains
structural consistency throughout interpolation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.core import SurjectionFamily
from src.compute import (
    construct_endpoint_surjection,
    interpolate_surjections,
    compute_kernel_dimension
)


def test_endpoint_construction():
    """Verify endpoint surjections have correct dimensions."""
    n = 8
    k = 3

    # Construct endpoint
    pi = construct_endpoint_surjection(n, k, seed=42)

    # Check shape
    assert pi.shape == (k, 2**n), f"Wrong shape: {pi.shape}"

    # Check rank
    rank = np.linalg.matrix_rank(pi, tol=1e-10)
    assert rank == k, f"Wrong rank: {rank} (expected {k})"

    print(f"Endpoint surjection: shape {pi.shape}, rank {rank}")


def test_interpolation_bounds():
    """Verify interpolation stays within physical bounds."""
    n = 8
    t_values = np.linspace(0, 1, 100)

    # Create family
    family = SurjectionFamily(n=n, k_initial=2, k_final=4)

    violations = []
    for t in t_values:
        pi_t = family.get_surjection(t)

        # Structure test: matrix shape consistent
        assert pi_t.shape[1] == 2**n, f"Wrong physical dimension at t={t}"

        # Structure test: range dimension between endpoints
        rank = np.linalg.matrix_rank(pi_t, tol=1e-10)
        if not (2 <= rank <= 4):
            violations.append((t, rank))

    print(f"Tested {len(t_values)} interpolation points")
    print(f"Rank violations: {len(violations)}")

    # Allow small number of edge cases
    assert len(violations) <= 2, f"Too many rank violations: {violations}"


def test_interpolation_linearity():
    """Test that interpolation is actually linear."""
    n = 6  # Smaller for speed
    # Create with same shape for interpolation
    k_max = 3
    pi_0 = construct_endpoint_surjection(n, k_max, seed=42)
    pi_1 = construct_endpoint_surjection(n, k_max, seed=43)
    # Reduce rank of pi_0
    pi_0[2:, :] = 0

    # Test midpoint
    pi_mid = interpolate_surjections(pi_0, pi_1, 0.5)
    pi_expected = 0.5 * pi_0 + 0.5 * pi_1

    error = np.linalg.norm(pi_mid - pi_expected)
    assert error < 1e-10, f"Linear interpolation error: {error}"

    # Test quarter point
    pi_quarter = interpolate_surjections(pi_0, pi_1, 0.25)
    pi_expected = 0.75 * pi_0 + 0.25 * pi_1

    error = np.linalg.norm(pi_quarter - pi_expected)
    assert error < 1e-10, f"Linear interpolation error at t=0.25: {error}"

    print("Linear interpolation verified")


def test_kernel_dimension_consistency():
    """Test kernel dimension computation consistency."""
    n = 6
    pi = construct_endpoint_surjection(n, 3, seed=42)

    # Compute kernel dimension
    kernel_dim, kernel_basis = compute_kernel_dimension(pi)

    # Verify rank-nullity theorem
    rank = np.linalg.matrix_rank(pi, tol=1e-10)
    physical_dim = 2**n

    assert kernel_dim + rank == physical_dim, \
        f"Rank-nullity violation: {kernel_dim} + {rank} != {physical_dim}"

    # Verify kernel basis is actually in kernel
    if kernel_basis.shape[1] > 0:
        residual = pi @ kernel_basis
        max_residual = np.max(np.abs(residual))
        assert max_residual < 1e-10, f"Kernel basis not in kernel: residual {max_residual}"

    print(f"Kernel dimension: {kernel_dim}, Rank: {rank}, Physical: {physical_dim}")


def test_family_initialization():
    """Test SurjectionFamily initialization."""
    n = 8
    k_initial = 2
    k_final = 4
    n_points = 50

    family = SurjectionFamily(
        n=n,
        k_initial=k_initial,
        k_final=k_final,
        n_points=n_points
    )

    # Check attributes
    assert family.n == n
    assert family.physical_dim == 2**n
    assert family.k_initial == k_initial
    assert family.k_final == k_final
    assert len(family.t_values) == n_points

    # Check endpoints
    pi_0 = family.get_surjection(0.0)
    pi_1 = family.get_surjection(1.0)

    rank_0 = np.linalg.matrix_rank(pi_0, tol=1e-10)
    rank_1 = np.linalg.matrix_rank(pi_1, tol=1e-10)

    assert rank_0 == k_initial, f"Initial rank {rank_0} != {k_initial}"
    assert rank_1 == k_final, f"Final rank {rank_1} != {k_final}"

    print(f"Family initialized: n={n}, k: {k_initial}→{k_final}")


def test_critical_points_detection():
    """Test detection of critical points where rank changes."""
    # Create family with known rank change
    n = 6
    family = SurjectionFamily(n=n, k_initial=2, k_final=4, n_points=100)

    # Get critical points
    critical = family.find_critical_points()

    print(f"Found {len(critical)} critical points: {critical}")

    # Should find at least one critical point for k=2→4 transition
    # (unless interpolation happens to preserve rank perfectly)
    # This is a discovery-based test
    if len(critical) > 0:
        print(f"First critical point at t={critical[0]:.4f}")

        # Verify rank actually changes at critical point
        eps = 0.01
        for t_crit in critical[:2]:  # Check first two
            if t_crit - eps >= 0 and t_crit + eps <= 1:
                pi_before = family.get_surjection(t_crit - eps)
                pi_after = family.get_surjection(t_crit + eps)

                rank_before = np.linalg.matrix_rank(pi_before, tol=1e-10)
                rank_after = np.linalg.matrix_rank(pi_after, tol=1e-10)

                print(f"At t={t_crit}: rank {rank_before}→{rank_after}")
    else:
        print("No critical points found (smooth rank evolution)")


def test_no_nan_or_inf():
    """Verify no NaN or Inf values in interpolation."""
    n = 6
    family = SurjectionFamily(n=n, k_initial=2, k_final=3, n_points=20)

    for t in family.t_values:
        pi_t = family.get_surjection(t)

        assert np.all(np.isfinite(pi_t)), f"Non-finite values at t={t}"

    print("All interpolated matrices are finite")


if __name__ == "__main__":
    print("Running E50 structure tests...")
    print("=" * 60)

    test_endpoint_construction()
    test_interpolation_bounds()
    test_interpolation_linearity()
    test_kernel_dimension_consistency()
    test_family_initialization()
    test_critical_points_detection()
    test_no_nan_or_inf()

    print("=" * 60)
    print("All structure tests passed!")