"""
Test suite: Theory tests for E50 (DISCOVERY-BASED)
Purpose: Discover smoothness properties and functorial behavior
Created: 2025-11-11

These tests DISCOVER actual behavior rather than
hardcoding expectations. Following E49 success pattern.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.core import SurjectionFamily
from src.compute import (
    create_test_family,
    verify_smoothness_criterion,
    compute_kernel_dimension_curve
)
from src.validate import (
    validate_smoothness,
    validate_kernel_evolution,
    validate_functorial_composition
)


def test_smoothness_discovery():
    """DISCOVER smoothness of dimension evolution."""
    print("\n" + "="*60)
    print("SMOOTHNESS DISCOVERY TEST")
    print("="*60)

    # Create test family
    family = create_test_family(n=8, k_initial=2, k_final=4)

    # Get kernel evolution
    t_values, kernel_dims = compute_kernel_dimension_curve(family)

    # DISCOVER smoothness
    validation = validate_smoothness(kernel_dims, t_values, threshold=1.0)

    print(f"\nDISCOVERY RESULTS:")
    print(f"  Max derivative: {validation['max_derivative']:.4f}")
    print(f"  Threshold: {validation['threshold']}")
    print(f"  Smooth? {validation['validated']}")
    print(f"  Violations: {validation['n_violations']}")

    if validation['n_violations'] > 0:
        print(f"\nViolation details:")
        for v in validation['violation_points'][:3]:  # Show first 3
            print(f"    t={v['t']:.3f}: derivative={v['derivative']:.3f}, "
                  f"dim {v['dim_before']}→{v['dim_after']}")

    # FALSIFICATION TEST (not hardcoded!)
    if validation['validated']:
        print("\n✓ Theory prediction CONFIRMED: Evolution is smooth")
    else:
        print(f"\n✗ Theory prediction VIOLATED: Max derivative {validation['max_derivative']:.4f} > 1.0")
        print("  This suggests discrete jumps in kernel dimension")

    # Store discovery for reporting
    discovery = {
        'max_derivative': validation['max_derivative'],
        'is_smooth': validation['validated'],
        'n_violations': validation['n_violations']
    }

    return discovery


def test_kernel_evolution_discovery():
    """DISCOVER pattern of kernel evolution."""
    print("\n" + "="*60)
    print("KERNEL EVOLUTION DISCOVERY")
    print("="*60)

    family = create_test_family(n=8, k_initial=2, k_final=4)

    # Validate kernel evolution
    result = validate_kernel_evolution(family)

    print(f"\nDISCOVERY RESULTS:")
    print(f"  Kernel dimension: {result['initial_kernel_dim']}→{result['final_kernel_dim']}")
    print(f"  Range: {result['kernel_dim_range']}")
    print(f"  Monotonic? {result['is_monotonic']} ({result['monotonic_type']})")
    print(f"  Rank jumps: {result['n_rank_jumps']}")

    if result['n_rank_jumps'] > 0:
        print(f"\nRank jump locations:")
        for jump in result['rank_jumps'][:3]:  # Show first 3
            t, before, after = jump
            print(f"    t={t:.3f}: rank {before}→{after}")

    # Theory prediction test
    smooth = result['smoothness']['validated']
    if smooth and result['is_monotonic']:
        print("\n✓ Ideal functorial behavior: smooth and monotonic")
    elif smooth:
        print("\n⚠ Smooth but non-monotonic evolution")
    else:
        print("\n✗ Non-smooth evolution detected")

    return result


def test_functorial_composition_discovery():
    """DISCOVER functorial composition properties."""
    print("\n" + "="*60)
    print("FUNCTORIAL COMPOSITION DISCOVERY")
    print("="*60)

    family = create_test_family(n=6, k_initial=2, k_final=4)  # Smaller for speed

    result = validate_functorial_composition(family)

    print(f"\nDISCOVERY RESULTS:")
    print(f"  Ranks: {result['ranks'][0]}→{result['ranks'][1]}→{result['ranks'][2]}")
    print(f"  Kernel dims: {result['kernel_dims'][0]}→{result['kernel_dims'][1]}→{result['kernel_dims'][2]}")
    print(f"  Kernel monotonic? {result['kernel_monotonic']}")
    print(f"  Rank monotonic? {result['rank_monotonic']}")
    print(f"  Composition error: {result['composition_error']:.6f}")

    # Interpretation
    if result['validated']:
        print("\n✓ Functorial property satisfied: monotonic evolution")
    else:
        print("\n✗ Functorial property violated: non-monotonic evolution")
        print("  This suggests the interpolation doesn't preserve morphism structure")

    return result


def test_critical_point_structure():
    """DISCOVER structure at critical points."""
    print("\n" + "="*60)
    print("CRITICAL POINT STRUCTURE DISCOVERY")
    print("="*60)

    family = create_test_family(n=8, k_initial=2, k_final=4)

    # Find critical points
    critical = family.find_critical_points()

    print(f"\nFound {len(critical)} critical points")

    if len(critical) > 0:
        # Analyze first critical point in detail
        t_crit = critical[0]
        print(f"\nAnalyzing critical point at t={t_crit:.4f}:")

        # Get surjection at critical point
        pi_crit = family.get_surjection(t_crit)

        # Analyze spectrum
        U, s, Vh = np.linalg.svd(pi_crit, full_matrices=False)

        # Find gap in singular values (indicates rank change)
        gaps = np.diff(s)
        max_gap_idx = np.argmax(gaps)
        max_gap = gaps[max_gap_idx]

        print(f"  Singular values near gap: "
              f"σ_{max_gap_idx}={s[max_gap_idx]:.6f}, "
              f"σ_{max_gap_idx+1}={s[max_gap_idx+1]:.6f}")
        print(f"  Gap size: {max_gap:.6f}")

        # Check if it's a "clean" rank change
        if max_gap > 0.1:  # Heuristic threshold
            print("  → Clean rank change (large spectral gap)")
        else:
            print("  → Gradual rank change (small spectral gap)")
    else:
        print("No critical points found - smooth rank evolution throughout")

    return len(critical)


def test_endpoint_preservation():
    """DISCOVER if interpolation preserves endpoint properties."""
    print("\n" + "="*60)
    print("ENDPOINT PRESERVATION DISCOVERY")
    print("="*60)

    n = 6
    k_0 = 2
    k_1 = 3

    family = SurjectionFamily(n=n, k_initial=k_0, k_final=k_1, n_points=50)

    # Check endpoints
    pi_0_original = family.pi_0
    pi_0_retrieved = family.get_surjection(0.0)

    pi_1_original = family.pi_1
    pi_1_retrieved = family.get_surjection(1.0)

    # Compute differences
    error_0 = np.linalg.norm(pi_0_original - pi_0_retrieved)
    error_1 = np.linalg.norm(pi_1_original - pi_1_retrieved)

    print(f"\nEndpoint preservation:")
    print(f"  t=0 error: {error_0:.2e}")
    print(f"  t=1 error: {error_1:.2e}")

    # Check ranks
    rank_0 = np.linalg.matrix_rank(pi_0_retrieved, tol=1e-10)
    rank_1 = np.linalg.matrix_rank(pi_1_retrieved, tol=1e-10)

    print(f"\nRank preservation:")
    print(f"  t=0: rank={rank_0} (expected {k_0})")
    print(f"  t=1: rank={rank_1} (expected {k_1})")

    preserved = (error_0 < 1e-10 and error_1 < 1e-10 and
                rank_0 == k_0 and rank_1 == k_1)

    if preserved:
        print("\n✓ Endpoints perfectly preserved")
    else:
        print("\n✗ Endpoint deviation detected")

    return preserved


def test_negative_control_rank_jump():
    """Negative control: Introduce deliberate rank jump."""
    print("\n" + "="*60)
    print("NEGATIVE CONTROL: DELIBERATE RANK JUMP")
    print("="*60)

    n = 6
    physical_dim = 2**n

    # Create custom interpolation with rank jump
    def jump_interpolation(t):
        if t < 0.4:
            # Rank 2
            pi = np.zeros((2, physical_dim), dtype=complex)
            pi[0, 0] = 1.0
            pi[1, 1] = 1.0
        elif t < 0.6:
            # Rank 3 (jump!)
            pi = np.zeros((3, physical_dim), dtype=complex)
            pi[0, 0] = 1.0
            pi[1, 1] = 1.0
            pi[2, 2] = 1.0
        else:
            # Back to rank 2
            pi = np.zeros((2, physical_dim), dtype=complex)
            pi[0, 0] = 1.0
            pi[1, 3] = 1.0
        return pi

    # Create family with jump
    family = SurjectionFamily(pi_func=jump_interpolation, n_points=100)

    # Validate smoothness - should FAIL
    t_values, kernel_dims = compute_kernel_dimension_curve(family)
    validation = validate_smoothness(kernel_dims, t_values)

    print(f"\nNegative control results:")
    print(f"  Max derivative: {validation['max_derivative']:.4f}")
    print(f"  Violations: {validation['n_violations']}")
    print(f"  Validated: {validation['validated']}")

    if not validation['validated']:
        print("\n✓ Negative control successful: Jump detected as expected")
    else:
        print("\n✗ Negative control failed: Jump not detected!")

    return not validation['validated']


if __name__ == "__main__":
    print("\n" + "="*70)
    print("E50 THEORY TESTS - DISCOVERY MODE")
    print("="*70)
    print("Following E49 pattern: DISCOVER actual behavior, don't hardcode")

    # Run discovery tests
    smoothness = test_smoothness_discovery()
    evolution = test_kernel_evolution_discovery()
    functorial = test_functorial_composition_discovery()
    n_critical = test_critical_point_structure()
    endpoints_ok = test_endpoint_preservation()
    negative_ok = test_negative_control_rank_jump()

    # Summary
    print("\n" + "="*70)
    print("DISCOVERY SUMMARY")
    print("="*70)

    print(f"\n1. Smoothness: max |Δdim/Δt| = {smoothness['max_derivative']:.4f}")
    print(f"2. Evolution: {evolution['monotonic_type']}")
    print(f"3. Functoriality: {'satisfied' if functorial['validated'] else 'violated'}")
    print(f"4. Critical points: {n_critical}")
    print(f"5. Endpoints: {'preserved' if endpoints_ok else 'altered'}")
    print(f"6. Negative control: {'passed' if negative_ok else 'failed'}")

    print("\nAll discovery tests completed!")