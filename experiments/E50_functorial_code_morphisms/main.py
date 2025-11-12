#!/usr/bin/env python3
"""
E50: Functorial Code Morphisms - Main Execution Script
Purpose: Run complete experiment for continuous code families
Created: 2025-11-11

This script executes the full E50 experiment, generating
all data, validations, and visualizations.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import SurjectionFamily
from src.compute import (
    compute_kernel_dimension_curve,
    compute_distance_evolution,
    find_rank_jumps,
    verify_smoothness_criterion
)
from src.validate import (
    validate_kernel_evolution,
    validate_endpoint_consistency,
    validate_functorial_composition,
    validate_interpolation_bounds,
    validate_spectral_evolution,
    create_validation_summary
)
from src.visualize import create_all_visualizations


def run_experiment(n: int = 8,
                  k_initial: int = 2,
                  k_final: int = 4,
                  n_points: int = 100,
                  output_dir: str = None):
    """
    Run complete E50 experiment.

    Args:
        n: Number of physical qubits
        k_initial: Initial logical dimension
        k_final: Final logical dimension
        n_points: Number of interpolation points
        output_dir: Output directory (auto-generated if None)

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print("E50: FUNCTORIAL CODE MORPHISMS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Physical qubits: n = {n}")
    print(f"  Logical dimension: k = {k_initial} → {k_final}")
    print(f"  Interpolation points: {n_points}")

    start_time = time.time()

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", f"session_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # ========================================
    # Phase 1: Create Code Family
    # ========================================
    print("\n" + "-"*50)
    print("PHASE 1: Creating Code Family")
    print("-"*50)

    family = SurjectionFamily(
        n=n,
        k_initial=k_initial,
        k_final=k_final,
        n_points=n_points
    )

    print(f"✓ Created family with {n_points} interpolation points")

    # ========================================
    # Phase 2: Compute Evolution
    # ========================================
    print("\n" + "-"*50)
    print("PHASE 2: Computing Evolution")
    print("-"*50)

    # Kernel dimension evolution
    print("Computing kernel dimension evolution...")
    t_values, kernel_dims = compute_kernel_dimension_curve(family)
    logical_dims = family.compute_dimension_evolution()

    # Critical points
    critical_points = family.find_critical_points()
    print(f"✓ Found {len(critical_points)} critical points")

    # Smoothness analysis
    is_smooth, max_deriv, derivatives = verify_smoothness_criterion(
        t_values, kernel_dims, threshold=1.0
    )
    print(f"✓ Smoothness: max |Δdim/Δt| = {max_deriv:.4f} ({'smooth' if is_smooth else 'non-smooth'})")

    # Singular value evolution
    print("Computing singular value evolution...")
    sv_evolution = family.get_singular_value_evolution()
    print(f"✓ Computed {sv_evolution.shape[0]} × {sv_evolution.shape[1]} singular value matrix")

    # Code distance evolution (subset for speed)
    print("Computing code distance evolution (subset)...")
    distance_sample_indices = np.linspace(0, n_points-1, min(10, n_points), dtype=int)
    distance_samples = []
    for idx in distance_sample_indices:
        t = t_values[idx]
        params = family.get_code_parameters(t)
        distance_samples.append((t, params['d']))
    print(f"✓ Computed distance at {len(distance_samples)} points")

    # ========================================
    # Phase 3: Validation Suite
    # ========================================
    print("\n" + "-"*50)
    print("PHASE 3: Running Validation Suite")
    print("-"*50)

    validation_tests = []

    # Test 1: Kernel evolution
    print("\n1. Validating kernel evolution...")
    result = validate_kernel_evolution(family)
    validation_tests.append(result)
    print(f"   Result: {'PASSED' if result['validated'] else 'FAILED'}")
    print(f"   {result['discovery']}")

    # Test 2: Endpoint consistency
    print("\n2. Validating endpoint consistency...")
    result = validate_endpoint_consistency(family)
    validation_tests.append(result)
    print(f"   Result: {'PASSED' if result['validated'] else 'FAILED'}")
    print(f"   {result['discovery']}")

    # Test 3: Functorial composition
    print("\n3. Validating functorial composition...")
    result = validate_functorial_composition(family)
    validation_tests.append(result)
    print(f"   Result: {'PASSED' if result['validated'] else 'FAILED'}")
    print(f"   {result['discovery']}")

    # Test 4: Interpolation bounds
    print("\n4. Validating interpolation bounds...")
    result = validate_interpolation_bounds(family, n_samples=20)
    validation_tests.append(result)
    print(f"   Result: {'PASSED' if result['validated'] else 'FAILED'}")
    print(f"   {result['discovery']}")

    # Test 5: Spectral evolution
    print("\n5. Validating spectral evolution...")
    result = validate_spectral_evolution(family, n_samples=10)
    validation_tests.append(result)
    print(f"   Result: {'PASSED' if result['validated'] else 'FAILED'}")
    print(f"   {result['discovery']}")

    # Create validation summary
    validation_summary = create_validation_summary(validation_tests)

    print(f"\nValidation Summary: {validation_summary['n_passed']}/{validation_summary['n_tests']} tests passed")

    # ========================================
    # Phase 4: Generate Outputs
    # ========================================
    print("\n" + "-"*50)
    print("PHASE 4: Generating Outputs")
    print("-"*50)

    # Save CSV data
    csv_path = os.path.join(output_dir, "kernel_evolution.csv")
    with open(csv_path, 'w') as f:
        f.write("t,kernel_dim,logical_dim,rank\n")
        for t, kd, ld in zip(t_values, kernel_dims, logical_dims):
            f.write(f"{t:.6f},{kd},{ld},{ld}\n")
    print(f"✓ Saved: kernel_evolution.csv")

    # Save singular values
    sv_path = os.path.join(output_dir, "singular_values.npy")
    np.save(sv_path, sv_evolution)
    print(f"✓ Saved: singular_values.npy")

    # Save code parameters
    params_path = os.path.join(output_dir, "code_parameters.csv")
    with open(params_path, 'w') as f:
        f.write("t,n,k,d,kernel_dim\n")
        for t, d in distance_samples:
            idx = np.argmin(np.abs(t_values - t))
            f.write(f"{t:.6f},{n},{logical_dims[idx]},{d},{kernel_dims[idx]}\n")
    print(f"✓ Saved: code_parameters.csv")

    # Save smoothness analysis
    smoothness_path = os.path.join(output_dir, "smoothness_analysis.json")
    smoothness_data = {
        'max_derivative': float(max_deriv),
        'is_smooth': bool(is_smooth),
        'threshold': 1.0,
        'n_critical_points': len(critical_points),
        'critical_points': [float(cp) for cp in critical_points],
        'discovery': f"Max |Δdim/Δt| = {max_deriv:.4f}"
    }
    with open(smoothness_path, 'w') as f:
        json.dump(smoothness_data, f, indent=2)
    print(f"✓ Saved: smoothness_analysis.json")

    # Save validation report
    validation_path = os.path.join(output_dir, "validation_report.json")
    with open(validation_path, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    print(f"✓ Saved: validation_report.json")

    # Save experiment summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    experiment_summary = {
        'experiment': 'E50_functorial_code_morphisms',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n': n,
            'k_initial': k_initial,
            'k_final': k_final,
            'n_points': n_points,
            'physical_dim': 2**n
        },
        'results': {
            'kernel_dim_range': [int(np.min(kernel_dims)), int(np.max(kernel_dims))],
            'logical_dim_range': [int(np.min(logical_dims)), int(np.max(logical_dims))],
            'n_critical_points': len(critical_points),
            'max_derivative': float(max_deriv),
            'is_smooth': bool(is_smooth),
            'validation_passed': validation_summary['n_passed'],
            'validation_total': validation_summary['n_tests']
        },
        'discoveries': validation_summary['discoveries'],
        'output_files': [
            'kernel_evolution.csv',
            'singular_values.npy',
            'code_parameters.csv',
            'smoothness_analysis.json',
            'validation_report.json'
        ]
    }
    with open(summary_path, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"✓ Saved: experiment_summary.json")

    # ========================================
    # Phase 5: Create Visualizations
    # ========================================
    print("\n" + "-"*50)
    print("PHASE 5: Creating Visualizations")
    print("-"*50)

    viz_files = create_all_visualizations(
        family,
        output_dir=output_dir,
        validation_results=validation_summary
    )

    print(f"✓ Created {len(viz_files)} visualization files")

    # ========================================
    # Phase 6: Final Summary
    # ========================================
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    print(f"\nKey Discoveries:")
    print(f"  • Kernel dimension: {np.min(kernel_dims):.0f} → {np.max(kernel_dims):.0f}")
    print(f"  • Max |Δdim/Δt| = {max_deriv:.4f} ({'smooth' if is_smooth else 'non-smooth'})")
    print(f"  • Critical points: {len(critical_points)}")
    print(f"  • Validation: {validation_summary['n_passed']}/{validation_summary['n_tests']} passed")

    print(f"\nExecution time: {elapsed:.2f} seconds")
    print(f"Output directory: {output_dir}")

    return experiment_summary


if __name__ == "__main__":
    # Run with default parameters
    results = run_experiment(
        n=8,           # 8 qubits
        k_initial=2,   # Start with k=2
        k_final=4,     # End with k=4
        n_points=100   # 100 interpolation points
    )

    print("\n" + "="*70)
    print("E50 EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)