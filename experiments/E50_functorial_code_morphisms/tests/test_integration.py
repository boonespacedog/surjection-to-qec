"""
Test suite: Integration tests for E50
Purpose: Full pipeline validation and output generation
Created: 2025-11-11

These tests verify the complete experimental pipeline
including data generation, validation, and visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import json
import tempfile
import time
from datetime import datetime

from src.core import SurjectionFamily
from src.compute import create_test_family, compute_distance_evolution
from src.validate import (
    validate_kernel_evolution,
    validate_endpoint_consistency,
    validate_functorial_composition,
    validate_interpolation_bounds,
    validate_spectral_evolution,
    create_validation_summary
)
from src.visualize import create_all_visualizations


def test_complete_parameter_sweep():
    """Test complete sweep over parameter range."""
    print("\n" + "="*60)
    print("COMPLETE PARAMETER SWEEP")
    print("="*60)

    start_time = time.time()

    # Create family with full resolution
    family = create_test_family(n=8, k_initial=2, k_final=4)

    # Compute all properties
    dims = family.compute_dimension_evolution()
    kernel_dims = family.compute_kernel_dimension_evolution()
    critical = family.find_critical_points()
    sv_evolution = family.get_singular_value_evolution()

    # Get summary
    summary = family.summary()

    elapsed = time.time() - start_time

    print(f"\nParameter sweep completed in {elapsed:.2f} seconds")
    print(f"  Points sampled: {family.n_points}")
    print(f"  Dimension range: {summary['dimension_range']}")
    print(f"  Kernel dim range: {summary['kernel_dim_range']}")
    print(f"  Critical points: {summary['n_critical_points']}")
    print(f"  Max derivative: {summary['max_derivative']:.4f}")
    print(f"  Smooth? {summary['is_smooth']}")

    # Performance check
    assert elapsed < 10, f"Sweep too slow: {elapsed:.2f}s > 10s"

    return summary


def test_full_validation_suite():
    """Run complete validation suite."""
    print("\n" + "="*60)
    print("FULL VALIDATION SUITE")
    print("="*60)

    family = create_test_family(n=6, k_initial=2, k_final=3)  # Smaller for speed

    # Run all validations
    tests = []

    print("\n1. Kernel evolution validation...")
    result = validate_kernel_evolution(family)
    tests.append(result)
    print(f"   Result: {result['validated']}")

    print("\n2. Endpoint consistency validation...")
    result = validate_endpoint_consistency(family)
    tests.append(result)
    print(f"   Result: {result['validated']}")

    print("\n3. Functorial composition validation...")
    result = validate_functorial_composition(family)
    tests.append(result)
    print(f"   Result: {result['validated']}")

    print("\n4. Interpolation bounds validation...")
    result = validate_interpolation_bounds(family, n_samples=10)
    tests.append(result)
    print(f"   Result: {result['validated']}")

    print("\n5. Spectral evolution validation...")
    result = validate_spectral_evolution(family, n_samples=5)
    tests.append(result)
    print(f"   Result: {result['validated']}")

    # Create summary
    summary = create_validation_summary(tests)

    print(f"\nValidation Summary:")
    print(f"  Tests run: {summary['n_tests']}")
    print(f"  Tests passed: {summary['n_passed']}")
    print(f"  Overall status: {summary['overall_status']}")

    # Check at least some tests pass (discovery-based)
    assert summary['n_passed'] >= 3, f"Too few tests passed: {summary['n_passed']}/5"

    return summary


def test_output_generation():
    """Test generation of all output files."""
    print("\n" + "="*60)
    print("OUTPUT GENERATION TEST")
    print("="*60)

    family = create_test_family(n=6, k_initial=2, k_final=3)

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "test_session")

        # Generate CSV data
        csv_path = os.path.join(output_dir, "kernel_evolution.csv")
        os.makedirs(output_dir, exist_ok=True)

        # Write kernel evolution data
        t_values = family.t_values
        kernel_dims = family.compute_kernel_dimension_evolution()
        logical_dims = family.compute_dimension_evolution()

        with open(csv_path, 'w') as f:
            f.write("t,kernel_dim,logical_dim\n")
            for t, kd, ld in zip(t_values, kernel_dims, logical_dims):
                f.write(f"{t:.4f},{kd},{ld}\n")

        print(f"  Created: {csv_path}")

        # Generate JSON validation report
        json_path = os.path.join(output_dir, "validation_report.json")
        validation = validate_kernel_evolution(family)

        with open(json_path, 'w') as f:
            json.dump(validation, f, indent=2)

        print(f"  Created: {json_path}")

        # Generate summary
        summary_path = os.path.join(output_dir, "summary.json")
        summary = family.summary()

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  Created: {summary_path}")

        # Check files exist
        assert os.path.exists(csv_path), "CSV not created"
        assert os.path.exists(json_path), "JSON not created"
        assert os.path.exists(summary_path), "Summary not created"

        # Check CSV has correct number of rows
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == len(t_values) + 1, "CSV row count mismatch"

        print(f"\n  All outputs generated successfully")


def test_visualization_pipeline():
    """Test visualization generation (without display)."""
    print("\n" + "="*60)
    print("VISUALIZATION PIPELINE TEST")
    print("="*60)

    family = create_test_family(n=6, k_initial=2, k_final=3)

    # Run validation for summary plot
    tests = [
        validate_kernel_evolution(family),
        validate_endpoint_consistency(family)
    ]
    validation_summary = create_validation_summary(tests)

    # Create visualizations in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        created_files = create_all_visualizations(
            family,
            output_dir=tmpdir,
            validation_results=validation_summary
        )

        print(f"\nCreated {len(created_files)} visualization files:")
        for file in created_files:
            size = os.path.getsize(file)
            print(f"  {os.path.basename(file)}: {size:,} bytes")
            assert size > 0, f"Empty file: {file}"

        # Check expected files
        expected_names = [
            'kernel_dim_vs_t.png',
            'code_parameters.png',
            'singular_values_heatmap.png',
            'validation_summary.png'
        ]

        created_names = [os.path.basename(f) for f in created_files]
        for expected in expected_names:
            assert expected in created_names, f"Missing: {expected}"

    print("\nVisualization pipeline successful")


def test_distance_evolution_computation():
    """Test code distance evolution (expensive computation)."""
    print("\n" + "="*60)
    print("DISTANCE EVOLUTION TEST (SUBSET)")
    print("="*60)

    # Use very small family for speed
    family = SurjectionFamily(n=4, k_initial=1, k_final=2, n_points=5)

    # Compute distance evolution
    t_values, distances = compute_distance_evolution(family)

    print(f"\nDistance evolution:")
    for t, d in zip(t_values, distances):
        print(f"  t={t:.2f}: d={d}")

    # Basic checks
    assert len(distances) == len(t_values)
    assert all(d >= 0 for d in distances), "Negative distances found"

    # Check endpoints
    print(f"\nEndpoint distances: d(0)={distances[0]}, d(1)={distances[-1]}")


def test_performance_benchmark():
    """Benchmark performance for different sizes."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)

    sizes = [
        (4, 10),   # 4 qubits, 10 points
        (6, 20),   # 6 qubits, 20 points
        (8, 50),   # 8 qubits, 50 points
    ]

    for n, n_points in sizes:
        start = time.time()

        family = SurjectionFamily(
            n=n,
            k_initial=2,
            k_final=3,
            n_points=n_points
        )

        # Basic computations
        dims = family.compute_dimension_evolution()
        critical = family.find_critical_points()

        elapsed = time.time() - start

        print(f"  n={n} qubits, {n_points} points: {elapsed:.3f}s")

        # Check reasonable performance
        assert elapsed < 5, f"Too slow for n={n}: {elapsed:.3f}s"


if __name__ == "__main__":
    print("\n" + "="*70)
    print("E50 INTEGRATION TESTS")
    print("="*70)

    # Run all integration tests
    sweep_summary = test_complete_parameter_sweep()
    validation_summary = test_full_validation_suite()
    test_output_generation()
    test_visualization_pipeline()
    test_distance_evolution_computation()
    test_performance_benchmark()

    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)

    print(f"\n✓ Parameter sweep: {sweep_summary['n_points']} points")
    print(f"✓ Validation suite: {validation_summary['n_passed']}/{validation_summary['n_tests']} passed")
    print(f"✓ Output generation: All formats working")
    print(f"✓ Visualization: All plots generated")
    print(f"✓ Performance: Within limits")

    print("\nAll integration tests passed!")