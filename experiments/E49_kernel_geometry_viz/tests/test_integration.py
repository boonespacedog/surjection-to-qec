"""
📄 File: test_integration.py
Purpose: End-to-end workflow tests for E49
Created: November 11, 2025
Used by: pytest test suite
"""

import numpy as np
import pytest
import sys
import os
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import SurjectionCode
from compute import create_standard_surjection
from validate import run_full_validation, save_validation_report
from visualize import save_all_figures


class TestE49Workflow:
    """Test complete E49 experimental workflow."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: Load → Compute → Validate → Save."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup paths
            results_dir = os.path.join(tmpdir, "results")
            figures_dir = os.path.join(tmpdir, "figures")
            data_dir = os.path.join(tmpdir, "data")

            os.makedirs(results_dir)
            os.makedirs(figures_dir)
            os.makedirs(data_dir)

            # Step 1: Create surjection
            pi = create_standard_surjection()
            assert pi.shape == (2, 4)

            # Step 2: Create code object
            code = SurjectionCode(pi)
            assert code.physical_dim == 4
            assert code.logical_dim == 2

            # Step 3: Run validation
            validation_report = run_full_validation(code)
            assert 'overall_pass' in validation_report
            assert 'falsification_result' in validation_report

            # Step 4: Save validation report
            validation_path = os.path.join(results_dir, "validation_report.json")
            save_validation_report(validation_report, validation_path)
            assert os.path.exists(validation_path)

            # Step 5: Generate and save figures
            figures = save_all_figures(code, figures_dir)
            assert len(figures) >= 4  # At least 4 figures

            # Verify all figure files exist
            for name, path in figures.items():
                assert os.path.exists(path), f"Figure {name} not saved"

            # Step 6: Save main results
            results = {
                'experiment_id': 'E49',
                'surjection_matrix': pi.tolist(),
                'kernel_dimension': code.kernel_dim,
                'singular_values': code.singular_values.tolist(),
                'theory_predictions_met': validation_report['overall_pass']
            }

            results_path = os.path.join(results_dir, "e49_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            assert os.path.exists(results_path)

            print(f"\n✓ End-to-end workflow completed successfully")
            print(f"  Results saved to: {tmpdir}")

    def test_reproducibility(self):
        """Test that same input produces same output."""
        pi = create_standard_surjection()

        # Create two independent code objects
        code1 = SurjectionCode(pi.copy())
        code2 = SurjectionCode(pi.copy())

        # Kernel should be identical
        assert np.allclose(code1.kernel, code2.kernel)

        # Quotient metric should be identical
        assert np.allclose(code1.quotient_metric, code2.quotient_metric)

        # Singular values should be identical
        assert np.allclose(code1.singular_values, code2.singular_values)

        print("\n✓ Reproducibility test passed")

    def test_performance(self):
        """Test that computation completes in reasonable time."""
        import time

        pi = create_standard_surjection()

        start = time.time()

        # Full computation
        code = SurjectionCode(pi)
        _ = code.kernel
        _ = code.quotient_metric
        _ = code.singular_values
        _ = code.compute_code_distance()

        # Validation
        report = run_full_validation(code)

        elapsed = time.time() - start

        print(f"\n✓ Performance test: {elapsed:.3f} seconds")

        # Should complete in < 1 second for this trivial example
        assert elapsed < 1.0, f"Computation took too long: {elapsed:.3f}s"

    def test_output_schemas(self):
        """Test that all outputs conform to documented schemas."""
        pi = create_standard_surjection()
        code = SurjectionCode(pi)

        # Test validation report schema
        report = run_full_validation(code)

        assert 'experiment_id' in report
        assert 'timestamp' in report
        assert 'predictions' in report
        assert isinstance(report['predictions'], list)
        assert 'overall_pass' in report
        assert 'falsification_result' in report

        # Test each prediction
        for pred in report['predictions']:
            assert 'test' in pred
            assert 'theory_predicts' in pred
            assert 'discovered' in pred
            assert 'passed' in pred
            assert 'error' in pred
            assert 'tolerance' in pred

        # Test results schema
        results = code.get_state_dict()

        assert 'surjection_matrix' in results
        assert 'physical_dim' in results
        assert 'logical_dim' in results
        assert 'kernel_dim' in results
        assert 'singular_values' in results
        assert 'quotient_metric' in results
        assert 'code_distance' in results
        assert 'orthogonality_error' in results

        print("\n✓ Output schemas validated")

    def test_error_handling(self):
        """Test graceful handling of edge cases."""
        # Test with non-surjective matrix
        pi_bad = np.array([[1, 0, 1, 0], [2, 0, 2, 0]], dtype=np.complex128)

        with pytest.raises(ValueError, match="not full row rank"):
            SurjectionCode(pi_bad)

        # Test with wrong shape input
        with pytest.raises(TypeError):
            SurjectionCode("not a matrix")

        print("\n✓ Error handling test passed")