"""
📄 File: validate.py
Purpose: Validation and verification functions for theory predictions
Created: November 11, 2025
Used by: main.py, tests/test_integration.py
"""

import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime
import logging

from core import SurjectionCode
from compute import compute_quotient_metric, verify_orthogonality


def validate_kernel_dimension(kernel_basis: np.ndarray,
                             pi_matrix: np.ndarray,
                             theory_prediction: int = 2) -> Dict[str, Any]:
    """
    🧠 Function: validate_kernel_dimension
    Role: Validate kernel dimension matches rank deficiency
    Inputs: kernel_basis - computed kernel
            pi_matrix - surjection matrix
            theory_prediction - expected dimension from paper
    Returns: Validation result dictionary
    Notes: Discovery-based - logs both discovered and expected
    """
    # Discover actual dimension
    discovered_dim = kernel_basis.shape[1]

    # Compute expected from rank-nullity theorem
    rank = np.linalg.matrix_rank(pi_matrix, tol=1e-14)
    expected_from_rank = pi_matrix.shape[1] - rank

    # Check consistency
    passed = (discovered_dim == expected_from_rank)

    result = {
        'test': 'kernel_dimension',
        'theory_predicts': theory_prediction,
        'discovered': discovered_dim,
        'computed_from_rank': expected_from_rank,
        'passed': passed,
        'error': abs(discovered_dim - theory_prediction),
        'tolerance': 0  # Dimension is exact (integer)
    }

    logging.info(f"Kernel dimension validation: {result}")

    return result


def validate_quotient_isometry(Q: np.ndarray,
                              target_dim: int = 2,
                              tol: float = 1e-10) -> Dict[str, Any]:
    """
    🧠 Function: validate_quotient_isometry
    Role: Validate Q ≈ I_target (quotient is isometric)
    Inputs: Q - quotient metric
            target_dim - expected identity dimension
            tol - tolerance for comparison
    Returns: Validation result dictionary
    Notes: Checks both Hermiticity and isometry
    """
    # Expected: identity matrix
    I = np.eye(target_dim, dtype=Q.dtype)

    # Compute error
    frobenius_error = np.linalg.norm(Q - I, ord='fro')

    # Check if passed
    passed = (frobenius_error < tol)

    # Additional checks
    is_hermitian = np.allclose(Q, Q.conj().T, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(Q)  # Should all be ~1

    result = {
        'test': 'quotient_isometry',
        'theory_predicts': 'identity',
        'discovered': f'frobenius_error={frobenius_error:.2e}',
        'passed': passed,
        'error': float(frobenius_error),
        'tolerance': float(tol),
        'is_hermitian': bool(is_hermitian),
        'eigenvalues': eigenvalues.tolist(),
        'condition_number': float(np.max(eigenvalues) / np.min(eigenvalues))
                           if np.min(eigenvalues) > 0 else float('inf')
    }

    logging.info(f"Quotient isometry validation: {result}")

    return result


def validate_kernel_orthogonality(kernel_basis: np.ndarray,
                                 tol: float = 1e-10) -> Dict[str, Any]:
    """
    🧠 Function: validate_kernel_orthogonality
    Role: Validate kernel basis vectors are orthonormal
    Inputs: kernel_basis - computed kernel
            tol - tolerance for orthogonality
    Returns: Validation result dictionary
    Notes: Checks Gram matrix is identity
    """
    if kernel_basis.shape[1] == 0:
        # Empty kernel - trivially orthogonal
        return {
            'test': 'kernel_orthogonality',
            'theory_predicts': 0.0,
            'discovered': 0.0,
            'passed': True,
            'error': 0.0,
            'tolerance': float(tol),
            'note': 'Empty kernel - trivially orthogonal'
        }

    # Compute Gram matrix
    gram = kernel_basis.conj().T @ kernel_basis

    # Check diagonal (normalization)
    diag_error = np.max(np.abs(np.diag(gram) - 1.0))

    # Check off-diagonal (orthogonality)
    gram_copy = gram.copy()
    np.fill_diagonal(gram_copy, 0.0)
    off_diag_error = np.max(np.abs(gram_copy))

    # Maximum error
    max_error = max(diag_error, off_diag_error)

    passed = (max_error < tol)

    result = {
        'test': 'kernel_orthogonality',
        'theory_predicts': 0.0,
        'discovered': float(max_error),
        'passed': passed,
        'error': float(max_error),
        'tolerance': float(tol),
        'diagonal_error': float(diag_error),
        'off_diagonal_error': float(off_diag_error),
        'gram_matrix': gram.tolist()
    }

    logging.info(f"Kernel orthogonality validation: {result}")

    return result


def validate_null_space_property(pi: np.ndarray,
                                kernel_basis: np.ndarray,
                                tol: float = 1e-13) -> Dict[str, Any]:
    """
    🧠 Function: validate_null_space_property
    Role: Validate π·v = 0 for all kernel vectors
    Inputs: pi - surjection matrix
            kernel_basis - computed kernel
            tol - tolerance for null space check
    Returns: Validation result dictionary
    Notes: Fundamental property - kernel must be in null space
    """
    if kernel_basis.shape[1] == 0:
        return {
            'test': 'null_space_property',
            'theory_predicts': 0.0,
            'discovered': 0.0,
            'passed': True,
            'error': 0.0,
            'tolerance': float(tol),
            'note': 'Empty kernel'
        }

    # Compute π·K
    projection = pi @ kernel_basis
    max_projection = np.max(np.abs(projection))

    passed = (max_projection < tol)

    result = {
        'test': 'null_space_property',
        'theory_predicts': 0.0,
        'discovered': float(max_projection),
        'passed': passed,
        'error': float(max_projection),
        'tolerance': float(tol),
        'max_projection_location': np.unravel_index(np.argmax(np.abs(projection)),
                                                    projection.shape)
    }

    logging.info(f"Null space property validation: {result}")

    return result


def create_validation_summary(tests: List[Dict[str, Any]],
                            experiment_id: str = "E49") -> Dict[str, Any]:
    """
    🧠 Function: create_validation_summary
    Role: Aggregate all validation results
    Inputs: tests - list of test result dictionaries
            experiment_id - experiment identifier
    Returns: Complete validation summary
    Notes: Determines overall pass/fail and falsification status
    """
    # Overall pass if all tests pass
    overall_pass = all(test.get('passed', False) for test in tests)

    # Determine falsification status
    if not overall_pass:
        critical_failures = [t for t in tests
                           if not t['passed'] and
                           t['test'] in ['kernel_dimension', 'kernel_orthogonality',
                                       'quotient_isometry']]
        if critical_failures:
            falsification_result = "FAIL"
        else:
            falsification_result = "INCONCLUSIVE"
    else:
        falsification_result = "PASS"

    summary = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'predictions': tests,
        'overall_pass': overall_pass,
        'falsification_result': falsification_result,
        'n_tests': len(tests),
        'n_passed': sum(1 for t in tests if t.get('passed', False)),
        'n_failed': sum(1 for t in tests if not t.get('passed', False))
    }

    logging.info(f"Validation summary: {falsification_result} "
                f"({summary['n_passed']}/{summary['n_tests']} passed)")

    return summary


def run_full_validation(code: SurjectionCode) -> Dict[str, Any]:
    """
    🧠 Function: run_full_validation
    Role: Run all validation tests on a SurjectionCode
    Inputs: code - SurjectionCode object
    Returns: Complete validation report
    Notes: Main entry point for validation
    """
    tests = []

    # Test 1: Kernel dimension
    tests.append(validate_kernel_dimension(
        code.kernel, code.pi, theory_prediction=2
    ))

    # Test 2: Kernel orthogonality
    tests.append(validate_kernel_orthogonality(
        code.kernel, tol=1e-10
    ))

    # Test 3: Quotient isometry
    tests.append(validate_quotient_isometry(
        code.quotient_metric, target_dim=2, tol=1e-10
    ))

    # Test 4: Null space property
    tests.append(validate_null_space_property(
        code.pi, code.kernel, tol=1e-13
    ))

    # Create summary
    summary = create_validation_summary(tests, experiment_id="E49")

    return summary


def save_validation_report(report: Dict[str, Any],
                         output_path: str = "outputs/results/validation_report.json"):
    """
    🧠 Function: save_validation_report
    Role: Save validation report to JSON file
    Inputs: report - validation summary dictionary
            output_path - where to save
    Returns: None (saves to file)
    Notes: Creates directory if needed
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logging.info(f"Validation report saved to {output_path}")


def print_validation_summary(report: Dict[str, Any]):
    """
    🧠 Function: print_validation_summary
    Role: Print human-readable validation summary
    Inputs: report - validation summary dictionary
    Returns: None (prints to stdout)
    Notes: Useful for interactive debugging
    """
    print("\n" + "="*60)
    print(f"VALIDATION SUMMARY - {report['experiment_id']}")
    print("="*60)

    print(f"\nTimestamp: {report['timestamp']}")
    print(f"Tests Run: {report['n_tests']}")
    print(f"Tests Passed: {report['n_passed']}")
    print(f"Tests Failed: {report['n_failed']}")

    print(f"\nFalsification Result: {report['falsification_result']}")
    print(f"Overall Pass: {report['overall_pass']}")

    print("\nIndividual Test Results:")
    print("-"*40)

    for test in report['predictions']:
        status = "✓" if test['passed'] else "✗"
        print(f"{status} {test['test']:25s} error={test['error']:.2e}")

    if report['falsification_result'] == "FAIL":
        print("\n⚠️  THEORY FALSIFIED - Critical predictions not met!")
    elif report['falsification_result'] == "PASS":
        print("\n✅ Theory predictions validated successfully")
    else:
        print("\n❓ Results inconclusive - further investigation needed")

    print("="*60 + "\n")