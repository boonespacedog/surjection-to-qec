#!/usr/bin/env python3
"""
Main runner for E49: Kernel Geometry Visualization
Executes complete workflow and generates all outputs
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.complexfloating):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return super().default(obj)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core import SurjectionCode
from src.compute import create_standard_surjection
from src.validate import run_full_validation
from src.visualize import save_all_figures


def main():
    """Execute E49 experiment workflow."""
    logger.info("="*80)
    logger.info("E49: Kernel Geometry Visualization - STARTING")
    logger.info("="*80)

    # Create output directories
    output_dir = Path(__file__).parent / 'outputs'
    results_dir = output_dir / 'results'
    figures_dir = output_dir / 'figures'

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Step 1: Create standard surjection from paper
    logger.info("\n--- Step 1: Creating standard 4→2 surjection ---")
    pi = create_standard_surjection()
    logger.info(f"Surjection matrix shape: {pi.shape}")
    logger.info(f"Surjection matrix:\n{pi}")

    # Step 2: Initialize SurjectionCode
    logger.info("\n--- Step 2: Initializing SurjectionCode ---")
    code = SurjectionCode(pi)
    logger.info(f"Physical dimension: {code.physical_dim}")
    logger.info(f"Logical dimension: {code.logical_dim}")
    logger.info(f"Kernel dimension: {code.kernel_dim}")

    # Step 3: Compute kernel geometry
    logger.info("\n--- Step 3: Computing kernel geometry ---")
    kernel_basis = code.kernel
    singular_values = code.singular_values
    quotient_metric = code.quotient_metric

    logger.info(f"Kernel basis shape: {kernel_basis.shape}")
    logger.info(f"Kernel basis:\n{kernel_basis}")
    logger.info(f"Singular values: {singular_values}")
    logger.info(f"Quotient metric:\n{quotient_metric}")

    # Step 4: Run validation
    logger.info("\n--- Step 4: Running validation tests ---")
    validation_results = run_full_validation(code)

    logger.info("\nValidation Summary:")
    logger.info(f"  Overall pass: {validation_results['overall_pass']}")
    logger.info(f"  Falsification result: {validation_results['falsification_result']}")
    logger.info(f"  Tests passed: {validation_results['n_passed']}/{validation_results['n_tests']}")

    for test in validation_results['predictions']:
        status = "✓ PASS" if test['passed'] else "✗ FAIL"
        logger.info(f"  {status}: {test['test']}")

    # Step 5: Save numerical results
    logger.info("\n--- Step 5: Saving numerical results ---")

    # Save kernel basis
    np.savetxt(results_dir / 'kernel_basis_real.csv',
               kernel_basis.real, delimiter=',', fmt='%.16e')
    np.savetxt(results_dir / 'kernel_basis_imag.csv',
               kernel_basis.imag, delimiter=',', fmt='%.16e')

    # Save singular values
    np.savetxt(results_dir / 'singular_values.csv',
               singular_values, delimiter=',', fmt='%.16e')

    # Save quotient metric
    np.savetxt(results_dir / 'quotient_metric_real.csv',
               quotient_metric.real, delimiter=',', fmt='%.16e')
    np.savetxt(results_dir / 'quotient_metric_imag.csv',
               quotient_metric.imag, delimiter=',', fmt='%.16e')

    # Save validation results
    with open(results_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

    # Save complete state
    state_dict = code.get_state_dict()
    with open(results_dir / 'complete_state.json', 'w') as f:
        json.dump(state_dict, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Saved {len(list(results_dir.glob('*')))} result files")

    # Step 6: Generate visualizations
    logger.info("\n--- Step 6: Generating visualizations ---")
    figures_dict = save_all_figures(code, figures_dir)

    logger.info(f"Generated {len(figures_dict)} figures:")
    for fig_name, fig_path in figures_dict.items():
        logger.info(f"  - {fig_name}: {fig_path}")

    # Step 7: Create quick summary
    logger.info("\n--- Step 7: Creating summary ---")

    summary = f"""E49: Kernel Geometry Visualization - COMPLETE

Status: {'SUCCESS' if validation_results['overall_pass'] else 'PARTIAL'}
Theory: {validation_results['falsification_result']}

Key Results:
- Kernel dimension: {code.kernel_dim} (theory predicted: 2)
- Singular values: {singular_values.tolist()}
- Orthogonality error: {code.verify_orthogonality():.2e}
- Code distance: {code.compute_code_distance()}

Validation:
- Tests passed: {validation_results['n_passed']}/{validation_results['n_tests']}
- Falsification result: {validation_results['falsification_result']}

Outputs:
- Results files: {len(list(results_dir.glob('*')))}
- Figures: {len(figures_dict)}

See EXPERIMENTAL_REPORT_E49.md for detailed analysis.
"""

    summary_file = output_dir / 'QUICK_SUMMARY_E49.txt'
    summary_file.write_text(summary)

    logger.info("\n" + "="*80)
    logger.info("E49: EXPERIMENT COMPLETE")
    logger.info("="*80)
    print("\n" + summary)

    return 0 if validation_results['overall_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
