"""
File: visualize.py
Purpose: Visualization of code evolution and properties
Created: 2025-11-11 (E50 implementation)
Used by: main.py

This module creates visualizations for continuous code families,
including dimension evolution, singular value heatmaps, and
critical point identification.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

# === CONFIG ===
# File Paths
DEFAULT_OUTPUT_DIR = "outputs/"

# Visualization Parameters
FIGURE_DPI = 100
FIGURE_SIZE = (10, 6)
HEATMAP_SIZE = (12, 8)
GRID_ALPHA = 0.3
MARKER_SIZE = 50

# Color scheme
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'
COLOR_TERTIARY = '#F18F01'
COLOR_CRITICAL = '#C73E1D'

# Notes
# - High-quality plots for publication
# - Discovery-based annotations
# - Clear labeling of critical points


def plot_kernel_evolution(family,
                         save_path: Optional[str] = None,
                         show_critical: bool = True) -> plt.Figure:
    """
    Plot kernel dimension evolution k(t).

    Function: plot_kernel_evolution
    Role: Visualize how kernel dimension changes with parameter
    Inputs: family, save_path, show_critical
    Returns: matplotlib Figure
    Notes: Main visualization for smoothness analysis
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE,
                                   sharex=True, height_ratios=[2, 1])

    # Get kernel dimension curve
    from .compute import compute_kernel_dimension_curve
    t_values, kernel_dims = compute_kernel_dimension_curve(family)

    # Get logical dimension (rank) evolution
    logical_dims = family.compute_dimension_evolution()

    # Plot kernel dimension
    ax1.plot(t_values, kernel_dims, color=COLOR_PRIMARY, linewidth=2,
             label='Kernel dimension')
    ax1.scatter([t_values[0], t_values[-1]],
               [kernel_dims[0], kernel_dims[-1]],
               color=COLOR_PRIMARY, s=MARKER_SIZE, zorder=5,
               label='Endpoints')

    # Mark critical points
    if show_critical:
        critical_points = family.find_critical_points()
        for t_crit in critical_points:
            idx = np.argmin(np.abs(t_values - t_crit))
            ax1.axvline(t_crit, color=COLOR_CRITICAL, linestyle='--',
                       alpha=0.7, label='Critical point' if t_crit == critical_points[0] else '')
            ax1.scatter(t_crit, kernel_dims[idx], color=COLOR_CRITICAL,
                       s=MARKER_SIZE, zorder=5)

    ax1.set_ylabel('Kernel Dimension', fontsize=12)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc='best')
    ax1.set_title('Kernel Dimension Evolution in Code Family', fontsize=14)

    # Plot discrete derivative |Δdim/Δt|
    dt = np.diff(t_values)
    ddim = np.diff(kernel_dims)
    with np.errstate(divide='ignore', invalid='ignore'):
        derivatives = ddim / dt
        derivatives[~np.isfinite(derivatives)] = 0

    t_mid = (t_values[:-1] + t_values[1:]) / 2
    ax2.plot(t_mid, np.abs(derivatives), color=COLOR_SECONDARY,
             linewidth=2, label='|Δdim/Δt|')

    # Mark smoothness threshold
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7,
               label='Smoothness threshold')

    # Highlight violations
    violations = np.where(np.abs(derivatives) > 1.0)[0]
    if len(violations) > 0:
        ax2.scatter(t_mid[violations], np.abs(derivatives[violations]),
                   color=COLOR_CRITICAL, s=MARKER_SIZE, zorder=5,
                   label='Violations')

    ax2.set_xlabel('Parameter t', fontsize=12)
    ax2.set_ylabel('|Δdim/Δt|', fontsize=12)
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(loc='best')

    # Add discovery annotation
    max_deriv = np.max(np.abs(derivatives))
    ax2.text(0.02, 0.98, f'Max |Δdim/Δt| = {max_deriv:.4f}',
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_code_parameters(family,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot evolution of [n, k, d] parameters.

    Function: plot_code_parameters
    Role: Track code parameters through family
    Inputs: family, save_path
    Returns: matplotlib Figure
    Notes: Shows how error correction capability evolves
    """
    fig, axes = plt.subplots(2, 2, figsize=HEATMAP_SIZE)

    # Get parameter evolution
    t_values = family.t_values
    n_vals = []
    k_vals = []
    d_vals = []

    print("Computing code parameters (may take a moment)...")
    for i, t in enumerate(t_values):
        if i % 20 == 0:  # Progress indicator
            print(f"  Processing t = {t:.2f}")
        params = family.get_code_parameters(t)
        n_vals.append(params['n'])
        k_vals.append(params['k'])
        d_vals.append(params['d'])

    # Plot logical dimension k
    ax = axes[0, 0]
    ax.plot(t_values, k_vals, color=COLOR_PRIMARY, linewidth=2)
    ax.scatter([t_values[0], t_values[-1]], [k_vals[0], k_vals[-1]],
              color=COLOR_PRIMARY, s=MARKER_SIZE, zorder=5)
    ax.set_xlabel('Parameter t')
    ax.set_ylabel('Logical dimension k')
    ax.set_title('Logical Dimension Evolution')
    ax.grid(True, alpha=GRID_ALPHA)

    # Plot code distance d
    ax = axes[0, 1]
    ax.plot(t_values, d_vals, color=COLOR_SECONDARY, linewidth=2)
    ax.scatter([t_values[0], t_values[-1]], [d_vals[0], d_vals[-1]],
              color=COLOR_SECONDARY, s=MARKER_SIZE, zorder=5)
    ax.set_xlabel('Parameter t')
    ax.set_ylabel('Code distance d')
    ax.set_title('Code Distance Evolution')
    ax.grid(True, alpha=GRID_ALPHA)

    # Plot kernel dimension (redundant with above but useful comparison)
    kernel_dims = family.compute_kernel_dimension_evolution()
    ax = axes[1, 0]
    ax.plot(t_values, kernel_dims, color=COLOR_TERTIARY, linewidth=2)
    ax.set_xlabel('Parameter t')
    ax.set_ylabel('Kernel dimension')
    ax.set_title('Kernel Dimension')
    ax.grid(True, alpha=GRID_ALPHA)

    # Plot rate k/n
    rates = np.array(k_vals) / np.array(n_vals)
    ax = axes[1, 1]
    ax.plot(t_values, rates, color='purple', linewidth=2)
    ax.set_xlabel('Parameter t')
    ax.set_ylabel('Code rate k/n')
    ax.set_title('Code Rate Evolution')
    ax.grid(True, alpha=GRID_ALPHA)

    # Add discovery annotations
    fig.suptitle(f'Code Parameters: n={n_vals[0]}, k: {k_vals[0]}→{k_vals[-1]}, '
                f'd: {d_vals[0]}→{d_vals[-1]}', fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_singular_values(family,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot singular value evolution as heatmap.

    Function: plot_singular_values
    Role: Visualize spectral evolution of code family
    Inputs: family, save_path
    Returns: matplotlib Figure
    Notes: Shows continuous spectral changes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=HEATMAP_SIZE)

    # Get singular value evolution
    sv_array = family.get_singular_value_evolution()

    # Plot heatmap of singular values
    im1 = ax1.imshow(sv_array.T, aspect='auto', origin='lower',
                     extent=[family.t_range[0], family.t_range[1],
                            0, sv_array.shape[1]],
                     cmap='viridis')
    ax1.set_xlabel('Parameter t', fontsize=12)
    ax1.set_ylabel('Singular value index', fontsize=12)
    ax1.set_title('Singular Value Evolution σᵢ(t)', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Singular value')

    # Plot log singular values for better visibility
    with np.errstate(divide='ignore'):
        log_sv = np.log10(sv_array + 1e-16)  # Avoid log(0)

    im2 = ax2.imshow(log_sv.T, aspect='auto', origin='lower',
                     extent=[family.t_range[0], family.t_range[1],
                            0, sv_array.shape[1]],
                     cmap='plasma', vmin=-16, vmax=1)
    ax2.set_xlabel('Parameter t', fontsize=12)
    ax2.set_ylabel('Singular value index', fontsize=12)
    ax2.set_title('Log₁₀ Singular Values', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='log₁₀(σᵢ)')

    # Mark critical points
    critical_points = family.find_critical_points()
    for ax in [ax1, ax2]:
        for t_crit in critical_points:
            ax.axvline(t_crit, color='white', linestyle='--',
                      alpha=0.5, linewidth=1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_validation_summary(validation_results: Dict,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create summary plot of validation results.

    Function: plot_validation_summary
    Role: Visualize validation test outcomes
    Inputs: validation_results, save_path
    Returns: matplotlib Figure
    Notes: Shows pass/fail status and discoveries
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract test results
    if 'individual_results' in validation_results:
        tests = validation_results['individual_results']
    else:
        tests = [validation_results]

    # Create bar chart of test results
    test_names = []
    test_status = []
    test_colors = []

    for i, test in enumerate(tests):
        # Try to extract test name from discovery or use index
        if 'discovery' in test:
            name = f"Test {i+1}"
        else:
            name = f"Test {i+1}"

        test_names.append(name)
        validated = test.get('validated', False)
        test_status.append(1 if validated else 0)
        test_colors.append('green' if validated else 'red')

    y_pos = np.arange(len(test_names))
    bars = ax.barh(y_pos, test_status, color=test_colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(test_names)
    ax.set_xlabel('Validation Status')
    ax.set_xlim(0, 1.2)
    ax.set_title('Validation Summary', fontsize=14)

    # Add text annotations
    for i, (bar, test) in enumerate(zip(bars, tests)):
        status_text = "PASS" if test.get('validated', False) else "FAIL"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               status_text, va='center', fontweight='bold',
               color='green' if status_text == "PASS" else 'red')

        # Add discovery text if available
        if 'discovery' in test:
            ax.text(0.02, bar.get_y() + bar.get_height()/2,
                   test['discovery'][:50] + '...' if len(test['discovery']) > 50
                   else test['discovery'],
                   va='center', fontsize=8)

    # Add overall status
    overall_text = (f"Overall: {validation_results.get('n_passed', 0)}/"
                   f"{validation_results.get('n_tests', len(tests))} passed")
    ax.text(0.5, 1.05, overall_text, transform=ax.transAxes,
           ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def create_all_visualizations(family,
                             output_dir: Optional[str] = None,
                             validation_results: Optional[Dict] = None) -> List[str]:
    """
    Create all visualizations for the experiment.

    Function: create_all_visualizations
    Role: Generate complete visualization suite
    Inputs: family, output_dir, validation_results
    Returns: List of created file paths
    Notes: Main entry point for visualization
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"session_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Create figures subdirectory
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print("Creating visualizations...")

    # Kernel evolution plot
    print("  - Kernel dimension evolution...")
    kernel_path = os.path.join(fig_dir, 'kernel_dim_vs_t.png')
    plot_kernel_evolution(family, save_path=kernel_path)
    created_files.append(kernel_path)
    plt.close()

    # Code parameters plot
    print("  - Code parameters...")
    params_path = os.path.join(fig_dir, 'code_parameters.png')
    plot_code_parameters(family, save_path=params_path)
    created_files.append(params_path)
    plt.close()

    # Singular values heatmap
    print("  - Singular value heatmap...")
    sv_path = os.path.join(fig_dir, 'singular_values_heatmap.png')
    plot_singular_values(family, save_path=sv_path)
    created_files.append(sv_path)
    plt.close()

    # Validation summary if results provided
    if validation_results:
        print("  - Validation summary...")
        val_path = os.path.join(fig_dir, 'validation_summary.png')
        plot_validation_summary(validation_results, save_path=val_path)
        created_files.append(val_path)
        plt.close()

    print(f"Created {len(created_files)} visualizations in {fig_dir}")

    return created_files