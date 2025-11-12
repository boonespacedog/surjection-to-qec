"""
📄 File: visualize.py
Purpose: Plotting functions for kernel geometry visualization
Created: November 11, 2025
Used by: main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, Tuple, Any
import json
import os
import logging


# === CONFIG ===
# 🛠️ Visualization Parameters
FIG_DPI = 300           # Publication quality
FIG_SIZE_3D = (10, 8)   # 3D plot size
FIG_SIZE_2D = (8, 6)    # 2D plot size
COLORMAP = 'coolwarm'   # For heatmaps
FONT_SIZE = 12          # Base font size

# 🧠 Notes
# - High DPI for publication quality
# - Consistent color scheme across plots
# - Clear labels and legends


def plot_kernel_3d_projection(kernel_basis: np.ndarray,
                             title: str = "Kernel Basis 3D Projection") -> Figure:
    """
    🧠 Function: plot_kernel_3d_projection
    Role: 3D projection of 4D kernel vectors
    Inputs: kernel_basis - 4×k matrix of kernel vectors
            title - plot title
    Returns: matplotlib Figure object
    Notes: Projects 4D complex vectors to 3D real space
    """
    if kernel_basis.shape[1] == 0:
        # Empty kernel - create placeholder
        fig = plt.figure(figsize=FIG_SIZE_3D, dpi=FIG_DPI)
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, 'Empty Kernel', ha='center', va='center')
        ax.set_title(title)
        return fig

    # Extract real and imaginary parts for visualization
    # Use first 3 components for 3D projection
    fig = plt.figure(figsize=FIG_SIZE_3D, dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection='3d')

    # Plot each kernel vector
    for i in range(kernel_basis.shape[1]):
        v = kernel_basis[:3, i]  # Take first 3 components

        # Plot real part
        ax.quiver(0, 0, 0,
                 v[0].real, v[1].real, v[2].real,
                 color=f'C{i}', arrow_length_ratio=0.1,
                 label=f'v{i+1} (real)', alpha=0.7, linewidth=2)

        # Plot imaginary part (if significant)
        if np.max(np.abs(v.imag)) > 1e-10:
            ax.quiver(0, 0, 0,
                     v[0].imag, v[1].imag, v[2].imag,
                     color=f'C{i}', arrow_length_ratio=0.1,
                     linestyle='--', alpha=0.5,
                     label=f'v{i+1} (imag)')

    # Formatting
    ax.set_xlabel('Component 1', fontsize=FONT_SIZE)
    ax.set_ylabel('Component 2', fontsize=FONT_SIZE)
    ax.set_zlabel('Component 3', fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE+2)
    ax.legend(loc='upper right', fontsize=FONT_SIZE-2)

    # Set equal aspect ratio for better visualization
    max_range = np.max(np.abs(kernel_basis[:3, :]))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()
    return fig


def plot_singular_values(s: np.ndarray,
                        threshold: float = 1e-14,
                        title: str = "Singular Value Spectrum") -> Figure:
    """
    🧠 Function: plot_singular_values
    Role: Bar plot of singular values with zero threshold
    Inputs: s - array of singular values
            threshold - zero threshold line
            title - plot title
    Returns: matplotlib Figure object
    Notes: Shows rank deficiency visually
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_2D, dpi=FIG_DPI)

    # Create bar plot
    indices = np.arange(len(s))
    colors = ['green' if sv > threshold else 'red' for sv in s]

    bars = ax.bar(indices, s, color=colors, alpha=0.7, edgecolor='black')

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--',
              label=f'Zero threshold ({threshold:.0e})', linewidth=2)

    # Add value labels on bars
    for i, (bar, sv) in enumerate(zip(bars, s)):
        if sv > threshold:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{sv:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE-2)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, threshold*2,
                   f'{sv:.2e}', ha='center', va='bottom', fontsize=FONT_SIZE-3,
                   rotation=45)

    # Formatting
    ax.set_xlabel('Singular Value Index', fontsize=FONT_SIZE)
    ax.set_ylabel('Singular Value', fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE+2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=FONT_SIZE-2)

    # Set x-axis labels
    ax.set_xticks(indices)
    ax.set_xticklabels([f'σ{i+1}' for i in indices])

    plt.tight_layout()
    return fig


def plot_quotient_heatmap(Q: np.ndarray,
                         title: str = "Quotient Metric Q = ππ†") -> Figure:
    """
    🧠 Function: plot_quotient_heatmap
    Role: Heatmap of quotient metric
    Inputs: Q - quotient metric matrix
            title - plot title
    Returns: matplotlib Figure object
    Notes: Should show identity for isometry
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_DPI)

    # Plot 1: Real part
    im1 = ax1.imshow(Q.real, cmap=COLORMAP, aspect='equal',
                    interpolation='nearest', vmin=-0.1, vmax=1.1)
    ax1.set_title('Real Part', fontsize=FONT_SIZE)
    ax1.set_xlabel('Column', fontsize=FONT_SIZE)
    ax1.set_ylabel('Row', fontsize=FONT_SIZE)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Value', fontsize=FONT_SIZE-2)

    # Add text annotations
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            text = ax1.text(j, i, f'{Q[i,j].real:.3f}',
                          ha="center", va="center", color="black",
                          fontsize=FONT_SIZE)

    # Plot 2: Imaginary part
    im2 = ax2.imshow(Q.imag, cmap=COLORMAP, aspect='equal',
                    interpolation='nearest', vmin=-0.1, vmax=0.1)
    ax2.set_title('Imaginary Part', fontsize=FONT_SIZE)
    ax2.set_xlabel('Column', fontsize=FONT_SIZE)
    ax2.set_ylabel('Row', fontsize=FONT_SIZE)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Value', fontsize=FONT_SIZE-2)

    # Add text annotations
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            text = ax2.text(j, i, f'{Q[i,j].imag:.3f}',
                          ha="center", va="center", color="black",
                          fontsize=FONT_SIZE)

    # Overall title
    fig.suptitle(title, fontsize=FONT_SIZE+2)
    plt.tight_layout()

    return fig


def plot_kernel_angles(gram_matrix: np.ndarray,
                      title: str = "Kernel Inner Products") -> Figure:
    """
    🧠 Function: plot_kernel_angles
    Role: Visualize inner products between kernel vectors
    Inputs: gram_matrix - Gram matrix of kernel basis
            title - plot title
    Returns: matplotlib Figure object
    Notes: Diagonal should be 1, off-diagonal should be 0
    """
    if gram_matrix.size == 0:
        # Empty matrix
        fig, ax = plt.subplots(figsize=FIG_SIZE_2D, dpi=FIG_DPI)
        ax.text(0.5, 0.5, 'Empty Kernel - No Inner Products',
               ha='center', va='center', fontsize=FONT_SIZE)
        ax.set_title(title)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIG_DPI)

    # Create heatmap
    im = ax.imshow(np.abs(gram_matrix), cmap='RdBu_r', aspect='equal',
                  interpolation='nearest', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|⟨vi, vj⟩|', fontsize=FONT_SIZE)

    # Add text annotations
    for i in range(gram_matrix.shape[0]):
        for j in range(gram_matrix.shape[1]):
            value = np.abs(gram_matrix[i, j])
            color = 'white' if value > 0.5 else 'black'
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", color=color,
                          fontsize=FONT_SIZE-2)

    # Labels
    ax.set_xlabel('Vector j', fontsize=FONT_SIZE)
    ax.set_ylabel('Vector i', fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE+2)

    # Set ticks
    ticks = list(range(gram_matrix.shape[0]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'v{i+1}' for i in ticks])
    ax.set_yticklabels([f'v{i+1}' for i in ticks])

    plt.tight_layout()
    return fig


def create_figure_manifest(figures: Dict[str, str]) -> Dict[str, Any]:
    """
    🧠 Function: create_figure_manifest
    Role: Create manifest for all generated figures
    Inputs: figures - dict mapping figure names to file paths
    Returns: Manifest dictionary
    Notes: Includes metadata for each figure
    """
    from datetime import datetime

    manifest = {
        'generated': datetime.now().isoformat(),
        'figures': {}
    }

    for name, path in figures.items():
        # Extract figure type from name
        if '3d' in name.lower():
            fig_type = '3d_projection'
            size = [1000, 800]
        elif 'heatmap' in name.lower():
            fig_type = 'heatmap'
            size = [1200, 500]
        elif 'singular' in name.lower():
            fig_type = 'bar_plot'
            size = [800, 600]
        elif 'angle' in name.lower() or 'inner' in name.lower():
            fig_type = 'heatmap'
            size = [600, 600]
        else:
            fig_type = 'plot'
            size = [800, 600]

        manifest['figures'][path] = {
            'description': name.replace('_', ' ').title(),
            'type': fig_type,
            'dpi': FIG_DPI,
            'size': size
        }

    return manifest


def save_all_figures(code,
                    output_dir: str = "outputs/figures/") -> Dict[str, str]:
    """
    🧠 Function: save_all_figures
    Role: Generate and save all visualization figures
    Inputs: code - SurjectionCode object
            output_dir - where to save figures
    Returns: Dictionary of figure names to paths
    Notes: Main entry point for visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # Figure 1: Kernel 3D projection
    fig1 = plot_kernel_3d_projection(code.kernel)
    path1 = os.path.join(output_dir, "figure_1_kernel_3d.png")
    fig1.savefig(path1, dpi=FIG_DPI, bbox_inches='tight')
    figures['kernel_3d_projection'] = path1
    plt.close(fig1)
    logging.info(f"Saved kernel 3D projection to {path1}")

    # Figure 2: Singular values
    fig2 = plot_singular_values(code.singular_values)
    path2 = os.path.join(output_dir, "figure_2_singular_values.png")
    fig2.savefig(path2, dpi=FIG_DPI, bbox_inches='tight')
    figures['singular_value_spectrum'] = path2
    plt.close(fig2)
    logging.info(f"Saved singular value plot to {path2}")

    # Figure 3: Quotient heatmap
    fig3 = plot_quotient_heatmap(code.quotient_metric)
    path3 = os.path.join(output_dir, "figure_3_quotient_heatmap.png")
    fig3.savefig(path3, dpi=FIG_DPI, bbox_inches='tight')
    figures['quotient_metric_heatmap'] = path3
    plt.close(fig3)
    logging.info(f"Saved quotient heatmap to {path3}")

    # Figure 4: Kernel inner products
    gram = code.kernel.conj().T @ code.kernel if code.kernel.shape[1] > 0 \
           else np.array([[]])
    fig4 = plot_kernel_angles(gram)
    path4 = os.path.join(output_dir, "figure_4_kernel_angles.png")
    fig4.savefig(path4, dpi=FIG_DPI, bbox_inches='tight')
    figures['kernel_inner_products'] = path4
    plt.close(fig4)
    logging.info(f"Saved kernel angles plot to {path4}")

    # Create and save manifest
    manifest = create_figure_manifest(figures)
    manifest_path = os.path.join(output_dir, "plots_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Saved figure manifest to {manifest_path}")

    return figures