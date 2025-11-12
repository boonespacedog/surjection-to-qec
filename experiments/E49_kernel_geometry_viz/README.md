# E49: Kernel Geometry Visualization (4-to-2 QEC Code)

**Experiment ID**: E49
**Paper**: Surjection→QEC v3 (Section 4.1, Example 4.1, lines 263-304)
**Status**: Pre-registered
**Created**: November 11, 2025
**Priority**: HIGH
**Difficulty**: TRIVIAL

## Objective

Visualize the kernel structure of a simple 4→2 dimensional quantum error-correcting code constructed from a bounded surjection, demonstrating how code properties emerge from kernel geometry.

## Hypothesis Tested

**H-SURJ-1**: Quantum error-correcting codes emerge naturally from kernel structure of bounded surjections, with code properties determined by kernel geometry.

## Theory Predictions

From paper (lines 276-297):
- **Surjection**: π: C^4 → C^2 defined by matrix [[1,0,1,0], [0,1,0,1]]
- **Kernel dimension**: 2 (codim = 2)
- **Kernel basis**: {(1,0,-1,0)^T, (0,1,0,-1)^T} (orthogonal)
- **Code distance**: d=1 (no error correction - pedagogical example)
- **Quotient isometry**: π induces isometry B/ker(π) → C^2

## Falsification Criteria

Theory is FALSE if:
- Kernel dimension ≠ 2
- Kernel basis vectors not orthogonal (angle ≠ 90°)
- Quotient metric doesn't match target Hilbert space
- Code structure doesn't emerge from kernel geometry

## Computational Method

**Approach**: Compute kernel via SVD, visualize geometry

```python
import numpy as np

# Define surjection
pi = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1]], dtype=complex)

# Compute kernel via SVD
U, s, Vh = np.linalg.svd(pi, full_matrices=True)
kernel_basis = Vh[2:, :].T  # Last 2 right singular vectors

# Compute quotient metric
quotient_metric = pi @ pi.conj().T

# Visualizations:
# 1. Kernel basis vectors (4D → 2D/3D projections)
# 2. Quotient space geometry
# 3. Angular relationships between basis vectors
```

## Expected Outcomes

- Kernel dimension: 2 (computed via SVD rank deficiency)
- Kernel basis orthogonality: <v1, v2> ≈ 0 (within machine precision 1e-15)
- Quotient metric: Identity matrix (isometry confirmed)
- Visualization: Clear geometric interpretation of code structure

## Hardware Requirements

- **Platform**: macOS (M1 Mac)
- **Memory**: < 1KB (trivial matrices)
- **Runtime**: < 1ms
- **Dependencies**: NumPy, Matplotlib

## Structure

```
E49_kernel_geometry_viz/
├── src/
│   ├── surjection.py       # Define π, compute kernel
│   └── visualization.py    # Plot kernel geometry
├── tests/
│   ├── test_kernel.py      # Verify dimension, orthogonality
│   └── test_quotient.py    # Verify isometry
├── outputs/
│   └── session_[date]/
│       ├── kernel_basis.csv
│       ├── kernel_angles.csv
│       ├── quotient_metric.csv
│       └── figures/
├── data/                   # (none needed - synthetic)
├── docs/
│   └── derivation.md       # Math background from paper
├── README.md               # This file
├── requirements.txt        # numpy, matplotlib, pytest
├── config.yaml             # Parameters (matrix dimensions)
└── hypothesis.md           # H-SURJ-1 details
```

## Quick Start

```bash
cd E49_kernel_geometry_viz
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/surjection.py
pytest tests/ -v
```

## Timeline

- **Phase 0** (Pre-registration): 1 hour
- **Phase 1** (Setup): 15 minutes
- **Phase 2** (Implementation): 2 hours
- **Phase 3** (Validation): 1 hour
- **Phase 4** (Archiving): 30 minutes
- **Total**: 4.75 hours

## Related Work

- **Paper**: ZERO-ATOMIC/surjection-to-qec/surjection_to_qec_v3_arxiv.tex
- **E35**: TPA GAP verification (Grade A+ example to follow)
- **MATRIX-DB**: Pre-registration in Zero2/hypothesis-matrix/MATRIX-DB/experiments/E49_kernel_geometry_viz/

## Success Criteria

- [ ] Kernel dimension = 2 (exact)
- [ ] Kernel basis orthogonal (inner product < 1e-10)
- [ ] Quotient metric = Identity (Frobenius norm < 1e-10)
- [ ] All tests pass (pytest ≥ 95%)
- [ ] Visualizations clear and publication-ready

## License

See LICENSE file

## Contact

Oksana Sudoma, Researcher
Claude Code (Anthropic), Co-author
