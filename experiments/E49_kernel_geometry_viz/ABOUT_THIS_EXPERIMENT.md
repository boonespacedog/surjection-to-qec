# E49: Kernel Geometry Visualization - Complete Documentation

## What We Wanted to Test

**Research Question**: Can bounded surjections rigorously construct quantum error-correcting codes with verifiable parameters?

**Hypothesis**: A 4→2 surjection with 2-dimensional kernel should produce a [[4,2,2]] quantum code where:
- Logical dimension k = 2
- Code distance d ≥ 2
- Kernel orthogonality enables error detection

**Why this matters**: Tests if geometric construction (kernel-based) produces valid QEC codes without needing stabilizer formalism.

---

## What We Suspected

**Expected outcomes** (pre-registration):
1. Code distance d = 2 (minimum for error detection)
2. Kernel orthogonality ~10⁻¹⁴ (numerical precision)
3. Normalization factor needed (Q = αI for some α)
4. Singular values related to code geometry

**Potential failures**:
- Code distance d = 1 (trivial, no protection)
- Non-orthogonal kernel (no clean error space)
- Normalization breakdown (non-unitary)

---

## What We Received (Results)

**From EXPERIMENTAL_REPORT_E49.md** (execution day):
- ✅ Code distance: d = 2 (verified via minimum Hamming weight)
- ✅ Kernel orthogonality: 1.3×10⁻¹⁶ (machine precision)
- ✅ Singular values: Both exactly √2
- ✅ Normalization: Q = 2I (requires rescaling by 1/√2)

**From ANALYTICAL_REPORT_E49.md** (next day analysis):
- The √2 factor emerges naturally from SVD (not hardcoded)
- Machine-precision orthogonality is expected for constructed orthogonal basis
- Code distance d=2 confirms error-detecting capability
- Construction is mathematically sound

---

## What This Means

**Scientific interpretation**:

1. **Surjection method works** - Geometric kernel construction produces valid QEC codes
2. **√2 is fundamental** - Emerges from 4D→2D dimensional reduction geometry (not arbitrary)
3. **Machine precision achieved** - Numerical stability of the construction is excellent
4. **[[4,2,2]] code validated** - Known code, but our construction is novel (kernel-based)

**Comparison to prior work**:
- Standard approach: Stabilizer formalism
- Our approach: Geometric (kernel projection)
- Advantage: Explicit kernel structure, easier visualization

**Limitations**:
- Demonstrated on one specific code ([[4,2,2]])
- Numerical validation only (not formal proof)
- Needs extension to larger codes

---

## File Organization

**Pre-registration** (what we planned):
- `README.md` - Aspirational experimental design (may reference files not implemented)

**Execution** (what actually ran):
- `main.py` - Actual entry point
- `src/` - Implemented modules
- `tests/` - Unit tests

**Results** (what we found):
- `outputs/EXPERIMENTAL_REPORT_E49.md` - Raw results from execution
- `outputs/ANALYTICAL_REPORT_E49.md` - Interpretation and meaning
- `outputs/results/validation_results.json` - Numerical data
- `outputs/figures/` - Visualizations

**Note**: Pre-registration README may mention planned files (hypothesis.md, derivation.md, etc.) that were not created as separate files but are incorporated in the reports.

---

## Quick Start

```bash
cd /Users/mac/Desktop/egg-paper/git-hub-repos/surjection-to-qec/experiments/E49_kernel_geometry_viz
python3 main.py
```

**Runtime**: < 1 second
**Outputs**: Validation results in `outputs/results/`

---

**For full context**: Read EXPERIMENTAL_REPORT (execution) then ANALYTICAL_REPORT (interpretation).

**Created**: November 15, 2025
**Purpose**: Clarify documentation hierarchy for outside readers
