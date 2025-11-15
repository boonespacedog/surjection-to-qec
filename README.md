# Bounded Surjections to Quantum Error-Correcting Codes

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17585624-blue.svg)](https://doi.org/10.5281/zenodo.17585624)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Computational verification code for**: "From Bounded Surjections to Quantum Error-Correcting Codes"

**Author**: Oksana Sudoma
**Status**: Publication-ready

---

## Overview

This repository provides complete computational validation for quantum error-correcting codes constructed from bounded surjections. We demonstrate the surjection framework produces valid quantum codes with verified parameters and explore continuous code family evolution.

**Key discoveries**:
- **[[4,2,2]] quantum code** validated (kernel geometry)
- **Rank transitions** in continuous code families
- **Machine-precision kernel orthogonality** (< 10^-16)
- **Functorial morphisms** preserve quantum code structure

---

## Experiments

### E49: Kernel Geometry Visualization

Validates [[4,2,2]] quantum code construction from bounded surjections.

**Key results**:
- Code distance d=2 verified
- Kernel orthogonality: 10^-16 (machine precision)
- Logical operators detected
- Runtime: < 1 second

**Quick start**:
```bash
cd experiments/E49_kernel_geometry_viz
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### E50: Functorial Code Morphisms

Continuous code family evolution with rank transition detection.

**Key results**:
- Rank transitions detected in continuous families
- Morphism preservation verified
- Continuous parameter evolution tracked
- Runtime: ~1.5 seconds

**Quick start**:
```bash
cd experiments/E50_functorial_code_morphisms
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

---

## Repository Structure

```
surjection-to-qec/
├── experiments/
│   ├── E49_kernel_geometry_viz/           # [[4,2,2]] code validation
│   │   ├── src/                           # Core implementation
│   │   ├── tests/                         # Unit tests
│   │   ├── outputs/
│   │   │   ├── EXPERIMENTAL_REPORT_E49.md # Execution results
│   │   │   ├── ANALYTICAL_REPORT_E49.md   # Analysis and interpretation
│   │   │   └── figures/                   # Visualizations
│   │   ├── README.md                      # Experiment pre-registration
│   │   ├── main.py                        # Entry point
│   │   └── requirements.txt
│   └── E50_functorial_code_morphisms/     # Code family evolution
│       ├── src/                           # Core implementation
│       ├── tests/                         # Unit tests
│       ├── outputs/
│       │   ├── EXPERIMENTAL_REPORT_E50.md # Execution results
│       │   ├── ANALYTICAL_REPORT_E50.md   # Analysis and interpretation
│       │   └── session_*/                 # Validated numerical data
│       ├── outputs/             # Results (main session)
│       ├── main.py              # Entry point
│       └── requirements.txt
├── paper/
│   ├── surjection_to_qec_v7.tex
│   ├── surjection_to_qec_v7.pdf
│   ├── surjection_to_qec_v6.tex
│   └── surjection_to_qec_v6.pdf
├── LICENSE
├── README.md
└── .gitignore
```

---

## Mathematical Background

**Quantum error-correcting codes** protect quantum information from decoherence. The [[n,k,d]] notation represents:
- **n**: Number of physical qubits
- **k**: Number of logical qubits
- **d**: Code distance (error-correcting capability)

This work constructs quantum codes from **bounded surjections** φ: H₁ → H₂ satisfying ‖φ‖ ≤ C. The kernel geometry determines code parameters, and continuous parameter variation reveals rank transitions.

**Novel phenomena**:
- Surjection framework provides geometric construction
- Kernel orthogonality verified to machine precision
- Continuous families exhibit sharp rank transitions
- Functorial properties preserved under morphisms

---

## Reproducibility

All results are computationally verified:

1. **Run individual experiments**: See Quick Start sections above
2. **Run tests**: `python3 -m pytest tests/ -v` (in each experiment directory)
3. **Expected runtime**: < 3 seconds total (both experiments)

All numerical results match paper claims to stated precision.

---

## Citation

```bibtex
@misc{sudoma2025surjection,
  author = {Sudoma, Oksana},
  title = {From Bounded Surjections to Quantum Error-Correcting Codes},
  year = {2025},
  doi = {10.5281/zenodo.17585624},
  url = {https://github.com/boonespacedog/surjection-to-qec}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Oksana Sudoma** - Independent Researcher

Computational validation and mathematical formalism assisted by Claude (Anthropic). All scientific conclusions and theoretical insights are the author's sole responsibility.

---

## Links

- **Repository**: https://github.com/boonespacedog/surjection-to-qec
- **Zenodo Archive**: https://doi.org/10.5281/zenodo.17585624
- **Paper**: See `paper/` directory for latest version (v7)
