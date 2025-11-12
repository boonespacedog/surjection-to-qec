# Architecture: E49 Kernel Geometry Visualization

## Module Structure

```
src/
├── __init__.py          # Package initialization
├── core.py              # Core mathematical objects (SurjectionCode class)
├── compute.py           # Main computation functions (kernel, metrics)
├── validate.py          # Validation and verification functions
├── visualize.py         # Plotting functions for kernel geometry
└── utils.py             # Helper functions (I/O, logging)
```

## Class Designs

### Core Module (core.py)

```python
class SurjectionCode:
    """
    Quantum error correction code from bounded surjection.

    Attributes:
        pi: np.ndarray - Surjection matrix π: B → H
        _kernel: np.ndarray - Cached kernel basis (lazy)
        _logical_dim: int - Cached logical dimension
        _quotient_metric: np.ndarray - Cached quotient metric

    Methods:
        __init__(pi_matrix: np.ndarray)
        kernel() -> np.ndarray  # Property with caching
        quotient_metric() -> np.ndarray  # Property
        code_distance() -> int  # Compute minimum weight
        validate_surjection() -> bool  # Check full rank
    """
```

### Compute Module (compute.py)

```python
def compute_kernel_via_svd(pi: np.ndarray, tol: float = 1e-14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kernel basis using SVD with specified tolerance.

    Returns:
        kernel_basis: Orthonormal basis for ker(π)
        singular_values: All singular values for diagnostics
    """

def compute_quotient_metric(pi: np.ndarray) -> np.ndarray:
    """Compute Q = π π^† quotient metric."""

def compute_kernel_angles(kernel_basis: np.ndarray) -> np.ndarray:
    """Compute inner product matrix between kernel vectors."""

def verify_orthogonality(basis: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if basis vectors are orthonormal."""
```

### Validate Module (validate.py)

```python
def validate_kernel_dimension(kernel_basis: np.ndarray,
                            pi_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Validate kernel dimension matches rank deficiency.

    Returns:
        Dict with 'discovered', 'expected', 'passed', 'error'
    """

def validate_quotient_isometry(Q: np.ndarray,
                              target_dim: int,
                              tol: float = 1e-10) -> Dict[str, Any]:
    """Validate Q ≈ I_target."""

def create_validation_summary(tests: List[Dict]) -> Dict[str, Any]:
    """Aggregate all validation results."""
```

### Visualize Module (visualize.py)

```python
def plot_kernel_3d_projection(kernel_basis: np.ndarray) -> Figure:
    """3D projection of 4D kernel vectors."""

def plot_singular_values(s: np.ndarray, threshold: float) -> Figure:
    """Bar plot of singular values with zero threshold."""

def plot_quotient_heatmap(Q: np.ndarray) -> Figure:
    """Heatmap of quotient metric."""

def create_figure_manifest(figures: Dict[str, str]) -> Dict:
    """Create manifest for all generated figures."""
```

## Data Flow

```
Input: Surjection matrix π (2×4)
    ↓
[Core Module]
    Create SurjectionCode object
    Validate surjection (full rank)
    ↓
[Compute Module]
    SVD decomposition → singular values
    Extract kernel basis (null space)
    Compute quotient metric Q = ππ†
    Compute kernel inner products
    ↓
[Validate Module]
    Check kernel dimension (theory: 2)
    Check orthogonality (|⟨vi,vj⟩| < 1e-10)
    Check isometry (||Q - I||_F < 1e-10)
    Create validation summary
    ↓
[Visualize Module]
    Generate 3D kernel projection
    Plot singular values
    Create quotient heatmap
    ↓
[Output]
    CSV: kernel_basis, singular_values, quotient_metric
    JSON: validation_summary
    PNG: figures with manifest
```

## Test Strategy

### Structure Tests (test_structure.py)
- **Input validation**: Matrix dimensions, complex dtype
- **SVD convergence**: Singular values computed correctly
- **Kernel extraction**: Correct null space identification
- **Caching**: Lazy properties work correctly
- **Error handling**: Invalid inputs raise appropriate errors

### Theory Tests (test_theory.py)
- **Kernel dimension**: Discover via SVD, compare to theory (2)
- **Orthogonality**: Kernel basis vectors orthonormal
- **Quotient isometry**: Q = I₂ within tolerance
- **Boundary cases**: Perturbed surjection, full rank matrix

### Integration Tests (test_integration.py)
- **End-to-end workflow**: Load → Compute → Validate → Save
- **Output existence**: All required files generated
- **Reproducibility**: Same input → same output
- **Performance**: Complete in < 1 second

## Dependencies

```yaml
# requirements.txt
numpy==1.24.3          # Core numerical operations
scipy==1.11.4          # Additional linear algebra
matplotlib==3.8.0      # Visualization
pytest==7.4.3          # Testing framework
pytest-cov==4.1.0      # Code coverage
pyyaml==6.0.1         # Config file handling

# Why these versions:
# - numpy 1.24.3: Stable, M1-optimized, good SVD performance
# - scipy 1.11.4: Compatible sparse matrix support (future)
# - matplotlib 3.8.0: Latest stable, good 3D support
# - pytest 7.4.3: Modern fixtures, good error reporting
```

## Error Handling Strategy

1. **Input Validation**: Check matrix dimensions, dtype at boundaries
2. **Numerical Tolerance**: All comparisons use explicit tolerances
3. **Graceful Degradation**: If visualization fails, still save data
4. **Logging**: Structured logs with computation steps
5. **Provenance**: Track where each number came from

## Performance Considerations

- **Memory**: < 1KB for 4×2 matrices (trivial)
- **Computation**: SVD of 2×4 matrix < 1ms
- **Visualization**: Matplotlib rendering < 100ms
- **Total runtime**: < 1 second for complete workflow

## Quality Gates

✓ All tests pass (100% coverage of core logic)
✓ Validation summary shows all theory predictions met
✓ Outputs follow standard structure
✓ PROVENANCE.md documents every computed value
✓ No hardcoded expectations in tests