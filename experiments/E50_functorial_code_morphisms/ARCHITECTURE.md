# Architecture: E50 Functorial Code Morphisms

## Module Structure

```
src/
├── __init__.py          # Package initialization
├── core.py              # Core objects (SurjectionFamily class)
├── compute.py           # Interpolation and kernel evolution
├── validate.py          # Smoothness and functoriality validation
├── visualize.py         # Evolution plots
└── utils.py             # Helper functions
```

## Class Designs

### Core Module (core.py)

```python
class SurjectionFamily:
    """
    Continuous family of surjections π_t: C^n → C^k(t).

    Attributes:
        n: int - Physical dimension (8 qubits)
        t_values: np.ndarray - Interpolation parameters [0,1]
        pi_0: np.ndarray - Initial surjection (t=0)
        pi_1: np.ndarray - Final surjection (t=1)
        _cache: dict - Cached computations

    Methods:
        __init__(n=8, k_initial=2, k_final=4)
        get_surjection(t: float) -> np.ndarray
        compute_kernel_evolution() -> dict
        verify_smoothness() -> Tuple[bool, float]
        compute_code_parameters(t: float) -> dict
    """
```

### Compute Module (compute.py)

```python
def construct_endpoint_surjection(n: int, k: int, seed: int = 42) -> np.ndarray:
    """
    Construct random surjection with specified dimensions.
    Uses Gaussian random matrix with verified rank.
    """

def interpolate_surjections(pi_0: np.ndarray, pi_1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation: π_t = (1-t)π_0 + t·π_1."""

def compute_kernel_dimension_curve(family: SurjectionFamily) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kernel dimension at all t values."""

def compute_code_distance_numerical(pi: np.ndarray) -> int:
    """Compute code distance from kernel weight."""

def verify_smoothness_criterion(t_values: np.ndarray, kernel_dims: np.ndarray) -> Tuple[bool, float]:
    """Check |Δdim/Δt| ≤ 1 (discrete derivative)."""
```

### Validate Module (validate.py)

```python
def validate_kernel_evolution(family: SurjectionFamily) -> dict:
    """Validate smooth evolution of kernel dimension."""

def validate_endpoint_consistency(family: SurjectionFamily) -> dict:
    """Check t=0 and t=1 match specifications."""

def validate_functorial_composition(family: SurjectionFamily) -> dict:
    """Verify composition preserves code structure."""

def create_validation_summary(tests: List[dict]) -> dict:
    """Aggregate all validation results."""
```

## Data Flow

```
Input: n=8 qubits, k_0=2, k_1=4
    ↓
[Core Module]
    Create SurjectionFamily
    Generate endpoint surjections
    ↓
[Compute Module]
    For t ∈ [0,1] with 100 points:
        Interpolate π_t
        Compute kernel dimension
        Compute code distance
        Track singular values
    ↓
[Validate Module]
    Check smoothness (no jumps)
    Verify endpoints
    Test functoriality
    ↓
[Visualize Module]
    Plot kernel evolution
    Plot distance evolution
    Create heatmaps
    ↓
[Output]
    CSV: evolution data
    JSON: validation report
    PNG: evolution plots
```

## Test Strategy

### Structure Tests
- Interpolation stays in bounds
- Matrix dimensions preserved
- Rank constraints satisfied

### Theory Tests
- **Kernel smoothness**: No discontinuities
- **Monotonicity**: Kernel grows as range shrinks
- **Functoriality**: Composition consistency

### Integration Tests
- Complete parameter sweep
- Output generation
- Performance (<10 seconds)

## Dependencies

```yaml
numpy==1.24.3
scipy==1.11.4
matplotlib==3.8.0
pytest==7.4.3
```

## Key Algorithms

### Smoothness Verification
```python
# Discrete derivative test
discrete_deriv = np.diff(kernel_dims) / np.diff(t_values)
max_jump = np.max(np.abs(discrete_deriv))
smooth = (max_jump <= 1.0)  # Theory prediction
```

### Code Distance Computation
```python
# Minimum weight of kernel vectors
weights = [np.sum(np.abs(v) > tol) for v in kernel_basis.T]
distance = min(weights) if weights else float('inf')
```

## Quality Gates

✓ Kernel evolution is smooth (no jumps > 1)
✓ Endpoints match specifications
✓ All 100 interpolation points computed
✓ Validation report shows theory predictions met
✓ No hardcoded dimensions in tests