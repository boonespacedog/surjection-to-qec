"""
File: core.py
Purpose: Core data structures for continuous code families
Created: 2025-11-11 (E50 implementation)
Used by: compute.py, validate.py, visualize.py, main.py

This module defines the SurjectionFamily class that represents
a continuous family of surjection-based quantum codes.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import warnings

# === CONFIG ===
# File Paths
OUTPUT_PATH = "outputs/"

# Parameters
DEFAULT_N_POINTS = 100  # Resolution for parameter sampling
RANK_TOLERANCE = 1e-10  # Tolerance for rank computation
SMOOTHNESS_THRESHOLD = 1.0  # Maximum allowed |Δdim/Δt|

# Notes
# - Centralize changeables for debug-proofing
# - Track dimension evolution through continuous families


class SurjectionFamily:
    """
    Continuous family of surjection codes π(t): B → H(t).

    This class represents a parameterized family of quantum error
    correcting codes that interpolate between two endpoint codes.
    The interpolation tracks how kernel dimension and code properties
    evolve continuously with the parameter t ∈ [0,1].

    Attributes:
        pi_func: Function that maps t → π(t) surjection matrix
        t_range: Parameter interval (default [0,1])
        n_points: Number of sampling points
        t_values: Array of parameter values
        _codes: Cached surjection matrices (lazy)
        _dimensions: Cached dimension evolution
        _critical_points: Cached critical points where rank changes
    """

    def __init__(self,
                 pi_func: Optional[Callable[[float], np.ndarray]] = None,
                 t_range: Tuple[float, float] = (0, 1),
                 n_points: int = DEFAULT_N_POINTS,
                 n: int = 8,
                 k_initial: int = 2,
                 k_final: int = 4):
        """
        Initialize parameterized code family.

        Args:
            pi_func: Function t → π(t) matrix. If None, creates
                     linear interpolation between random endpoints
            t_range: Parameter interval (default [0,1])
            n_points: Sampling resolution
            n: Physical dimension (2^n for n qubits)
            k_initial: Logical dimension at t=0
            k_final: Logical dimension at t=1
        """
        self.t_range = t_range
        self.n_points = n_points
        self.t_values = np.linspace(t_range[0], t_range[1], n_points)

        # Physical and logical dimensions
        self.n = n
        self.physical_dim = 2**n  # Hilbert space dimension
        self.k_initial = k_initial
        self.k_final = k_final

        # If no function provided, create linear interpolation
        if pi_func is None:
            # Import here to avoid circular dependency
            from .compute import construct_endpoint_surjection, interpolate_surjections

            # Create random endpoint surjections
            # Both must have same shape for linear interpolation
            # Use max(k_initial, k_final) rows, then project to lower rank
            k_max = max(k_initial, k_final)

            # Create full-rank versions
            pi_0_full = construct_endpoint_surjection(n, k_max, seed=42)
            pi_1_full = construct_endpoint_surjection(n, k_max, seed=43)

            # Project to desired ranks
            if k_initial < k_max:
                # Zero out some rows to reduce rank
                pi_0_full[k_initial:, :] = 0
            if k_final < k_max:
                # Zero out some rows to reduce rank
                pi_1_full[k_final:, :] = 0

            self.pi_0 = pi_0_full
            self.pi_1 = pi_1_full

            # Define interpolation function
            def default_pi_func(t: float) -> np.ndarray:
                return interpolate_surjections(self.pi_0, self.pi_1, t)

            self.pi_func = default_pi_func
        else:
            self.pi_func = pi_func
            # Extract endpoints for reference
            self.pi_0 = pi_func(t_range[0])
            self.pi_1 = pi_func(t_range[1])

        # Lazy caches
        self._codes = None
        self._dimensions = None
        self._critical_points = None
        self._kernel_bases = {}

    def get_surjection(self, t: float) -> np.ndarray:
        """
        Get surjection matrix at parameter value t.

        Function: get_surjection
        Role: Retrieve or compute surjection at given parameter
        Inputs: t - parameter value
        Returns: Complex matrix π(t)
        Notes: Uses cache when possible
        """
        if not (self.t_range[0] <= t <= self.t_range[1]):
            warnings.warn(f"Parameter t={t} outside range {self.t_range}")

        return self.pi_func(t)

    def compute_dimension_evolution(self) -> np.ndarray:
        """
        Track logical dimension k(t) = rank(π(t)).

        Function: compute_dimension_evolution
        Role: Compute how code dimension changes with parameter
        Inputs: None (uses internal t_values)
        Returns: Array of dimensions at each t
        Notes: This is a key observable for smoothness
        """
        if self._dimensions is not None:
            return self._dimensions

        dimensions = []
        for t in self.t_values:
            pi_t = self.get_surjection(t)
            k_t = np.linalg.matrix_rank(pi_t, tol=RANK_TOLERANCE)
            dimensions.append(k_t)

        self._dimensions = np.array(dimensions)
        return self._dimensions

    def compute_kernel_dimension_evolution(self) -> np.ndarray:
        """
        Track kernel dimension evolution: dim(ker(π(t))).

        Function: compute_kernel_dimension_evolution
        Role: Compute how kernel dimension changes with parameter
        Inputs: None (uses internal t_values)
        Returns: Array of kernel dimensions at each t
        Notes: Kernel dimension = n - rank(π(t))
        """
        dimensions = self.compute_dimension_evolution()
        kernel_dims = self.physical_dim - dimensions
        return kernel_dims

    def find_critical_points(self) -> List[float]:
        """
        Find t where rank changes (dimension jumps).

        Function: find_critical_points
        Role: Identify parameter values where code structure changes
        Inputs: None
        Returns: List of critical parameter values
        Notes: These are potential non-smooth points
        """
        if self._critical_points is not None:
            return self._critical_points

        dims = self.compute_dimension_evolution()
        critical = []

        for i in range(1, len(dims)):
            if dims[i] != dims[i-1]:
                # Rank changed between t[i-1] and t[i]
                # Could refine with bisection for exact location
                critical.append(self.t_values[i])

        self._critical_points = critical
        return critical

    def compute_kernel_basis(self, t: float) -> np.ndarray:
        """
        Compute orthonormal basis for kernel at parameter t.

        Function: compute_kernel_basis
        Role: Find basis vectors spanning ker(π(t))
        Inputs: t - parameter value
        Returns: Matrix with kernel basis as columns
        Notes: Used for code distance computation
        """
        if t in self._kernel_bases:
            return self._kernel_bases[t]

        pi_t = self.get_surjection(t)

        # Use SVD to find kernel (null space)
        U, s, Vh = np.linalg.svd(pi_t, full_matrices=True)

        # Find indices where singular values are essentially zero
        rank = np.sum(s > RANK_TOLERANCE)

        # Kernel basis vectors are last (n - rank) columns of V
        V = Vh.conj().T
        kernel_basis = V[:, rank:]

        # Cache result
        self._kernel_bases[t] = kernel_basis

        return kernel_basis

    def get_code_parameters(self, t: float) -> Dict[str, float]:
        """
        Get [n, k, d] parameters of code at parameter t.

        Function: get_code_parameters
        Role: Extract standard quantum code parameters
        Inputs: t - parameter value
        Returns: Dictionary with n, k, d values
        Notes: d (distance) requires expensive computation
        """
        pi_t = self.get_surjection(t)

        # Physical qubits (log2 of dimension)
        n = int(np.log2(self.physical_dim))

        # Logical qubits (rank of surjection)
        k = np.linalg.matrix_rank(pi_t, tol=RANK_TOLERANCE)

        # Code distance (requires kernel analysis)
        # Import here to avoid circular dependency
        from .compute import compute_code_distance_numerical
        d = compute_code_distance_numerical(pi_t)

        return {
            'n': n,
            'k': k,
            'd': d,
            't': t
        }

    def verify_smoothness(self, threshold: float = SMOOTHNESS_THRESHOLD) -> Tuple[bool, float]:
        """
        Check if dimension evolution is smooth.

        Function: verify_smoothness
        Role: Validate smoothness criterion |Δdim/Δt| ≤ threshold
        Inputs: threshold - maximum allowed derivative
        Returns: (is_smooth, max_derivative)
        Notes: Key validation for functorial property
        """
        dims = self.compute_dimension_evolution()

        # Compute discrete derivatives
        dt = np.diff(self.t_values)
        ddim = np.diff(dims)

        # Handle zero dt (shouldn't happen with linspace)
        with np.errstate(divide='ignore', invalid='ignore'):
            derivatives = ddim / dt
            derivatives[~np.isfinite(derivatives)] = 0

        max_derivative = np.max(np.abs(derivatives))
        is_smooth = max_derivative <= threshold

        return is_smooth, max_derivative

    def get_singular_value_evolution(self) -> np.ndarray:
        """
        Track singular value evolution σᵢ(t).

        Function: get_singular_value_evolution
        Role: Compute how singular values change with parameter
        Inputs: None
        Returns: 2D array [n_points, n_singular_values]
        Notes: Shows continuous spectral evolution
        """
        all_svs = []
        max_sv_count = 0

        for t in self.t_values:
            pi_t = self.get_surjection(t)
            _, s, _ = np.linalg.svd(pi_t, full_matrices=False)
            all_svs.append(s)
            max_sv_count = max(max_sv_count, len(s))

        # Pad to uniform size
        sv_array = np.zeros((len(self.t_values), max_sv_count))
        for i, svs in enumerate(all_svs):
            sv_array[i, :len(svs)] = svs

        return sv_array

    def summary(self) -> Dict:
        """
        Generate summary of family properties.

        Function: summary
        Role: Create comprehensive report of code family
        Inputs: None
        Returns: Dictionary with all key metrics
        Notes: Used for validation and reporting
        """
        dims = self.compute_dimension_evolution()
        kernel_dims = self.compute_kernel_dimension_evolution()
        critical = self.find_critical_points()
        is_smooth, max_deriv = self.verify_smoothness()

        # Get endpoint parameters
        params_0 = self.get_code_parameters(self.t_range[0])
        params_1 = self.get_code_parameters(self.t_range[1])

        return {
            't_range': self.t_range,
            'n_points': self.n_points,
            'physical_dim': self.physical_dim,
            'dimension_range': [int(np.min(dims)), int(np.max(dims))],
            'kernel_dim_range': [int(np.min(kernel_dims)), int(np.max(kernel_dims))],
            'n_critical_points': len(critical),
            'critical_points': critical,
            'is_smooth': is_smooth,
            'max_derivative': float(max_deriv),
            'smoothness_threshold': SMOOTHNESS_THRESHOLD,
            'endpoint_0': params_0,
            'endpoint_1': params_1,
            'discovery': f"Family has {len(critical)} rank changes, max |Δdim/Δt| = {max_deriv:.4f}"
        }