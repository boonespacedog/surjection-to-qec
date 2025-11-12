# E50: Functorial Code Morphisms (Continuous Code Family)

**Experiment ID**: E50
**Paper**: Surjection→QEC v3 (Section 4.3, Example 4.3, lines 344-381)
**Status**: Pre-registered
**Created**: November 11, 2025
**Priority**: HIGH
**Difficulty**: EASY

## Objective

Demonstrate continuous code family parameterized by t ∈ [0,1], tracking how logical dimension and code properties evolve as the surjection morphs, showing functorial structure.

## Hypothesis Tested

**H-SURJ-2**: Morphisms between surjections induce functorial transformations between quantum codes, providing systematic code optimization pathways.

## Theory Predictions

- Smooth evolution of kernel dimension (no discontinuities)
- Code distance varies continuously (contradict stabilizer requiring discrete generators)
- Functoriality: Composition of morphisms preserves code structure
- Surjection approach handles continuous families naturally

## Falsification Criteria

- Discontinuities in kernel dimension at any t
- Code distance jumps non-smoothly
- Morphism composition breaks code properties
- Stabilizer formalism handles continuous families better

## Computational Method

Interpolate between two surjections, track properties at 100 points:
```python
pi_t = (1-t) * pi_0 + t * pi_1
kernel_dim(t), code_distance(t), logical_dim(t) = analyze(pi_t)
```

## Expected Outcomes

- Kernel dimension evolves smoothly (SVD singular values cross threshold continuously)
- Code parameters as continuous functions of t
- Clear visualization of code family landscape

## Hardware Requirements

- Memory: ~10MB (100 snapshots of 16×16 matrices)
- Runtime: ~10 seconds (100 SVD decompositions)
- Dependencies: NumPy, SciPy, Matplotlib

## Timeline

Total: 7.75 hours (pre-reg 1h, setup 15m, code 4h, test 2h, archive 30m)

## Success Criteria

- [ ] Kernel dimension smooth function of t
- [ ] No discontinuities in code properties
- [ ] Functorial composition verified
- [ ] Tests pass (pytest ≥ 95%)

## Contact

Oksana Sudoma + Claude (Anthropic)
