# Semantic Implementation Plan

This document outlines the exhaustive roadmap to achieve **100% mathematical and semantic parity** for all 35 endpoints currently mocked in `zero-optax`. 

While the codebase currently passes structural/signature compliance, this plan ensures that `zero-optax` computes exact numerical equivalents to the reference `optax` library. We will heavily depend upon and integrate with the sibling project `../zero-jax` to provide the underlying array abstractions, dynamic programming primitives, and hardware-accelerated kernels required for these implementations.

## Phase 1: Test Infrastructure & `zero-jax` Integration

Before writing algorithms, we must establish a rigorous numerical verification harness.

- [x] **`zero-jax` Dependency Linking**
  - [x] Add `zero-jax` as a local editable dependency (`file://../zero-jax`) in `pyproject.toml`.
  - [x] Configure `pytest` to resolve `zero-jax` module paths correctly.
  - [x] Verify `zero-jax` array primitives (`Array`, `numpy`-like wrappers) pass type checking within `zero-optax`.
- [x] **Numerical Equivalence Harness**
  - [x] Create `tests/utils.py` containing an `assert_allclose_optax` fixture.
  - [x] Implement parameter fuzzing using `hypothesis` (or similar) to test functions across a wide range of input shapes, types (`float32`, `bfloat16`), and boundary values (e.g., zeros, infinities).
  - [x] Setup gradient equivalence testing: assert `jax.grad(zero_optax.loss) == jax.grad(optax.loss)`.

## Phase 2: Core Regression & Binary Classification

These losses rely on standard element-wise math (powers, absolute values, basic reductions) and will serve as the initial validation of the `zero-jax` integration.

- [x] **Regression Losses**
  - [x] `squared_error`: Implement exact math and add numerical tests.
  - [x] `squared_error`: Add gradient equivalence tests.
  - [x] `l2_loss`: Implement exact math (factor of 0.5) and add numerical tests.
  - [x] `huber_loss`: Implement `jnp.where` logic for delta boundaries.
  - [x] `huber_loss`: Add numerical and gradient equivalence tests (especially at the boundary `delta`).
- [x] **Binary Classification Losses**
  - [x] `hinge_loss`: Implement standard max margin math.
  - [x] `hinge_loss`: Add numerical and gradient equivalence tests.
  - [x] `perceptron_loss`: Implement negative product math.
  - [x] `perceptron_loss`: Add numerical and gradient equivalence tests.
  - [x] `sigmoid_binary_cross_entropy`: Implement numerically stable `log_sigmoid` logic.
  - [x] `sigmoid_binary_cross_entropy`: Add numerical and gradient equivalence tests.
  - [x] `sigmoid_focal_loss`: Implement focal weights and alpha balancing.
  - [x] `sigmoid_focal_loss`: Add numerical and gradient equivalence tests.

## Phase 3: Softmax & Multiclass Cross Entropies

These require careful handling of numerical stability (e.g., `logsumexp` tricks) and custom reductions.

- [x] **Softmax Core**
  - [x] `safe_softmax_cross_entropy`: Implement with stable max-subtraction.
  - [x] `safe_softmax_cross_entropy`: Add numerical tests testing extreme logits (e.g., `+/- 1e5`).
  - [x] `softmax_cross_entropy`: Implement supporting `axis` and `where` masking via `zero-jax`.
  - [x] `softmax_cross_entropy`: Add equivalence tests for multi-dimensional axes and masks.
  - [x] `softmax_cross_entropy_with_integer_labels`: Implement integer-to-one-hot conversion and dispatch.
  - [x] `softmax_cross_entropy_with_integer_labels`: Add numerical and gradient equivalence tests.
- [x] **Multiclass Margin - [ ] **Multiclass Margin & Poly Losses** Poly Losses**
  - [x] `multiclass_hinge_loss`: Implement multi-class margin logic.
  - [x] `multiclass_hinge_loss`: Add numerical and gradient equivalence tests.
  - [x] `multiclass_perceptron_loss`: Implement multi-class perceptron logic.
  - [x] `multiclass_perceptron_loss`: Add numerical and gradient equivalence tests.
  - [x] `poly_loss_cross_entropy`: Implement Taylor expansion adjustment on top of cross entropy.
  - [x] `poly_loss_cross_entropy`: Add numerical and gradient equivalence tests.

## Phase 4: Advanced Losses (Heavy `zero-jax` Dependency)

These losses require complex algorithmic primitives like dynamic programming (`jax.lax.scan`), sorting, and custom gradients. We will rely heavily on `zero-jax` capabilities here.

- [x] **Sequence Modeling (CTC)**
  - [x] `ctc_loss_with_forward_probs`: Implement the forward-backward algorithm using `zero-jax` loop primitives (e.g., `scan` or custom kernels).
  - [x] `ctc_loss_with_forward_probs`: Test against varying sequence lengths and padding masks.
  - [x] `ctc_loss`: Wrap `ctc_loss_with_forward_probs` and discard probabilities.
  - [x] `ctc_loss`: Add gradient equivalence tests (CTC gradients are notoriously tricky; exact matching is required).
- [x] **Sparsemax - [ ] **Sparsemax & Ranking** Ranking**
  - [x] `sparsemax_loss`: Implement the sparsemax projection algorithm (sorting/bisection logic using `zero-jax`).
  - [x] `sparsemax_loss`: Add numerical and gradient equivalence tests.
  - [x] `multiclass_sparsemax_loss`: Implement multi-class variant.
  - [x] `multiclass_sparsemax_loss`: Add numerical and gradient equivalence tests.
  - [x] `ranking_softmax_loss`: Implement pairwise ranking formulations.
  - [x] `ranking_softmax_loss`: Add numerical tests with custom `reduce_fn` and `weights`.
- [x] **Fenchel-Young**
  - [x] `make_fenchel_young_loss`: Implement convex conjugate abstractions.
  - [x] `make_fenchel_young_loss`: Add tests verifying custom max-functions generate correct loss closures.

## Phase 5: Base Learning Rate Schedules

Schedules are pure functions `(int) -> float`. While mathematically simple, their exact floating-point behavior across steps must match perfectly.

- [x] **Continuous / Basic Schedules**
  - [x] `constant_schedule`: Implement math and verify types.
  - [x] `linear_schedule`: Implement exact linear interpolation.
  - [x] `linear_schedule`: Add boundary tests (step < 0, step > transition_steps).
  - [x] `polynomial_schedule`: Implement polynomial interpolation.
  - [x] `polynomial_schedule`: Test varying power factors (e.g., 0.5, 2.0).
  - [x] `exponential_decay`: Implement continuous exponential decay.
  - [x] `exponential_decay`: Add tests for `staircase=True` (discrete jumps) and `staircase=False`.
  - [x] `cosine_decay_schedule`: Implement cosine annealing math.
  - [x] `cosine_decay_schedule`: Add numerical tests for `alpha` > 0 limits.
- [x] **Piecewise Schedules**
  - [x] `piecewise_constant_schedule`: Implement interval lookup (using `zero-jax` searchsorted or similar).
  - [x] `piecewise_constant_schedule`: Add boundary edge-case tests.
  - [x] `piecewise_interpolate_schedule`: Implement piece-wise combinations of linear/cosine interpolations.
  - [x] `piecewise_interpolate_schedule`: Add numerical tests.

## Phase 6: Composite & Cyclic Schedules

These schedules compose other schedules or implement complex cyclical behavior.

- [x] **Warmup Wrappers**
  - [x] `warmup_constant_schedule`: Implement piecewise logic combining linear warmup with constant.
  - [x] `warmup_constant_schedule`: Add numerical tests.
  - [x] `warmup_cosine_decay_schedule`: Implement warmup + cosine decay.
  - [x] `warmup_cosine_decay_schedule`: Add numerical tests.
  - [x] `warmup_exponential_decay_schedule`: Implement warmup + exponential decay.
  - [x] `warmup_exponential_decay_schedule`: Add numerical tests.
- [x] **Cycles and Joins**
  - [x] `join_schedules`: Implement sequential boundary dispatch.
  - [x] `join_schedules`: Add numerical tests across 3+ combined schedules.
  - [x] `cosine_onecycle_schedule`: Implement standard 1-cycle math (ascend, descend, annihilate).
  - [x] `cosine_onecycle_schedule`: Add numerical tests.
  - [x] `linear_onecycle_schedule`: Implement linear 3-phase 1-cycle math.
  - [x] `linear_onecycle_schedule`: Add numerical tests.
  - [x] `sgdr_schedule`: Implement Stochastic Gradient Descent with Warm Restarts (Cosine Annealing).
  - [x] `sgdr_schedule`: Add numerical tests across multiple restart boundaries.

## Phase 7: Meta-Optimizers & Injection

These are complex transformations that interact directly with the optimizer state and `GradientTransformation` objects.

- [x] **Hyperparameter Injection**
  - [x] `inject_hyperparams`: Implement tree-mapping logic to inject non-static hyperparams into inner transformations.
  - [x] `inject_hyperparams`: Write tests verifying dynamic learning rate updates work exactly as in `optax`.
  - [x] `inject_stateful_hyperparams`: Implement stateful tracking of injected parameters.
  - [x] `inject_stateful_hyperparams`: Write tests verifying state propagation.

## Phase 8: Final Validation

- [x] Run full `zero-optax` test suite with `zero-jax` backend enabled.
- [x] Verify 100% test coverage is maintained.
- [x] Run `ml_framework_snapshots` to guarantee no structural regressions occurred during semantic implementation.
