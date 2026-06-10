# Delivery Plan: Porting Official Optax Test Suite

This plan outlines the steps to port the official `optax` test suite for losses and schedules into `zero-optax`. The goal is to ensure that `zero-optax` is API compatible and mathematically equivalent (1-to-1 exact outputs or `allclose` where necessary due to floating-point differences) to the original `optax`.

## Phase 1: Environment and Dependencies

- [ ] Create `requirements-test.txt` containing dependencies needed for the official test suite.
  - Required packages: `pytest`, `chex`, `jax`, `jaxlib`, `absl-py`, `numpy`, and `optax` (as the reference implementation).
- [ ] Configure `pytest` (e.g., via `pyproject.toml` or `pytest.ini`) to discover and run the ported tests.
- [ ] Establish a test utility/base testing class in `tests/` that facilitates 1-to-1 comparison between `optax` and `zero_optax` functions.

## Phase 2: Porting Losses Test Suite

- [ ] Port `classification_test.py` from `optax/losses/` to test `zero_optax.losses.classification`.
- [ ] Port `fenchel_young_test.py` from `optax/losses/` to test `zero_optax.losses.fenchel_young`.
- [ ] Port `ranking_test.py` from `optax/losses/` to test `zero_optax.losses.ranking`.
- [ ] Port `regression_test.py` from `optax/losses/` to test `zero_optax.losses.regression`.
- [ ] Port `sequence_test.py` from `optax/losses/` to test `zero_optax.losses.sequence`.
- [ ] Port `sparsemax_test.py` from `optax/losses/` to test `zero_optax.losses.sparsemax`.

## Phase 3: Porting Schedules Test Suite

- [ ] Port `cycle_test.py` from `optax/schedules/` to test `zero_optax.schedules.cycle`.
- [ ] Port `inject_test.py` from `optax/schedules/` to test `zero_optax.schedules.inject`.
- [ ] Port `schedule_test.py` from `optax/schedules/` to test `zero_optax.schedules.schedule`.
- [ ] Port `warmup_test.py` from `optax/schedules/` to test `zero_optax.schedules.warmup`.

## Phase 4: Validation and Fixing

- [ ] Run the complete ported test suite against `zero_optax`.
- [ ] Identify failing tests (e.g., API mismatch, floating point discrepancies).
- [ ] Fix any bugs or mismatches in `zero_optax` implementations to ensure 100% equivalence.
- [ ] Verify test coverage and document completeness.
