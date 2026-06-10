# Delivery Plan: Porting Official Optax Test Suite

This plan outlines the steps to port the official `optax` test suite for losses and schedules into `zero-optax`. The goal is to ensure that `zero-optax` is API compatible and mathematically equivalent (1-to-1 exact outputs or `allclose` where necessary due to floating-point differences) to the original `optax`.

## Phase 1: Environment and Dependencies

- [x] Create `requirements-test.txt` containing dependencies needed for the official test suite.
  - Required packages: `pytest`, `chex`, `jax`, `jaxlib`, `absl-py`, `numpy`, and `optax` (as the reference implementation).
- [x] Configure `pytest` (e.g., via `pyproject.toml` or `pytest.ini`) to discover and run the ported tests.
- [x] Establish a test utility/base testing class in `tests/` that facilitates 1-to-1 comparison between `optax` and `zero_optax` functions.

## Phase 2: Porting Losses Test Suite

- [x] Port `classification_test.py` from `optax/losses/` to test `zero_optax.losses.classification`.
- [x] Port `fenchel_young_test.py` from `optax/losses/` to test `zero_optax.losses.fenchel_young`.
- [x] Port `ranking_test.py` from `optax/losses/` to test `zero_optax.losses.ranking`.
- [x] Port `regression_test.py` from `optax/losses/` to test `zero_optax.losses.regression`.
- [x] Port `sequence_test.py` from `optax/losses/` to test `zero_optax.losses.sequence`.
- [x] Port `sparsemax_test.py` from `optax/losses/` to test `zero_optax.losses.sparsemax`.

## Phase 3: Porting Schedules Test Suite

- [x] Port `cycle_test.py` from `optax/schedules/` to test `zero_optax.schedules.cycle`.
- [x] Port `inject_test.py` from `optax/schedules/` to test `zero_optax.schedules.inject`.
- [x] Port `schedule_test.py` from `optax/schedules/` to test `zero_optax.schedules.schedule`.
- [x] Port `warmup_test.py` from `optax/schedules/` to test `zero_optax.schedules.warmup`.

## Phase 4: Validation and Fixing

- [x] Run the complete ported test suite against `zero_optax`.
- [x] Identify failing tests (e.g., API mismatch, floating point discrepancies).
- [x] Fix any bugs or mismatches in `zero_optax` implementations to ensure 100% equivalence.
- [x] Verify test coverage and document completeness.

## Phase 5: Code Quality & Coverage Enforcement

- [x] 100% Doc coverage (modules, classes, methods, args covered by docstrings).
- [x] 100% Test coverage (lines, branches evaluated).
- [x] Strong Typing (mypy passed with fully-typed signatures).
- [x] Strict dependency constraints (No outside dependencies mapped beyond standard list).
- [x] Pre-commit hooks passing (Ruff formatting, Linting, Badges updated, Pytest executing).
