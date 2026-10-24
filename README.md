# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-optax

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](https://github.com/SamuelMarks/zero-optax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
[![API Compliance](https://img.shields.io/badge/api_compliance-100%25-brightgreen.svg)](#)

A clean implementation of the [optax](https://github.com/google-deepmind/optax) API with **strictly zero heavy external dependencies** (relying solely on the [Python Standard Library](https://docs.python.org/3/library/), `ml-switcheroo-ir`, and `ml-switcheroo-compiler`).

Targeting API snapshot version: **0.2.8** (verified via `ml-ecosystem-snapshots` `v0.0.4`).

## Why this project exists

`zero-optax` is part of the broader **Abstract ML Machine Ecosystem** (the `zero-*` and `ml-switcheroo-*` projects), which is designed to solve the $N 	imes M$ translation problem in [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning). Instead of writing bespoke translators for every framework ([JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/), [Keras](https://keras.io/)) to every target ([WASM](https://webassembly.org/), [WebGPU](https://www.w3.org/TR/webgpu/), [TensorRT](https://developer.nvidia.com/tensorrt)), the ecosystem traces $N$ frontends into a strictly defined [Intermediate Representation (IR)](https://en.wikipedia.org/wiki/Intermediate_representation), which is then consumed by $M$ backends. This achieves a source-to-source and source-to-browser compilation pipeline utilizing strictly zero heavy dependencies.

As part of this hierarchy (Tier 4: Neural Networks & Frontends), `zero-optax` provides standard optimization schedules, gradient transformations, and loss functions matching [Google DeepMind](https://deepmind.google/)'s `optax` API directly on top of `ml-switcheroo-ir` and `ml-switcheroo-compiler`.

### Key Objectives

- **API Parity:** To serve as a drop-in semantic replacement for the real `optax`. It exactly replicates the public signatures, defaults, and type behavior, verified automatically via the `ml-ecosystem-snapshots` (`v0.0.4`) checker tool against the original framework (`optax_v0.2.8.json`).
- **Pure Python & Eager Evaluation:** Operations are implemented on top of `ml-switcheroo-compiler` primitives, making the code lightweight and transparent without relying on `jaxlib` or [XLA](https://openxla.org/) C++ binaries.
- **Tracing & Compilation:** Fully integrates with `ml-switcheroo-compiler`'s `Tensor` and `TracerTape`, enabling smooth reverse-mode automatic differentiation and generating [WASM](https://webassembly.org/)/WGSL executable browser payloads through the canonical [ONNX](https://onnx.ai/)-based Logical Graph Dialect (`ml-switcheroo-ir`).
- **Golden Seed Testing:** As part of `zero-zoo`, models utilizing `zero-optax` optimizers are trained deterministically to assert float-for-float `.allclose()` equivalence against the actual `optax` implementations.

## Current Feature-Set

The following modules from the [optax](https://github.com/google-deepmind/optax) API are fully implemented and verified via [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) and [ml-ecosystem-snapshots](https://github.com/SamuelMarks/ml-ecosystem-snapshots):

- **Optimizers:** `adabelief`, `adagrad`, `adam`, `adamw`, `fromage`, `lamb`, `lars`, `lion`, `novograd`, `rmsprop`, `sgd`, `yogi`
- **Losses:** `classification`, `dice`, `fenchel_young`, `metric`, `ranking`, `regression`, `sequence`, `sparsemax`
- **Schedules:** `base`, `cycle`, `inject`, `schedule`, `warmup`

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
