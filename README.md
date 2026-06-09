# zero-optax

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-optax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-optax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

A clean implementation of the [optax](https://github.com/deepmind/optax) API with **strictly zero external dependencies** (relying solely on the Python Standard Library and `numpy` for eager evaluations). 

Targeting API snapshot version: **0.2.4**.

## Why this project exists

`zero-optax` is part of the broader **Abstract ML Machine Ecosystem** (the `zero-*` and `ml-switcheroo-*` projects), which is designed to solve the $N \times M$ translation problem in Machine Learning. Instead of writing bespoke translators for every framework (JAX, PyTorch, Keras) to every target (WASM, WebGPU, TensorRT), the ecosystem traces $N$ frontends into a strictly defined Intermediate Representation (IR), which is then consumed by $M$ backends. This achieves a source-to-source and source-to-browser compilation pipeline utilizing strictly zero heavy dependencies.

As part of this hierarchy (Tier 4: Neural Networks & Frontends), `zero-optax` provides standard optimization schedules, gradient transformations, and loss functions matching Google DeepMind's `optax` API. It builds on `zero-jax` and `zero-chex` to maintain full typing and shape assertion compatibility.

### Key Objectives

- **API Parity:** To serve as a drop-in semantic replacement for the real `optax`. It exactly replicates the public signatures, defaults, and type behavior, verified automatically via the `ml-framework-snapshots` checker tool against the original framework.
- **Pure Python & Eager Evaluation:** Operations are implemented on top of Python primitives and `numpy`, making the code incredibly lightweight and transparent without relying on `jaxlib` or XLA C++ binaries. 
- **Tracing & Compilation:** Fully integrates with the `ml-switcheroo-compiler`'s ProxyTensors, enabling smooth reverse-mode automatic differentiation (via TracerTape) and generating WASM/WGSL executable browser payloads through the canonical ONNX-based Logical Graph Dialect.
- **Golden Seed Testing:** As part of `zero-zoo`, models utilizing `zero-optax` optimizers are trained deterministically to assert float-for-float `.allclose()` equivalence against the actual `optax` implementations.

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
