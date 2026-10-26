#!/bin/bash
set -e
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
echo "Setting up local editable environment..."
pip install "ml-switcheroo-ir @ https://github.com/SamuelMarks/ml-switcheroo-ir/releases/download/v0.0.4/ml_switcheroo_ir-0.0.4-py3-none-any.whl" || pip install -e ../ml-switcheroo-ir
pip install "ml-switcheroo-compiler @ https://github.com/SamuelMarks/ml-switcheroo-compiler/releases/download/v0.0.1/ml_switcheroo_compiler-0.0.1-py3-none-any.whl" || pip install -e ../ml-switcheroo-compiler
if [ -d "../griffe/packages/griffelib" ]; then
  pip install --ignore-requires-python --no-deps -e ../griffe/packages/griffelib -e ../griffe/packages/griffecli -e ../griffe
fi
pip install "cdd @ git+https://github.com/offscale/cdd-python.git@master" || pip install -e ../cdd-python
pip install --no-deps "ml-ecosystem-snapshots @ https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl" || pip install --no-deps -e ../ml-ecosystem-snapshots
mkdir -p ~/.cache/ml_ecosystem_snapshots/snapshots
curl -fsSL "https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/optax_v0.2.8.json" -o ~/.cache/ml_ecosystem_snapshots/snapshots/optax_v0.2.8.json || true
pip install --no-deps -e .
echo "Done!"
