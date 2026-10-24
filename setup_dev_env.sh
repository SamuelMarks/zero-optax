#!/bin/bash
set -e
echo "Setting up local editable environment..."
pip install -e ../ml-switcheroo-ir
pip install -e ../ml-ecosystem-snapshots
pip install -e ../ml-switcheroo-compiler
pip install -e .
echo "Done!"
