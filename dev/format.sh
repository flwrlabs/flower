#!/bin/bash
# Format examples and benchmarks
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

taplo fmt

# Examples
# Iterate over all example directories
for dir in ../examples/*/; do
    src_args=$(find "$dir" -maxdepth 1 -type d | sed 's/^/--src /' | tr '\n' ' ')
    python -m isort $dir $src_args --settings-path . &
done
wait
python -m black -q ../examples

# Benchmarks
python -m isort ../benchmarks
python -m black -q ../benchmarks

# Notebooks
KEYS="metadata.celltoolbar metadata.language_info metadata.toc metadata.notify_time metadata.varInspector metadata.accelerator metadata.vscode cell.metadata.id cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell cell.metadata.vscode cell.metadata.pycharm"
python -m nbstripout ../examples/*/*.ipynb --extra-keys "$KEYS"

# Markdown
python -m mdformat --number ../examples
