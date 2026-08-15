#!/usr/bin/env sh
set -eu

python -m ruff check . --config src/turkicdocgen/configs/project/ruff.toml
python -m pytest tests -v --tb=short -c src/turkicdocgen/configs/project/pytest.ini
cargo fmt --check
cargo test --workspace
