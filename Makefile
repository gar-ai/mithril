# Mithril Development Makefile
# Usage: make <target>

.PHONY: all build test lint fmt check clean bench doc python install help

# Default target
all: check test

# ============================================================================
# Build
# ============================================================================

## Build all crates in debug mode
build:
	cargo build --workspace --all-features

## Build all crates in release mode
release:
	cargo build --workspace --all-features --release

## Build CLI tools only
cli:
	cargo build --release -p mithril-checkpoint -p mithril-dedup

# ============================================================================
# Testing
# ============================================================================

## Run all tests
test:
	cargo test --workspace --all-features

## Run tests with output
test-verbose:
	cargo test --workspace --all-features -- --nocapture

## Run tests for a specific crate (usage: make test-crate CRATE=mithril-core)
test-crate:
	cargo test -p $(CRATE) --all-features

## Run ignored tests (requires fixtures)
test-ignored:
	cargo test --workspace --all-features -- --ignored

## Run benchmarks
bench:
	cargo bench --workspace

# ============================================================================
# Code Quality
# ============================================================================

## Run all checks (fmt, clippy, test)
check: fmt-check lint test

## Check formatting
fmt-check:
	cargo fmt --all -- --check

## Format code
fmt:
	cargo fmt --all

## Run clippy lints
lint:
	cargo clippy --workspace --all-features -- -D warnings

## Run clippy with fixes
lint-fix:
	cargo clippy --workspace --all-features --fix --allow-dirty

## Security audit
audit:
	cargo audit

# ============================================================================
# Documentation
# ============================================================================

## Build documentation
doc:
	cargo doc --workspace --all-features --no-deps

## Build and open documentation
doc-open:
	cargo doc --workspace --all-features --no-deps --open

# ============================================================================
# Python
# ============================================================================

## Build Python package (development)
python-dev:
	cd crates/mithril-python && maturin develop

## Build Python package (release)
python-release:
	cd crates/mithril-python && maturin build --release

## Run Python tests
python-test:
	pytest crates/mithril-python/tests/ -v

## Install Python package locally
python-install:
	pip install crates/mithril-python/target/wheels/*.whl

# ============================================================================
# Docker (for testing S3/GCS)
# ============================================================================

## Start local S3/GCS emulators
docker-up:
	docker-compose up -d

## Stop local emulators
docker-down:
	docker-compose down

## Run S3 integration tests
test-s3:
	AWS_ACCESS_KEY_ID=mithril \
	AWS_SECRET_ACCESS_KEY=mithril123 \
	AWS_ENDPOINT_URL=http://localhost:9000 \
	cargo test --features s3 -p mithril-core -- --ignored

## Run GCS integration tests
test-gcs:
	STORAGE_EMULATOR_HOST=http://localhost:4443 \
	cargo test --features gcs -p mithril-core -- --ignored

# ============================================================================
# Fixtures
# ============================================================================

## Generate test fixtures
fixtures:
	python scripts/generate_fixtures.py

## Download HuggingFace fixtures
fixtures-hf:
	python scripts/download_hf_fixtures.py --checkpoints --datasets

# ============================================================================
# Clean
# ============================================================================

## Clean build artifacts
clean:
	cargo clean

## Clean everything including fixtures
clean-all: clean
	rm -rf fixtures/hf_checkpoints fixtures/hf_datasets

# ============================================================================
# Help
# ============================================================================

## Show this help
help:
	@echo "Mithril Development Commands"
	@echo ""
	@echo "Build:"
	@echo "  make build        - Build all crates (debug)"
	@echo "  make release      - Build all crates (release)"
	@echo "  make cli          - Build CLI tools only"
	@echo ""
	@echo "Test:"
	@echo "  make test         - Run all tests"
	@echo "  make test-verbose - Run tests with output"
	@echo "  make bench        - Run benchmarks"
	@echo ""
	@echo "Quality:"
	@echo "  make check        - Run all checks"
	@echo "  make fmt          - Format code"
	@echo "  make lint         - Run clippy"
	@echo ""
	@echo "Python:"
	@echo "  make python-dev   - Build Python package (dev)"
	@echo "  make python-test  - Run Python tests"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    - Start S3/GCS emulators"
	@echo "  make docker-down  - Stop emulators"
	@echo "  make test-s3      - Run S3 integration tests"
