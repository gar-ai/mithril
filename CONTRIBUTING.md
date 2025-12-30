# Contributing to Mithril

Thank you for your interest in contributing to Mithril! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Environment](#development-environment)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)

## Development Environment

### Prerequisites

- **Rust**: 1.75 or later (check with `rustc --version`)
- **Python**: 3.9 or later (check with `python --version`)
- **Git**: For version control

### Recommended Tools

- **rustup**: Rust toolchain manager
- **uv**: Fast Python package manager (recommended)
- **maturin**: Build tool for Python/Rust bindings

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/mithril.git
   cd mithril
   ```

2. Install Rust toolchain:
   ```bash
   rustup update stable
   rustup component add rustfmt clippy
   ```

3. Create Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install Python dependencies:
   ```bash
   pip install maturin torch pytest
   ```

## Building from Source

### Rust Crates

Build all Rust crates:
```bash
cargo build --workspace
```

Build in release mode:
```bash
cargo build --workspace --release
```

Build with optional features:
```bash
cargo build --workspace --features s3
```

### Python Package

Build and install the Python package for development:
```bash
cd crates/mithril-python
maturin develop
```

Or with release optimizations:
```bash
maturin develop --release
```

## Running Tests

### Rust Tests

Run all tests:
```bash
cargo test --workspace
```

Run tests for a specific crate:
```bash
cargo test -p mithril-checkpoint
cargo test -p mithril-dedup
cargo test -p mithril-cache
```

Run tests with features:
```bash
cargo test --workspace --features s3
```

### Python Tests

Run Python tests:
```bash
cd crates/mithril-python
pytest tests/
```

Run with verbose output:
```bash
pytest tests/ -v
```

### Benchmarks

Run benchmarks:
```bash
cargo bench --workspace
```

Run specific benchmark:
```bash
cargo bench -p mithril-checkpoint
```

## Code Style

### Rust

We use `rustfmt` for formatting and `clippy` for linting.

Format code:
```bash
cargo fmt --all
```

Check formatting:
```bash
cargo fmt --all -- --check
```

Run clippy:
```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Python

We use `ruff` for formatting and linting.

Format code:
```bash
ruff format .
```

Check linting:
```bash
ruff check .
```

### Documentation

Ensure documentation compiles without warnings:
```bash
cargo doc --workspace --no-deps
```

## Pull Request Process

1. **Fork the repository** and create a new branch from `main`.

2. **Make your changes** following the code style guidelines.

3. **Add tests** for new functionality.

4. **Update documentation** if needed.

5. **Run all checks locally**:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo doc --workspace --no-deps
   ```

6. **Submit a pull request** with a clear description of your changes.

### PR Requirements

- All CI checks must pass
- Code must be formatted with `rustfmt`
- No clippy warnings
- Tests must pass
- Documentation must compile
- At least one maintainer approval required

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, etc.

### Examples

```
feat(checkpoint): add delta encoding for training checkpoints

Implements XOR-based delta compression between consecutive checkpoints,
achieving 10-3500x compression ratios for typical training scenarios.

Closes #123
```

```
fix(dedup): handle empty documents in MinHash computation

Previously, empty documents caused a panic in the signature computation.
Now they are handled gracefully with a zero signature.
```

```
docs: update README with new benchmark results
```

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our community Discord (link TBD)

## License

By contributing to Mithril, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.
