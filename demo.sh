#!/usr/bin/env bash
# Mithril ML Infrastructure Toolkit — Live Demo Script
#
# Runs all three demos in sequence for a hackathon presentation.
# Total runtime: ~30-60 seconds depending on hardware.
#
# Usage:
#   ./demo.sh          # Run all demos
#   ./demo.sh rust      # Rust hero demo only
#   ./demo.sh real      # Real model checkpoint demo only
#   ./demo.sh python    # Python training demo only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
DIM="\033[2m"
RESET="\033[0m"

header() {
    echo ""
    echo -e "${CYAN}${BOLD}$1${RESET}"
    echo -e "${DIM}$(printf '%.0s─' {1..60})${RESET}"
}

# Build if needed
build_rust() {
    if [[ ! -f target/release/examples/demo ]] || [[ ! -f target/release/examples/real_checkpoint_demo ]]; then
        header "Building Mithril (release mode)..."
        RUSTFLAGS="-Ctarget-cpu=native" cargo build --release \
            --example demo \
            --example real_checkpoint_demo \
            2>&1 | tail -1
    fi
}

run_rust_demo() {
    header "[1/3] Mithril vs Naive Checkpointing (256 MB synthetic bf16)"
    echo ""
    ./target/release/examples/demo
}

run_real_demo() {
    header "[2/3] Real Model Weight Compression (HuggingFace safetensors)"
    echo ""
    ./target/release/examples/real_checkpoint_demo
}

run_python_demo() {
    header "[3/3] Python Training Loop with Mithril (distilgpt2)"
    echo ""
    if command -v python3 &>/dev/null && python3 -c "import mithril" 2>/dev/null; then
        python3 examples/training_demo.py --steps 3 --compare
    else
        echo "  Skipping Python demo (mithril not installed in Python)"
        echo "  To enable: maturin develop --release && pip install torch transformers"
    fi
}

# Main
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║         MITHRIL ML INFRASTRUCTURE TOOLKIT                ║${RESET}"
echo -e "${CYAN}${BOLD}║         Live Demo — Hackathon 2026                       ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"

case "${1:-all}" in
    rust)
        build_rust
        run_rust_demo
        ;;
    real)
        build_rust
        run_real_demo
        ;;
    python)
        run_python_demo
        ;;
    all)
        build_rust
        run_rust_demo
        run_real_demo
        run_python_demo
        ;;
    *)
        echo "Usage: $0 [rust|real|python|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}${BOLD}  Demo complete.${RESET}"
echo ""
