#!/usr/bin/env bash
# Record a polished demo video with natural pacing.
# The demo binary runs too fast for a good video, so we replay
# its output with artificial timing to make it watchable.

set -euo pipefail

# Capture the actual demo output (with ANSI colors)
OUTPUT=$(script -q /dev/null ./target/release/examples/demo)

# Print it line-by-line with delays that feel natural
echo "$OUTPUT" | while IFS= read -r line; do
    echo "$line"
    # Longer pause after section headers, shorter for data rows
    if echo "$line" | grep -q '═══\|───\|Simulating\|▶\|✓.*All\|✓.*Throughput\|Done'; then
        sleep 0.6
    elif echo "$line" | grep -q '^\s*[0-9].*│\|Total.*│'; then
        sleep 0.3
    elif echo "$line" | grep -q '┌\|└\|│'; then
        sleep 0.4
    else
        sleep 0.15
    fi
done
