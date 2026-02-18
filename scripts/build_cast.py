#!/usr/bin/env python3
"""Build a well-timed asciinema .cast file from demo output."""

import json
import subprocess
import sys

def main():
    # Run the demo and capture raw output with ANSI codes
    result = subprocess.run(
        ["./target/release/examples/demo"],
        capture_output=True,
        text=True,
        env={"TERM": "xterm-256color", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    output = result.stdout

    # Split into lines
    lines = output.split("\n")

    # Build asciicast v2 events with timing
    cols, rows = 90, 40
    header = {
        "version": 2,
        "width": cols,
        "height": rows,
        "timestamp": None,
        "title": "Mithril vs Naive Checkpointing Demo",
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
    }

    events = []
    t = 0.5  # Start after a small pause

    for line in lines:
        # Determine delay based on content
        if not line.strip():
            delay = 0.2
        elif "═══" in line or "MITHRIL" in line:
            delay = 0.8
        elif "Simulating" in line or "parameters" in line:
            delay = 0.6
        elif "Generating" in line or "Weights generated" in line:
            delay = 0.4
        elif "Step" in line and "Naive" in line:  # Table header
            delay = 0.5
        elif "───" in line:  # Separator
            delay = 0.3
        elif line.strip() and line.strip()[0].isdigit():  # Data row (step 1-10)
            delay = 0.35
        elif "Total" in line or "Time" in line:
            delay = 0.5
        elif "✓" in line:
            delay = 0.6
        elif "┌" in line or "└" in line:
            delay = 0.5
        elif "│" in line:
            delay = 0.35
        elif "Done" in line:
            delay = 0.8
        else:
            delay = 0.25

        events.append([round(t, 3), "o", line + "\r\n"])
        t += delay

    # Add final pause
    events.append([round(t + 1.0, 3), "o", ""])

    # Write cast file
    with open("demo.cast", "w") as f:
        f.write(json.dumps(header) + "\n")
        for event in events:
            f.write(json.dumps(event) + "\n")

    total_duration = events[-1][0]
    print(f"Generated demo.cast: {len(events)} frames, {total_duration:.1f}s duration")


if __name__ == "__main__":
    main()
