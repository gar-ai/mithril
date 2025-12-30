"""Mithril CLI - Checkpoint compression and management.

Usage:
    mithril compress <input> <output> [options]
    mithril decompress <input> <output>
    mithril info <file>
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()


def get_file_size_str(size_bytes: int) -> str:
    """Format file size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


@click.group()
@click.version_option(package_name="mithril")
def main():
    """Mithril - High-performance ML checkpoint compression.
    
    Examples:
    
        # Compress a file
        mithril compress model.safetensors model.mcp
        
        # Decompress a file
        mithril decompress model.mcp model.safetensors
        
        # Show file info
        mithril info model.mcp
    """
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--level", "-l", default=3, type=click.IntRange(1, 22),
              help="Compression level (1=fastest, 22=best). Default: 3")
@click.option("--byte-grouping/--no-byte-grouping", default=True,
              help="Enable byte grouping for better compression. Default: enabled")
@click.option("--force", "-f", is_flag=True, help="Overwrite output file if exists")
def compress(input_path: str, output_path: str, level: int, byte_grouping: bool, force: bool):
    """Compress a checkpoint file.
    
    Supports any binary file. For best results with tensor data,
    use raw tensor bytes (not pickle format).
    """
    from mithril import CompressionConfig, CheckpointCompressor
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if output_file.exists() and not force:
        console.print(f"[red]Error:[/red] Output file '{output_path}' already exists. Use --force to overwrite.")
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Read input
        task = progress.add_task("Reading input...", total=None)
        data = input_file.read_bytes()
        original_size = len(data)
        progress.update(task, completed=True)
        
        # Compress
        task = progress.add_task("Compressing...", total=None)
        config = CompressionConfig(zstd_level=level, byte_grouping=byte_grouping)
        compressor = CheckpointCompressor(config)
        compressed = compressor.compress_raw(data)
        compressed_size = len(compressed)
        progress.update(task, completed=True)
        
        # Write output
        task = progress.add_task("Writing output...", total=None)
        output_file.write_bytes(compressed)
        progress.update(task, completed=True)
    
    ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    savings = original_size - compressed_size
    savings_pct = (savings / original_size * 100) if original_size > 0 else 0
    
    console.print()
    console.print(f"[green]Compression complete![/green]")
    console.print(f"  Input:       {get_file_size_str(original_size)}")
    console.print(f"  Output:      {get_file_size_str(compressed_size)}")
    console.print(f"  Ratio:       {ratio:.2f}x")
    console.print(f"  Savings:     {get_file_size_str(savings)} ({savings_pct:.1f}%)")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--force", "-f", is_flag=True, help="Overwrite output file if exists")
def decompress(input_path: str, output_path: str, force: bool):
    """Decompress a compressed checkpoint file."""
    from mithril import CompressionConfig, CheckpointCompressor
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if output_file.exists() and not force:
        console.print(f"[red]Error:[/red] Output file '{output_path}' already exists. Use --force to overwrite.")
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Read input
        task = progress.add_task("Reading compressed data...", total=None)
        compressed = input_file.read_bytes()
        progress.update(task, completed=True)
        
        # We need to know original size - for raw compression, it's stored in header
        # For now, we'll use a simple approach that requires user to know size
        # or we detect from file format
        task = progress.add_task("Decompressing...", total=None)
        compressor = CheckpointCompressor(CompressionConfig())
        
        # Try to decompress - the Rust side handles size detection for zstd
        # For compress_raw, we need to provide estimated size
        # This is a limitation - in production, we'd store original size in header
        try:
            # Zstd stores original size in frame, so we can use max reasonable size
            # and let zstd handle it
            decompressed = compressor.decompress_raw(compressed, 0)  # 0 = auto-detect
        except Exception:
            # Fallback: try with a large buffer
            decompressed = compressor.decompress_raw(compressed, len(compressed) * 20)
        
        progress.update(task, completed=True)
        
        # Write output
        task = progress.add_task("Writing output...", total=None)
        output_file.write_bytes(decompressed)
        progress.update(task, completed=True)
    
    console.print()
    console.print(f"[green]Decompression complete![/green]")
    console.print(f"  Output: {get_file_size_str(len(decompressed))}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
def info(file_path: str):
    """Show information about a checkpoint file."""
    file = Path(file_path)
    size = file.stat().st_size
    
    table = Table(title=f"File: {file.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Path", str(file))
    table.add_row("Size", get_file_size_str(size))
    
    # Try to detect format
    with open(file, "rb") as f:
        magic = f.read(8)
    
    format_name = "Unknown"
    if magic[:4] == b"MCPT":
        format_name = "Mithril Checkpoint (MCP)"
    elif magic[:8] == b"GGUF\x00\x00\x00\x03" or magic[:4] == b"GGUF":
        format_name = "GGUF (llama.cpp)"
    elif magic[:8] == b"\x00\x00\x00\x00\x00\x00\x00\x08":
        format_name = "Safetensors"
    elif magic[:2] == b"\x80\x04" or magic[:2] == b"\x80\x05":
        format_name = "PyTorch Pickle"
    elif magic[:4] == b"\x28\xb5\x2f\xfd":
        format_name = "Zstd Compressed"
    
    table.add_row("Format", format_name)
    
    console.print(table)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--from", "from_format", type=click.Choice(["safetensors", "pt", "auto"]), default="auto",
              help="Input format")
@click.option("--to", "to_format", type=click.Choice(["mst", "safetensors"]), default="mst",
              help="Output format")
@click.option("--level", "-l", default=3, type=click.IntRange(1, 22),
              help="Compression level for MST output")
def convert(input_path: str, output_path: str, from_format: str, to_format: str, level: int):
    """Convert between checkpoint formats.
    
    Currently supports:
    - safetensors -> mst (compressed safetensors)
    - mst -> safetensors
    """
    console.print(f"[yellow]Note:[/yellow] Format conversion is not yet fully implemented.")
    console.print(f"Would convert: {input_path} ({from_format}) -> {output_path} ({to_format})")
    console.print(f"Compression level: {level}")


if __name__ == "__main__":
    main()
