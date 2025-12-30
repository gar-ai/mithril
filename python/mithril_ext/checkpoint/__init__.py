"""Mithril checkpoint Python extensions.

Provides PyTorch DCP integration:
- MithrilSavePlanner: Compress tensors during DCP save
- MithrilLoadPlanner: Decompress tensors during DCP load

For core compression functionality, import from the Rust bindings:
    from mithril import CheckpointCompressor, CompressionConfig
"""

# PyTorch DCP integration (requires torch)
try:
    from mithril_ext.checkpoint.torch import (
        MithrilSavePlanner,
        MithrilLoadPlanner,
        compress_state_dict,
        decompress_state_dict,
    )
    __all__ = [
        "MithrilSavePlanner",
        "MithrilLoadPlanner",
        "compress_state_dict",
        "decompress_state_dict",
    ]
except ImportError:
    # torch not installed
    __all__ = []
