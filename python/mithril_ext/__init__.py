"""Mithril Python extensions.

This package provides Python-native integrations for Mithril:
- checkpoint.torch: PyTorch DCP SavePlanner/LoadPlanner

The core mithril module (compression, cache, dedup) is provided by the
Rust bindings installed via maturin. Import those directly:

    import mithril  # Rust bindings
    from mithril_ext.checkpoint.torch import MithrilSavePlanner  # Python extensions
"""

# This package only provides Python extensions (like DCP integration).
# The core mithril module is the Rust/PyO3 package.
