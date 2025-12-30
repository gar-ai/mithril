"""Pytest configuration for integration tests."""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary directory for cache tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_state_dict():
    """Provide a sample PyTorch state dict for testing."""
    torch = pytest.importorskip("torch")

    return {
        "layer1.weight": torch.randn(256, 128, dtype=torch.bfloat16),
        "layer1.bias": torch.randn(256, dtype=torch.bfloat16),
        "layer2.weight": torch.randn(128, 256, dtype=torch.bfloat16),
        "layer2.bias": torch.randn(128, dtype=torch.bfloat16),
    }
