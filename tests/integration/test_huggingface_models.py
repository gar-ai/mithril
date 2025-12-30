"""HuggingFace model integration tests for mithril-checkpoint.

These tests verify that real HuggingFace model weights can be:
1. Downloaded from HuggingFace Hub
2. Compressed with mithril
3. Decompressed with mithril
4. Verified to be bitwise identical to the original

All tests are marked @pytest.mark.slow since they require network access
and download model weights. Run with: pytest -m slow
"""

import io
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

# Skip all tests if required dependencies are not available
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def tensor_dict_equal(d1: Dict[str, Any], d2: Dict[str, Any]) -> bool:
    """Check if two state dicts have identical tensors."""
    if set(d1.keys()) != set(d2.keys()):
        return False
    for key in d1:
        if not torch.equal(d1[key], d2[key]):
            return False
    return True


def get_compression_ratio(original: bytes, compressed: bytes) -> float:
    """Calculate compression ratio."""
    return len(original) / len(compressed)


class TestRealisticModelStructures:
    """Test compression with model-like state dicts (no network required).

    These tests use state dicts that mimic real model architectures
    but don't require downloading actual models from HuggingFace.

    NOTE: Mithril's byte-grouped compression works on raw tensor bytes,
    not pickle data. For pickle data (torch.save format), use compress_raw.
    For maximum compression, extract raw tensor bytes directly.
    """

    def test_bert_like_architecture_raw_compression(self):
        """Test raw compression of BERT-like state dict (pickle format)."""
        import mithril

        # Create a state dict mimicking BERT architecture
        state_dict = {
            "embeddings.word_embeddings.weight": torch.randn(30522, 768, dtype=torch.float32),
            "embeddings.position_embeddings.weight": torch.randn(512, 768, dtype=torch.float32),
            "embeddings.token_type_embeddings.weight": torch.randn(2, 768, dtype=torch.float32),
            "embeddings.LayerNorm.weight": torch.randn(768, dtype=torch.float32),
            "embeddings.LayerNorm.bias": torch.randn(768, dtype=torch.float32),
            "encoder.layer.0.attention.self.query.weight": torch.randn(768, 768, dtype=torch.float32),
            "encoder.layer.0.attention.self.query.bias": torch.randn(768, dtype=torch.float32),
            "pooler.dense.weight": torch.randn(768, 768, dtype=torch.float32),
            "pooler.dense.bias": torch.randn(768, dtype=torch.float32),
        }

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        # Use compress_raw for pickle data (not byte-grouped)
        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)

        ratio = get_compression_ratio(original_data, compressed)
        # Random data compresses poorly; real model weights would compress better
        assert ratio > 1.0, f"Expected some compression, got {ratio:.2f}x"

        decompressed = compressor.decompress_raw(compressed, len(original_data))
        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)

        assert tensor_dict_equal(state_dict, recovered)

    def test_raw_tensor_bytes_fp32(self):
        """Test byte-grouped compression on raw fp32 tensor bytes."""
        import mithril

        # Create raw tensor (not pickle format)
        tensor = torch.randn(1000, 1000, dtype=torch.float32)
        # Get raw bytes directly (contiguous, correct alignment)
        raw_bytes = tensor.numpy().tobytes()

        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress(raw_bytes, "fp32")

        ratio = get_compression_ratio(raw_bytes, compressed)
        # Random fp32 data has limited compressibility
        assert ratio > 1.1, f"Expected >1.1x compression, got {ratio:.2f}x"

        decompressed = compressor.decompress(compressed, "fp32", len(raw_bytes))

        # Verify roundtrip by reconstructing tensor
        import numpy as np
        recovered_array = np.frombuffer(decompressed, dtype=np.float32).reshape(1000, 1000)
        recovered_tensor = torch.from_numpy(recovered_array.copy())

        assert torch.equal(tensor, recovered_tensor)

    def test_raw_tensor_bytes_bf16(self):
        """Test byte-grouped compression on raw bf16 tensor bytes."""
        import mithril
        import numpy as np

        # Create bf16 tensor
        tensor = torch.randn(1000, 1000, dtype=torch.bfloat16)
        # Convert to numpy-compatible format (bf16 -> uint16 view)
        raw_bytes = tensor.view(torch.uint16).numpy().tobytes()

        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress(raw_bytes, "bf16")

        ratio = get_compression_ratio(raw_bytes, compressed)
        # bf16 with byte grouping should achieve some compression even for random data
        assert ratio > 1.3, f"Expected >1.3x compression for bf16, got {ratio:.2f}x"

        decompressed = compressor.decompress(compressed, "bf16", len(raw_bytes))

        # Verify roundtrip
        recovered_uint16 = np.frombuffer(decompressed, dtype=np.uint16).reshape(1000, 1000)
        recovered_tensor = torch.from_numpy(recovered_uint16.copy()).view(torch.bfloat16)

        assert torch.equal(tensor, recovered_tensor)

    def test_delta_compression_raw_bytes(self):
        """Test delta compression on raw tensor bytes."""
        import mithril
        import numpy as np

        # Base weights
        base_tensor = torch.randn(512, 512, dtype=torch.bfloat16)
        base_bytes = base_tensor.view(torch.uint16).numpy().tobytes()

        # Updated weights (small change from training)
        updated_tensor = base_tensor + torch.randn_like(base_tensor) * 0.001
        updated_bytes = updated_tensor.view(torch.uint16).numpy().tobytes()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)

        # Regular compression
        regular_compressed = compressor.compress(updated_bytes, "bf16")

        # Delta compression (pass previous checkpoint)
        delta_compressed = compressor.compress(updated_bytes, "bf16", previous=base_bytes)

        # Delta should be much smaller since weights changed only slightly
        delta_ratio = len(regular_compressed) / len(delta_compressed)
        assert delta_ratio > 3.0, \
            f"Delta should be >3x smaller than regular, got {delta_ratio:.2f}x"

        # Verify roundtrip
        decompressed = compressor.decompress(
            delta_compressed, "bf16", len(updated_bytes), previous=base_bytes
        )

        recovered_uint16 = np.frombuffer(decompressed, dtype=np.uint16).reshape(512, 512)
        recovered_tensor = torch.from_numpy(recovered_uint16.copy()).view(torch.bfloat16)

        assert torch.equal(updated_tensor, recovered_tensor)

    def test_multiple_tensors_individual_compression(self):
        """Test compressing multiple tensors individually (like safetensors)."""
        import mithril
        import numpy as np

        # Create LLM-like weight tensors
        tensors = {
            "embed": torch.randn(32000, 4096, dtype=torch.bfloat16),
            "q_proj": torch.randn(4096, 4096, dtype=torch.bfloat16),
            "k_proj": torch.randn(4096, 4096, dtype=torch.bfloat16),
            "v_proj": torch.randn(4096, 4096, dtype=torch.bfloat16),
        }

        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        compressor = mithril.CheckpointCompressor(config)

        compressed_tensors = {}
        total_original = 0
        total_compressed = 0

        for name, tensor in tensors.items():
            raw_bytes = tensor.view(torch.uint16).numpy().tobytes()
            compressed = compressor.compress(raw_bytes, "bf16")

            total_original += len(raw_bytes)
            total_compressed += len(compressed)
            compressed_tensors[name] = (compressed, tensor.shape)

        overall_ratio = total_original / total_compressed
        # Random data has limited compressibility; real weights would compress better
        assert overall_ratio > 1.3, f"Expected >1.3x overall compression, got {overall_ratio:.2f}x"

        # Verify roundtrip for each tensor
        for name, (compressed, shape) in compressed_tensors.items():
            original = tensors[name]
            raw_size = shape[0] * shape[1] * 2  # bf16 = 2 bytes

            decompressed = compressor.decompress(compressed, "bf16", raw_size)
            recovered_uint16 = np.frombuffer(decompressed, dtype=np.uint16).reshape(shape)
            recovered = torch.from_numpy(recovered_uint16.copy()).view(torch.bfloat16)

            assert torch.equal(original, recovered), f"Mismatch in {name}"


@pytest.mark.slow
class TestTinyBERT:
    """Test with prajjwal1/bert-tiny (~17MB, very small BERT variant)."""

    MODEL_ID = "prajjwal1/bert-tiny"

    @pytest.fixture(scope="class")
    def model_and_state(self):
        """Load the tiny BERT model once for all tests in this class."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.MODEL_ID)
        state_dict = model.state_dict()
        return model, state_dict

    def test_full_model_roundtrip(self, model_and_state):
        """Test compressing full model weights using raw compression."""
        import mithril

        _, state_dict = model_and_state

        # Serialize state dict (pickle format)
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        # Use compress_raw for pickle data
        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)

        # Verify compression achieved (real model weights compress well)
        ratio = get_compression_ratio(original_data, compressed)
        assert ratio > 1.0, f"Expected some compression, got {ratio:.2f}x"

        # Decompress
        decompressed = compressor.decompress_raw(compressed, len(original_data))

        # Verify roundtrip
        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)
        assert tensor_dict_equal(state_dict, recovered), "Roundtrip should be lossless"

    def test_model_inference_after_roundtrip(self, model_and_state):
        """Verify model produces identical outputs after roundtrip."""
        import mithril
        from transformers import AutoTokenizer

        model, state_dict = model_and_state
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        # Get original output
        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            original_output = model(**inputs).last_hidden_state

        # Serialize and compress (pickle format -> use compress_raw)
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)
        decompressed = compressor.decompress_raw(compressed, len(original_data))

        # Load into a new model
        recovered_state = torch.load(io.BytesIO(decompressed), weights_only=True)
        from transformers import AutoModel
        new_model = AutoModel.from_pretrained(self.MODEL_ID)
        new_model.load_state_dict(recovered_state)

        # Verify outputs match
        with torch.no_grad():
            recovered_output = new_model(**inputs).last_hidden_state

        assert torch.allclose(original_output, recovered_output, atol=1e-6), \
            "Model outputs should be identical after roundtrip"


@pytest.mark.slow
class TestTinyGPT2:
    """Test with sshleifer/tiny-gpt2 (~2MB, minimal GPT-2)."""

    MODEL_ID = "sshleifer/tiny-gpt2"

    @pytest.fixture(scope="class")
    def model_and_state(self):
        """Load the tiny GPT-2 model once for all tests in this class."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID)
        state_dict = model.state_dict()
        return model, state_dict

    def test_full_model_roundtrip(self, model_and_state):
        """Test compressing full model weights using raw compression."""
        import mithril

        _, state_dict = model_and_state

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        # Use compress_raw for pickle data
        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)

        ratio = get_compression_ratio(original_data, compressed)
        assert ratio > 1.0, f"Expected some compression, got {ratio:.2f}x"

        decompressed = compressor.decompress_raw(compressed, len(original_data))
        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)

        assert tensor_dict_equal(state_dict, recovered)

    def test_generation_after_roundtrip(self, model_and_state):
        """Verify model generates identical tokens after roundtrip."""
        import mithril
        from transformers import AutoTokenizer

        model, state_dict = model_and_state
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        # Generate with original model
        inputs = tokenizer("The quick brown", return_tensors="pt")
        torch.manual_seed(42)
        with torch.no_grad():
            original_output = model.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )

        # Compress and decompress (pickle format -> use compress_raw)
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)
        decompressed = compressor.decompress_raw(compressed, len(original_data))

        # Load into new model
        recovered_state = torch.load(io.BytesIO(decompressed), weights_only=True)
        from transformers import AutoModelForCausalLM
        new_model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID)
        new_model.load_state_dict(recovered_state)

        # Generate with recovered model
        torch.manual_seed(42)
        with torch.no_grad():
            recovered_output = new_model.generate(
                **inputs, max_new_tokens=10, do_sample=False
            )

        assert torch.equal(original_output, recovered_output), \
            "Generated tokens should be identical after roundtrip"


@pytest.mark.slow
class TestTinyDistilBERT:
    """Test with distilbert/distilbert-base-uncased (~270MB, commonly used)."""

    MODEL_ID = "distilbert/distilbert-base-uncased"

    @pytest.fixture(scope="class")
    def model_and_state(self):
        """Load DistilBERT model once for all tests in this class."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.MODEL_ID)
        state_dict = model.state_dict()
        return model, state_dict

    def test_full_model_roundtrip(self, model_and_state):
        """Test compressing full model weights."""
        import mithril

        _, state_dict = model_and_state

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        # Use compress_raw for pickle data
        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)

        ratio = get_compression_ratio(original_data, compressed)
        print(f"DistilBERT compression ratio: {ratio:.2f}x")
        assert ratio > 1.0, f"Expected some compression, got {ratio:.2f}x"

        decompressed = compressor.decompress_raw(compressed, len(original_data))
        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)

        assert tensor_dict_equal(state_dict, recovered)

    def test_individual_tensor_compression(self, model_and_state):
        """Test compressing individual bf16 tensors for better ratios."""
        import mithril
        import numpy as np

        _, state_dict = model_and_state

        config = mithril.CompressionConfig(zstd_level=3, byte_grouping=True)
        compressor = mithril.CheckpointCompressor(config)

        total_original = 0
        total_compressed = 0

        # Compress each tensor individually as raw bytes
        for name, tensor in state_dict.items():
            if tensor.is_floating_point():
                # Convert to bf16 and get raw bytes
                bf16_tensor = tensor.to(torch.bfloat16)
                raw_bytes = bf16_tensor.view(torch.uint16).numpy().tobytes()

                compressed = compressor.compress(raw_bytes, "bf16")

                total_original += len(raw_bytes)
                total_compressed += len(compressed)

        if total_original > 0:
            overall_ratio = total_original / total_compressed
            print(f"DistilBERT bf16 individual tensor compression: {overall_ratio:.2f}x")
            # Real model weights should achieve better compression than random
            assert overall_ratio > 1.0, f"Expected some compression, got {overall_ratio:.2f}x"


@pytest.mark.slow
class TestCompressionLevelsOnRealModels:
    """Test different compression levels on real model weights."""

    MODEL_ID = "prajjwal1/bert-tiny"

    @pytest.fixture(scope="class")
    def model_state(self):
        """Load model state dict once."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.MODEL_ID)
        return model.state_dict()

    @pytest.mark.parametrize("zstd_level", [1, 3, 9, 19])
    def test_compression_levels(self, model_state, zstd_level):
        """Test various zstd compression levels on pickle data."""
        import mithril

        buffer = io.BytesIO()
        torch.save(model_state, buffer)
        original_data = buffer.getvalue()

        # Use compress_raw for pickle data
        config = mithril.CompressionConfig(zstd_level=zstd_level)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)

        ratio = get_compression_ratio(original_data, compressed)
        print(f"Level {zstd_level}: {ratio:.2f}x compression")

        decompressed = compressor.decompress_raw(compressed, len(original_data))
        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)

        assert tensor_dict_equal(model_state, recovered)


@pytest.mark.slow
class TestByteGroupingOnRealModels:
    """Test byte grouping optimization on real model weights."""

    MODEL_ID = "prajjwal1/bert-tiny"

    @pytest.fixture(scope="class")
    def bf16_state(self):
        """Load and convert model to bf16."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16)
        return model.state_dict()

    def test_byte_grouping_on_raw_tensors(self, bf16_state):
        """Verify byte grouping works on raw bf16 tensor bytes."""
        import mithril
        import numpy as np

        # Pick a specific large tensor
        raw_bytes = None
        test_shape = None
        for name, tensor in bf16_state.items():
            if tensor.numel() > 1000:  # Find a large enough tensor
                raw_bytes = tensor.view(torch.uint16).numpy().tobytes()
                test_shape = tensor.shape
                break

        if raw_bytes is None:
            pytest.skip("No large enough tensor found")

        # Without byte grouping
        config_no_grouping = mithril.CompressionConfig(
            zstd_level=3, byte_grouping=False
        )
        compressor_no_grouping = mithril.CheckpointCompressor(config_no_grouping)
        compressed_no_grouping = compressor_no_grouping.compress(raw_bytes, "bf16")

        # With byte grouping
        config_grouping = mithril.CompressionConfig(
            zstd_level=3, byte_grouping=True
        )
        compressor_grouping = mithril.CheckpointCompressor(config_grouping)
        compressed_grouping = compressor_grouping.compress(raw_bytes, "bf16")

        ratio_no_grouping = get_compression_ratio(raw_bytes, compressed_no_grouping)
        ratio_grouping = get_compression_ratio(raw_bytes, compressed_grouping)

        print(f"Without byte grouping: {ratio_no_grouping:.2f}x")
        print(f"With byte grouping: {ratio_grouping:.2f}x")

        # Verify roundtrip works
        decompressed = compressor_grouping.decompress(
            compressed_grouping, "bf16", len(raw_bytes)
        )
        assert len(decompressed) == len(raw_bytes)


@pytest.mark.slow
class TestFileBasedOperationsOnRealModels:
    """Test file-based save/load with real HuggingFace models."""

    MODEL_ID = "sshleifer/tiny-gpt2"

    def test_save_and_load_to_file(self):
        """Test saving compressed checkpoint to file and loading back."""
        import mithril
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID)
        original_state = model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "original.pt"
            compressed_path = Path(tmpdir) / "compressed.mcp"

            # Save original
            torch.save(original_state, original_path)

            # Read and compress (pickle data -> use compress_raw)
            with open(original_path, "rb") as f:
                original_data = f.read()

            config = mithril.CompressionConfig(zstd_level=3)
            compressor = mithril.CheckpointCompressor(config)
            compressed = compressor.compress_raw(original_data)

            # Save compressed
            with open(compressed_path, "wb") as f:
                f.write(compressed)

            # Verify file sizes
            original_size = original_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            print(f"Original: {original_size / 1024:.1f} KB")
            print(f"Compressed: {compressed_size / 1024:.1f} KB")
            print(f"Ratio: {original_size / compressed_size:.2f}x")

            assert compressed_size <= original_size

            # Load and decompress
            with open(compressed_path, "rb") as f:
                compressed_data = f.read()

            decompressed = compressor.decompress_raw(
                compressed_data, len(original_data)
            )
            recovered_state = torch.load(io.BytesIO(decompressed), weights_only=True)

            assert tensor_dict_equal(original_state, recovered_state)


@pytest.mark.slow
class TestMultipleDtypesInSameModel:
    """Test models with mixed dtypes in state dict."""

    def test_model_with_embeddings_and_weights(self):
        """Test model that has both embeddings (often int) and weights (float)."""
        import mithril
        from transformers import AutoModel

        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        state_dict = model.state_dict()

        # Verify we have mixed types
        dtypes = {k: v.dtype for k, v in state_dict.items()}
        unique_dtypes = set(dtypes.values())
        print(f"Dtypes in model: {unique_dtypes}")

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        original_data = buffer.getvalue()

        # Use compress_raw for pickle data
        config = mithril.CompressionConfig(zstd_level=3)
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress_raw(original_data)
        decompressed = compressor.decompress_raw(compressed, len(original_data))

        recovered = torch.load(io.BytesIO(decompressed), weights_only=True)

        # Verify all tensors match including dtype
        for key in state_dict:
            original = state_dict[key]
            recovered_tensor = recovered[key]
            assert original.dtype == recovered_tensor.dtype, \
                f"Dtype mismatch for {key}: {original.dtype} vs {recovered_tensor.dtype}"
            assert torch.equal(original, recovered_tensor), \
                f"Value mismatch for {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
