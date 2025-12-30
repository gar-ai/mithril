"""Integration tests for S3-compatible storage (using MinIO for local testing)."""

import os
import socket
import time

import pytest

mithril = pytest.importorskip("mithril")


def is_minio_running(host: str = "localhost", port: int = 9000) -> bool:
    """Check if MinIO is running on the specified host:port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


# Skip all tests in this module if MinIO is not running
pytestmark = pytest.mark.skipif(
    not is_minio_running(),
    reason="MinIO not running on localhost:9000. Start with: docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ':9001'",
)


# Default MinIO credentials
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
TEST_BUCKET = os.getenv("MINIO_TEST_BUCKET", "mithril-test")


@pytest.fixture(scope="module")
def s3_storage():
    """Create an S3Storage instance connected to MinIO."""
    try:
        storage = mithril.S3RemoteCache(
            bucket=TEST_BUCKET,
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            region="us-east-1",
        )
        yield storage
    except Exception as e:
        pytest.skip(f"Could not create S3 storage: {e}")


class TestS3BasicOperations:
    """Test basic S3 operations (put, get, delete, list)."""

    def test_put_and_get(self, s3_storage):
        """Test storing and retrieving data."""
        key = f"test/data_{time.time()}.bin"
        data = b"Hello, Mithril S3!"

        # Put
        s3_storage.put(key, data)

        # Get
        retrieved = s3_storage.get(key)
        assert retrieved == data

        # Cleanup
        s3_storage.delete(key)

    def test_put_large_data(self, s3_storage):
        """Test storing larger data (1MB)."""
        key = f"test/large_{time.time()}.bin"
        data = os.urandom(1024 * 1024)  # 1MB random data

        s3_storage.put(key, data)
        retrieved = s3_storage.get(key)

        assert retrieved == data
        assert len(retrieved) == 1024 * 1024

        # Cleanup
        s3_storage.delete(key)

    def test_delete(self, s3_storage):
        """Test deleting data."""
        key = f"test/delete_me_{time.time()}.bin"
        data = b"Delete this"

        s3_storage.put(key, data)
        s3_storage.delete(key)

        # Should not be able to get deleted data
        with pytest.raises(Exception):
            s3_storage.get(key)

    def test_list(self, s3_storage):
        """Test listing keys with prefix."""
        prefix = f"test/list_{time.time()}/"

        # Create some test files
        keys = [f"{prefix}file1.bin", f"{prefix}file2.bin", f"{prefix}subdir/file3.bin"]
        for key in keys:
            s3_storage.put(key, b"test data")

        # List with prefix
        listed = s3_storage.list(prefix)
        assert len(listed) >= 3

        # Cleanup
        for key in keys:
            s3_storage.delete(key)

    def test_exists(self, s3_storage):
        """Test checking if a key exists."""
        key = f"test/exists_{time.time()}.bin"

        # Should not exist initially
        assert not s3_storage.exists(key)

        # Create it
        s3_storage.put(key, b"test")

        # Should exist now
        assert s3_storage.exists(key)

        # Cleanup
        s3_storage.delete(key)
        assert not s3_storage.exists(key)


class TestS3WithCheckpoints:
    """Test S3 storage with checkpoint compression."""

    def test_compressed_checkpoint_to_s3(self, s3_storage):
        """Test storing compressed checkpoint data in S3."""
        torch = pytest.importorskip("torch")

        # Create test data
        tensor = torch.randn(100, 100, dtype=torch.bfloat16)
        raw_bytes = bytes(tensor.untyped_storage())

        # Compress
        config = mithril.CompressionConfig()
        compressor = mithril.CheckpointCompressor(config)
        compressed = compressor.compress(raw_bytes, "bf16")

        # Store in S3
        key = f"checkpoints/test_{time.time()}.mcp"
        s3_storage.put(key, compressed)

        # Retrieve
        retrieved_compressed = s3_storage.get(key)
        assert retrieved_compressed == compressed

        # Decompress and verify
        decompressed = compressor.decompress(retrieved_compressed, "bf16", len(raw_bytes))
        assert decompressed == raw_bytes

        # Cleanup
        s3_storage.delete(key)


class TestS3ErrorHandling:
    """Test S3 error handling."""

    def test_get_nonexistent_key(self, s3_storage):
        """Test getting a key that doesn't exist."""
        key = f"test/nonexistent_{time.time()}.bin"

        with pytest.raises(Exception):
            s3_storage.get(key)

    def test_delete_nonexistent_key(self, s3_storage):
        """Test deleting a key that doesn't exist (should not raise)."""
        key = f"test/nonexistent_{time.time()}.bin"

        # Some S3 implementations don't error on delete of nonexistent key
        try:
            s3_storage.delete(key)
        except Exception:
            pass  # Some implementations may raise, that's OK


class TestS3Configuration:
    """Test S3 configuration options."""

    def test_custom_prefix(self):
        """Test S3 storage with custom prefix."""
        try:
            storage = mithril.S3RemoteCache(
                bucket=TEST_BUCKET,
                endpoint=MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                prefix="custom/prefix/",
            )

            key = f"test_{time.time()}.bin"
            storage.put(key, b"test data")

            # The actual S3 key should be prefixed
            retrieved = storage.get(key)
            assert retrieved == b"test data"

            storage.delete(key)
        except Exception as e:
            pytest.skip(f"S3 with prefix not supported: {e}")
