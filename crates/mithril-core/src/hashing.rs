//! Hashing functions.

/// Trait for hash functions.
pub trait HashFunction: Send + Sync {
    /// Hash data and return raw bytes.
    fn hash(&self, data: &[u8]) -> Vec<u8>;

    /// Hash data and return hex string.
    fn hash_hex(&self, data: &[u8]) -> String {
        hex::encode(self.hash(data))
    }

    /// Hash data and return u64 (for use in hash maps).
    fn hash_u64(&self, data: &[u8]) -> u64;
}

/// XXHash3 hasher - extremely fast, good for checksums.
pub struct XxHash3;

impl XxHash3 {
    /// Create a new XXHash3 hasher.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for XxHash3 {
    fn default() -> Self {
        Self::new()
    }
}

impl HashFunction for XxHash3 {
    fn hash(&self, data: &[u8]) -> Vec<u8> {
        xxhash_rust::xxh3::xxh3_128(data).to_le_bytes().to_vec()
    }

    fn hash_u64(&self, data: &[u8]) -> u64 {
        xxhash_rust::xxh3::xxh3_64(data)
    }
}

/// Blake3 hasher - cryptographically secure, still fast.
pub struct Blake3Hasher;

impl Blake3Hasher {
    /// Create a new Blake3 hasher.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for Blake3Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl HashFunction for Blake3Hasher {
    fn hash(&self, data: &[u8]) -> Vec<u8> {
        blake3::hash(data).as_bytes().to_vec()
    }

    fn hash_u64(&self, data: &[u8]) -> u64 {
        let hash = blake3::hash(data);
        let bytes = hash.as_bytes();
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

/// Hash with seed for MinHash-style algorithms.
#[inline]
pub fn hash_with_seed(data: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(data, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxhash3_deterministic() {
        let hasher = XxHash3::new();
        let data = b"hello world";

        let h1 = hasher.hash(data);
        let h2 = hasher.hash(data);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_blake3_deterministic() {
        let hasher = Blake3Hasher::new();
        let data = b"hello world";

        let h1 = hasher.hash(data);
        let h2 = hasher.hash(data);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_with_seed() {
        let data = b"hello";
        let h1 = hash_with_seed(data, 42);
        let h2 = hash_with_seed(data, 42);
        let h3 = hash_with_seed(data, 43);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
