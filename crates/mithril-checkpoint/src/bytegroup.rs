//! Byte grouping for improved compression of floating-point data.
//!
//! bfloat16 and float16 values have their bytes interleaved: [h0,l0,h1,l1,...]
//! where h=high byte (exponent), l=low byte (mantissa).
//!
//! Grouping separates these: [h0,h1,...,l0,l1,...] so similar bytes are
//! adjacent, improving compression ratios by ~20% or more.
//!
//! ## Parallel Variants
//!
//! For large data (>1MB), use the `_par` suffixed functions which leverage
//! rayon for parallel processing to achieve higher throughput.

use rayon::prelude::*;

/// Threshold above which parallel processing is beneficial.
/// Below this, sequential is faster due to thread overhead.
pub const PARALLEL_THRESHOLD: usize = 1024 * 1024; // 1MB

/// Group bf16/fp16 bytes: [h0,l0,h1,l1,...] -> [h0,h1,...,l0,l1,...]
///
/// # Panics
/// Panics if data length is not even (must be pairs of bytes).
///
/// # Example
/// ```
/// use mithril_checkpoint::bytegroup::byte_group_bf16;
///
/// let data = vec![0xAA, 0x11, 0xBB, 0x22, 0xCC, 0x33];
/// let grouped = byte_group_bf16(&data);
/// assert_eq!(grouped, vec![0xAA, 0xBB, 0xCC, 0x11, 0x22, 0x33]);
/// ```
#[inline]
pub fn byte_group_bf16(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() % 2 == 0,
        "Data length must be even for bf16 grouping"
    );

    let n = data.len() / 2;
    let mut grouped = Vec::with_capacity(data.len());

    // High bytes first (index 0, 2, 4, ...)
    for i in 0..n {
        grouped.push(data[i * 2]);
    }
    // Low bytes second (index 1, 3, 5, ...)
    for i in 0..n {
        grouped.push(data[i * 2 + 1]);
    }

    grouped
}

/// Ungroup bf16/fp16 bytes: [h0,h1,...,l0,l1,...] -> [h0,l0,h1,l1,...]
///
/// Inverse of `byte_group_bf16`.
///
/// # Panics
/// Panics if data length is not even.
///
/// # Example
/// ```
/// use mithril_checkpoint::bytegroup::byte_ungroup_bf16;
///
/// let grouped = vec![0xAA, 0xBB, 0xCC, 0x11, 0x22, 0x33];
/// let ungrouped = byte_ungroup_bf16(&grouped);
/// assert_eq!(ungrouped, vec![0xAA, 0x11, 0xBB, 0x22, 0xCC, 0x33]);
/// ```
#[inline]
pub fn byte_ungroup_bf16(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() % 2 == 0,
        "Data length must be even for bf16 ungrouping"
    );

    let n = data.len() / 2;
    let mut ungrouped = Vec::with_capacity(data.len());

    // Interleave high and low bytes
    for i in 0..n {
        ungrouped.push(data[i]); // high byte
        ungrouped.push(data[n + i]); // low byte
    }

    ungrouped
}

/// Group fp32 bytes: [b0,b1,b2,b3,b4,b5,b6,b7,...] -> [b0,b4,...,b1,b5,...,b2,b6,...,b3,b7,...]
///
/// Groups 4-byte floats by byte position for better compression.
///
/// # Panics
/// Panics if data length is not a multiple of 4.
#[inline]
pub fn byte_group_fp32(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() % 4 == 0,
        "Data length must be multiple of 4 for fp32 grouping"
    );

    let n = data.len() / 4;
    let mut grouped = Vec::with_capacity(data.len());

    // Group by byte position
    for byte_pos in 0..4 {
        for i in 0..n {
            grouped.push(data[i * 4 + byte_pos]);
        }
    }

    grouped
}

/// Ungroup fp32 bytes: inverse of `byte_group_fp32`.
///
/// # Panics
/// Panics if data length is not a multiple of 4.
#[inline]
pub fn byte_ungroup_fp32(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() % 4 == 0,
        "Data length must be multiple of 4 for fp32 ungrouping"
    );

    let n = data.len() / 4;
    let mut ungrouped = Vec::with_capacity(data.len());

    for i in 0..n {
        for byte_pos in 0..4 {
            ungrouped.push(data[byte_pos * n + i]);
        }
    }

    ungrouped
}

// ============================================================================
// Parallel variants using rayon
// ============================================================================

/// Parallel byte grouping for bf16/fp16 data.
///
/// Uses rayon for parallel processing on large data. Falls back to sequential
/// for data smaller than [`PARALLEL_THRESHOLD`].
///
/// # Panics
/// Panics if data length is not even.
pub fn byte_group_bf16_par(data: &[u8]) -> Vec<u8> {
    if data.len() < PARALLEL_THRESHOLD {
        return byte_group_bf16(data);
    }

    assert!(
        data.len() % 2 == 0,
        "Data length must be even for bf16 grouping"
    );

    let n = data.len() / 2;
    let mut grouped = vec![0u8; data.len()];

    // Split into high and low byte sections
    let (high, low) = grouped.split_at_mut(n);

    // Fill high bytes in parallel
    high.par_iter_mut()
        .enumerate()
        .for_each(|(i, byte)| *byte = data[i * 2]);

    // Fill low bytes in parallel
    low.par_iter_mut()
        .enumerate()
        .for_each(|(i, byte)| *byte = data[i * 2 + 1]);

    grouped
}

/// Parallel byte ungrouping for bf16/fp16 data.
///
/// Uses rayon for parallel processing on large data. Falls back to sequential
/// for data smaller than [`PARALLEL_THRESHOLD`].
///
/// # Panics
/// Panics if data length is not even.
pub fn byte_ungroup_bf16_par(data: &[u8]) -> Vec<u8> {
    if data.len() < PARALLEL_THRESHOLD {
        return byte_ungroup_bf16(data);
    }

    assert!(
        data.len() % 2 == 0,
        "Data length must be even for bf16 ungrouping"
    );

    let n = data.len() / 2;
    let mut ungrouped = vec![0u8; data.len()];

    // Each pair of output bytes can be filled independently
    ungrouped
        .par_chunks_mut(2)
        .enumerate()
        .for_each(|(i, chunk)| {
            chunk[0] = data[i]; // high byte from first half
            chunk[1] = data[n + i]; // low byte from second half
        });

    ungrouped
}

/// Parallel byte grouping for fp32 data.
///
/// Uses rayon for parallel processing on large data. Falls back to sequential
/// for data smaller than [`PARALLEL_THRESHOLD`].
///
/// # Panics
/// Panics if data length is not a multiple of 4.
pub fn byte_group_fp32_par(data: &[u8]) -> Vec<u8> {
    if data.len() < PARALLEL_THRESHOLD {
        return byte_group_fp32(data);
    }

    assert!(
        data.len() % 4 == 0,
        "Data length must be multiple of 4 for fp32 grouping"
    );

    let n = data.len() / 4;
    let mut grouped = vec![0u8; data.len()];

    // Split output into 4 sections, one per byte position
    let (section0, rest) = grouped.split_at_mut(n);
    let (section1, rest) = rest.split_at_mut(n);
    let (section2, section3) = rest.split_at_mut(n);

    // Fill each section in parallel
    rayon::scope(|s| {
        s.spawn(|_| {
            section0
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, byte)| *byte = data[i * 4]);
        });
        s.spawn(|_| {
            section1
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, byte)| *byte = data[i * 4 + 1]);
        });
        s.spawn(|_| {
            section2
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, byte)| *byte = data[i * 4 + 2]);
        });
        s.spawn(|_| {
            section3
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, byte)| *byte = data[i * 4 + 3]);
        });
    });

    grouped
}

/// Parallel byte ungrouping for fp32 data.
///
/// Uses rayon for parallel processing on large data. Falls back to sequential
/// for data smaller than [`PARALLEL_THRESHOLD`].
///
/// # Panics
/// Panics if data length is not a multiple of 4.
pub fn byte_ungroup_fp32_par(data: &[u8]) -> Vec<u8> {
    if data.len() < PARALLEL_THRESHOLD {
        return byte_ungroup_fp32(data);
    }

    assert!(
        data.len() % 4 == 0,
        "Data length must be multiple of 4 for fp32 ungrouping"
    );

    let n = data.len() / 4;
    let mut ungrouped = vec![0u8; data.len()];

    // Each 4-byte group can be filled independently
    ungrouped
        .par_chunks_mut(4)
        .enumerate()
        .for_each(|(i, chunk)| {
            chunk[0] = data[i]; // byte 0 from section 0
            chunk[1] = data[n + i]; // byte 1 from section 1
            chunk[2] = data[2 * n + i]; // byte 2 from section 2
            chunk[3] = data[3 * n + i]; // byte 3 from section 3
        });

    ungrouped
}

/// Auto-selecting byte grouping for bf16/fp16 that picks parallel or sequential.
///
/// Automatically chooses the optimal implementation based on data size.
#[inline]
pub fn byte_group_bf16_auto(data: &[u8]) -> Vec<u8> {
    if data.len() >= PARALLEL_THRESHOLD {
        byte_group_bf16_par(data)
    } else {
        byte_group_bf16(data)
    }
}

/// Auto-selecting byte ungrouping for bf16/fp16 that picks parallel or sequential.
#[inline]
pub fn byte_ungroup_bf16_auto(data: &[u8]) -> Vec<u8> {
    if data.len() >= PARALLEL_THRESHOLD {
        byte_ungroup_bf16_par(data)
    } else {
        byte_ungroup_bf16(data)
    }
}

/// Auto-selecting byte grouping for fp32 that picks parallel or sequential.
#[inline]
pub fn byte_group_fp32_auto(data: &[u8]) -> Vec<u8> {
    if data.len() >= PARALLEL_THRESHOLD {
        byte_group_fp32_par(data)
    } else {
        byte_group_fp32(data)
    }
}

/// Auto-selecting byte ungrouping for fp32 that picks parallel or sequential.
#[inline]
pub fn byte_ungroup_fp32_auto(data: &[u8]) -> Vec<u8> {
    if data.len() >= PARALLEL_THRESHOLD {
        byte_ungroup_fp32_par(data)
    } else {
        byte_ungroup_fp32(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_group_bf16_basic() {
        let data = vec![0xAA, 0x11, 0xBB, 0x22, 0xCC, 0x33];
        let grouped = byte_group_bf16(&data);
        assert_eq!(grouped, vec![0xAA, 0xBB, 0xCC, 0x11, 0x22, 0x33]);
    }

    #[test]
    fn test_byte_ungroup_bf16_basic() {
        let grouped = vec![0xAA, 0xBB, 0xCC, 0x11, 0x22, 0x33];
        let ungrouped = byte_ungroup_bf16(&grouped);
        assert_eq!(ungrouped, vec![0xAA, 0x11, 0xBB, 0x22, 0xCC, 0x33]);
    }

    #[test]
    fn test_bf16_roundtrip() {
        let original: Vec<u8> = (0..256u16).flat_map(|x| x.to_le_bytes()).collect();
        let grouped = byte_group_bf16(&original);
        let ungrouped = byte_ungroup_bf16(&grouped);
        assert_eq!(original, ungrouped);
    }

    #[test]
    fn test_bf16_roundtrip_large() {
        // Simulate typical bf16 tensor data
        let original: Vec<u8> = (0..10000u16)
            .map(|i| {
                // Simulate bf16 values with varying exponents
                let exp = ((i % 32) as u16) << 7; // exponent bits
                let mant = i.wrapping_mul(7) & 0x7F; // mantissa bits
                exp | mant
            })
            .flat_map(|x| x.to_le_bytes())
            .collect();

        let grouped = byte_group_bf16(&original);
        let ungrouped = byte_ungroup_bf16(&grouped);
        assert_eq!(original, ungrouped);
    }

    #[test]
    fn test_bf16_empty() {
        let empty: Vec<u8> = vec![];
        assert_eq!(byte_group_bf16(&empty), empty);
        assert_eq!(byte_ungroup_bf16(&empty), empty);
    }

    #[test]
    #[should_panic(expected = "Data length must be even")]
    fn test_bf16_odd_length_panics() {
        byte_group_bf16(&[0x00, 0x01, 0x02]);
    }

    #[test]
    fn test_byte_group_fp32_basic() {
        // 2 floats: [a0,a1,a2,a3,b0,b1,b2,b3] -> [a0,b0,a1,b1,a2,b2,a3,b3]
        let data = vec![0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14];
        let grouped = byte_group_fp32(&data);
        assert_eq!(
            grouped,
            vec![0x01, 0x11, 0x02, 0x12, 0x03, 0x13, 0x04, 0x14]
        );
    }

    #[test]
    fn test_fp32_roundtrip() {
        let original: Vec<u8> = (0..1000u32).flat_map(|x| x.to_le_bytes()).collect();
        let grouped = byte_group_fp32(&original);
        let ungrouped = byte_ungroup_fp32(&grouped);
        assert_eq!(original, ungrouped);
    }

    #[test]
    fn test_grouping_improves_compression() {
        // Create data that simulates real bf16 weights:
        // - Exponents clustered around bias (127)
        // - Most values are small (exponent 120-130)
        // - Mantissa varies more randomly
        let mut rng_state = 12345u64;
        let data: Vec<u8> = (0..10000)
            .flat_map(|_| {
                // Simple LCG for reproducibility
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = (rng_state >> 33) as u16;

                // bf16: 1 sign, 8 exp, 7 mantissa
                // Exponents cluster around 127 (bias) for small weight values
                let exp = 124 + (rand_val % 8) as u8; // 124-131, clustered exponents
                let sign = ((rand_val >> 8) & 1) as u8;
                let mantissa = (rand_val & 0x7F) as u8; // 7-bit mantissa
                let bf16_val = ((sign as u16) << 15) | ((exp as u16) << 7) | (mantissa as u16);
                bf16_val.to_le_bytes()
            })
            .collect();

        // Compress original
        let orig_compressed = zstd::encode_all(&data[..], 3).unwrap();

        // Compress grouped
        let grouped = byte_group_bf16(&data);
        let grouped_compressed = zstd::encode_all(&grouped[..], 3).unwrap();

        // Grouped should compress better for realistic weight data
        // (high bytes become very similar when exponents cluster)
        assert!(
            grouped_compressed.len() <= orig_compressed.len(),
            "Grouped ({} bytes) should compress at least as well as ungrouped ({} bytes)",
            grouped_compressed.len(),
            orig_compressed.len()
        );
    }

    // ========================================================================
    // Parallel variant tests
    // ========================================================================

    #[test]
    fn test_bf16_par_roundtrip_large() {
        // Create data larger than PARALLEL_THRESHOLD to trigger parallel path
        let size = PARALLEL_THRESHOLD + 10000;
        let original: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let grouped = byte_group_bf16_par(&original);
        let ungrouped = byte_ungroup_bf16_par(&grouped);

        assert_eq!(original, ungrouped);
    }

    #[test]
    fn test_fp32_par_roundtrip_large() {
        // Create data larger than PARALLEL_THRESHOLD (must be multiple of 4)
        let size = ((PARALLEL_THRESHOLD + 10000) / 4) * 4;
        let original: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let grouped = byte_group_fp32_par(&original);
        let ungrouped = byte_ungroup_fp32_par(&grouped);

        assert_eq!(original, ungrouped);
    }

    #[test]
    fn test_par_matches_sequential_bf16() {
        // Ensure parallel and sequential produce identical results
        let size = PARALLEL_THRESHOLD + 10000;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let seq_grouped = byte_group_bf16(&data);
        let par_grouped = byte_group_bf16_par(&data);
        assert_eq!(seq_grouped, par_grouped);

        let seq_ungrouped = byte_ungroup_bf16(&seq_grouped);
        let par_ungrouped = byte_ungroup_bf16_par(&par_grouped);
        assert_eq!(seq_ungrouped, par_ungrouped);
    }

    #[test]
    fn test_par_matches_sequential_fp32() {
        // Ensure parallel and sequential produce identical results
        let size = ((PARALLEL_THRESHOLD + 10000) / 4) * 4;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let seq_grouped = byte_group_fp32(&data);
        let par_grouped = byte_group_fp32_par(&data);
        assert_eq!(seq_grouped, par_grouped);

        let seq_ungrouped = byte_ungroup_fp32(&seq_grouped);
        let par_ungrouped = byte_ungroup_fp32_par(&par_grouped);
        assert_eq!(seq_ungrouped, par_ungrouped);
    }

    #[test]
    fn test_auto_uses_sequential_for_small() {
        // Small data should use sequential (same result as explicit sequential)
        let small_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let auto_result = byte_group_bf16_auto(&small_data);
        let seq_result = byte_group_bf16(&small_data);
        assert_eq!(auto_result, seq_result);
    }

    #[test]
    fn test_auto_uses_parallel_for_large() {
        // Large data should use parallel (same result as explicit parallel)
        let large_data: Vec<u8> = (0..PARALLEL_THRESHOLD + 10000)
            .map(|i| (i % 256) as u8)
            .collect();

        let auto_result = byte_group_bf16_auto(&large_data);
        let par_result = byte_group_bf16_par(&large_data);
        assert_eq!(auto_result, par_result);
    }
}
