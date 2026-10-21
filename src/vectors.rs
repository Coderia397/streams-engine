use anyhow::{bail, Result};
use half::f16;
use memmap2::MmapOptions;
use rayon::prelude::*;
use crate::det::HashMap;
use std::fs::File;
use std::sync::Arc;
use tracing::info;

pub const DIMS: usize = 1024;

/// In-memory vector store.
/// Vectors kept as f16 (1,024 dimensions, Qwen3-Embedding-8B cut by Matryoshka and renormalised; about 1.3 GB for 626k).
/// Keys (type aware ids) come from the sidecar `<vectors.bin>.keys`.
pub struct VectorStore {
    /// Flat f16 buffer: row i starts at i * DIMS
    pub data:      Vec<f16>,
    /// Item keys (`catalog::item_key`) parallel to data rows (index_of uses a fast HashMap)
    pub ids:       Vec<u32>,
    id_to_idx:     HashMap<u32, usize>,
    pub n_vecs:    usize,
}

impl VectorStore {
    /// Load the vectors: a raw f16 file of `DIMS` values per row, with the sidecar `<vectors.bin>.keys` (u32 LE `catalog::item_key` per row, same order).
    /// The file is copied into memory in one go (the engine only ever uses f16).
    pub fn load(vectors_bin: &str) -> Result<Arc<Self>> {
        let file = File::open(vectors_bin)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let keys_path = format!("{vectors_bin}.keys");
        let ids = le_u32s(&std::fs::read(&keys_path).map_err(|e| anyhow::anyhow!("vector keys {keys_path}: {e}"))?);
        let n = ids.len();
        if n == 0 || mmap.len() != n * DIMS * 2 {
            bail!("{vectors_bin} holds {} bytes, expected {} keys x {DIMS} dimensions x 2 bytes", mmap.len(), n);
        }
        let data = f16_from_le_bytes(&mmap);

        // Build O(1) reverse lookup
        let id_to_idx: HashMap<u32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        info!("Vectors loaded: {} × {} dims (f16, {} MB)", n, DIMS, data.len() * 2 / 1_048_576);

        Ok(Arc::new(Self { data, ids, id_to_idx, n_vecs: n }))
    }

    /// The stored f16 row of vector `i`.
    #[inline]
    pub fn row(&self, i: usize) -> &[f16] {
        &self.data[i * DIMS..(i + 1) * DIMS]
    }

    /// Get the f32 vector for row index `i`.
    #[inline]
    pub fn get_f32(&self, i: usize) -> Vec<f32> {
        self.row(i).iter().map(|x| x.to_f32()).collect()
    }

    /// O(1) lookup: catalog ID → row index.
    #[inline]
    pub fn index_of(&self, catalog_id: u32) -> Option<usize> {
        self.id_to_idx.get(&catalog_id).copied()
    }

    /// Brute-force top-k cosine similarity (rayon parallel).
    /// Used when HNSW index is not yet ready.
    pub fn brute_force_topk(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scores: Vec<(u32, f32)> = (0..self.n_vecs)
            .into_par_iter()
            .map(|i| {
                let sim = cosine_sim_f16(query, &self.data[i * DIMS..(i + 1) * DIMS]);
                (self.ids[i], sim)
            })
            .collect();

        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }
}

/// The f16 values of a little endian byte slice (length a multiple of 2): one copy on little endian machines, element by element otherwise.
fn f16_from_le_bytes(bytes: &[u8]) -> Vec<f16> {
    #[cfg(target_endian = "little")]
    {
        let n = bytes.len() / 2;
        let mut out: Vec<f16> = Vec::with_capacity(n);
        // SAFETY: `out` has room for n f16 (2 bytes each), f16 has no invalid bit patterns, and the regions do not overlap
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, n * 2);
            out.set_len(n);
        }
        out
    }
    #[cfg(not(target_endian = "little"))]
    {
        bytes.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c)).collect()
    }
}

/// Little endian f32 values of a byte slice (length a multiple of 4).
pub fn le_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Little endian u32 values of a byte slice (length a multiple of 4).
pub fn le_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

impl VectorStore {
    /// Mean of all vectors (unnormalised). Used to centre similarity scores.
    pub fn compute_mean(&self) -> Vec<f32> {
        let n = self.n_vecs.max(1) as f32;
        let mut sum = (0..self.n_vecs)
            .into_par_iter()
            .fold(
                || vec![0.0f32; DIMS],
                |mut acc, i| {
                    for (a, x) in acc.iter_mut().zip(&self.data[i * DIMS..(i + 1) * DIMS]) {
                        *a += x.to_f32();
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; DIMS],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        sum.iter_mut().for_each(|x| *x /= n);
        sum
    }

    /// True when row `i` is the all zero placeholder written for a failed embedding.
    pub fn is_zero_row(&self, i: usize) -> bool {
        self.data[i * DIMS..(i + 1) * DIMS].iter().all(|x| x.to_f32() == 0.0)
    }
}

/// Cosine similarity — f32 query vs f16 stored vector (avoids materialising full f32 row).
#[inline]
pub fn cosine_sim_f16(query: &[f32], stored: &[f16]) -> f32 {
    let mut acc = [0.0f32; 16];
    let (cq, rq) = query.as_chunks::<16>();
    let (cs, rs) = stored.as_chunks::<16>();
    let tail: f32 = rq.iter().zip(rs).map(|(q, s)| q * s.to_f32()).sum();
    for (q, s) in cq.iter().zip(cs) {
        for i in 0..16 {
            acc[i] += q[i] * s[i].to_f32();
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Cosine similarity between two f32 slices.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    // sixteen independent partial sums let the compiler use SIMD (one running sum is a serial chain of adds)
    let mut acc = [0.0f32; 16];
    let (ca, ra) = a.as_chunks::<16>();
    let (cb, rb) = b.as_chunks::<16>();
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// L2-normalise a vector in-place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Unit-sphere centroid (Fréchet mean) of a set of already-normalised vectors.
pub fn sphere_mean(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![0.0f32; DIMS];
    }
    let mut sum = vec![0.0f32; DIMS];
    for v in vecs {
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    l2_normalize(&mut sum);
    sum
}
