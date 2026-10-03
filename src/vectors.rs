use anyhow::{bail, Result};
use half::f16;
use memmap2::MmapOptions;
use rayon::prelude::*;
use rusqlite::Connection;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use tracing::info;

pub const DIMS: usize = 2560;

/// In-memory vector store.
/// Vectors kept as f16 (~512 MB for 100k × 2560).
/// IDs derived from catalog.db (top-N by popularity DESC) — same order as embedding pipeline.
pub struct VectorStore {
    /// Flat f16 buffer: row i starts at i * DIMS
    pub data:      Vec<f16>,
    /// TMDB IDs parallel to data rows (index_of uses a fast HashMap)
    pub ids:       Vec<u32>,
    id_to_idx:     HashMap<u32, usize>,
    pub n_vecs:    usize,
}

impl VectorStore {
    /// Load vectors directly from the raw f32 binary file.
    /// IDs are derived from catalog.db (ORDER BY popularity DESC LIMIT n_vecs)
    /// — this matches the original embedding pipeline ordering.
    ///
    /// No separate IDs file needed.
    pub fn load(vectors_bin: &str, catalog_db: &str) -> Result<Arc<Self>> {
        // --- mmap f32 vectors ---
        let file = File::open(vectors_bin)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() % (DIMS * 4) != 0 {
            bail!(
                "vectors.bin size {} not divisible by DIMS×4 ({}×4)",
                mmap.len(), DIMS
            );
        }
        let n = mmap.len() / (DIMS * 4);

        // Convert f32 → f16 in-place
        let data: Vec<f16> = mmap
            .chunks_exact(4)
            .map(|c| f16::from_f32(f32::from_le_bytes(c.try_into().unwrap())))
            .collect();

        let ids = resolve_ids(vectors_bin, catalog_db, n)?;

        // Build O(1) reverse lookup
        let id_to_idx: HashMap<u32, usize> = ids.iter().enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        info!(
            "Vectors loaded: {} × {} dims (f16, {} MB)",
            n, DIMS, data.len() * 2 / 1_048_576
        );

        Ok(Arc::new(Self { data, ids, id_to_idx, n_vecs: n }))
    }

    /// Get the f32 vector for row index `i`.
    #[inline]
    pub fn get_f32(&self, i: usize) -> Vec<f32> {
        self.data[i * DIMS..(i + 1) * DIMS]
            .iter()
            .map(|x| x.to_f32())
            .collect()
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

/// Row-order IDs for vectors.bin. Reads the sidecar `<vectors.bin>.ids` (u32 LE)
/// when present. Otherwise derives the order from catalog.db (popularity DESC,
/// the embedding pipeline order) and freezes it into the sidecar, so later
/// catalog changes cannot shift which ID a vector belongs to.
pub fn resolve_ids(vectors_bin: &str, catalog_db: &str, n: usize) -> Result<Vec<u32>> {
    let sidecar = format!("{vectors_bin}.ids");
    if let Ok(bytes) = std::fs::read(&sidecar) {
        if bytes.len() == n * 4 {
            info!("Vector IDs read from {sidecar}");
            return Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect());
        }
        tracing::warn!("{sidecar} has {} ids, vectors.bin has {n} — ignoring sidecar", bytes.len() / 4);
    }

    let conn = Connection::open(catalog_db)?;
    let mut stmt = conn.prepare("SELECT id FROM catalog ORDER BY popularity DESC LIMIT ?1")?;
    let ids: Vec<u32> = stmt
        .query_map([n as i64], |row| row.get::<_, i64>(0).map(|id| id as u32))?
        .filter_map(|r| r.ok())
        .collect();
    if ids.len() != n {
        bail!(
            "Expected {} IDs from catalog but got {} (catalog may have fewer items than vectors)",
            n, ids.len()
        );
    }

    let mut bytes = Vec::with_capacity(n * 4);
    for id in &ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    let tmp = format!("{sidecar}.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &sidecar)?;
    info!("Vector IDs derived from catalog.db and frozen to {sidecar}");
    Ok(ids)
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
    query.iter().zip(stored.iter()).map(|(q, s)| q * s.to_f32()).sum()
}

/// Cosine similarity between two f32 slices.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalise a vector in-place.
pub fn l2_normalize(v: &mut Vec<f32>) {
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
