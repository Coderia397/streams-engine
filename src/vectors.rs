use anyhow::{bail, Result};
use half::f16;
use memmap2::MmapOptions;
use rayon::prelude::*;
use rusqlite::Connection;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use tracing::info;

pub const DIMS: usize = 1024;

/// In-memory vector store.
/// Vectors kept as f16 (1,024 dimensions, Qwen3-Embedding-8B cut by Matryoshka and renormalised; about 1.3 GB for 626k).
/// Keys (type aware ids) come from the sidecar `<vectors.bin>.keys`, see `resolve_ids`.
pub struct VectorStore {
    /// Flat f16 buffer: row i starts at i * DIMS
    pub data:      Vec<f16>,
    /// Item keys (`catalog::item_key`) parallel to data rows (index_of uses a fast HashMap)
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

        let (n, half) = detect_layout(vectors_bin, mmap.len())?;

        // f32 file: convert to f16. f16 file: copy as is (the engine only ever uses f16).
        let data: Vec<f16> = if half {
            f16_from_le_bytes(&mmap)
        } else {
            le_f32s(&mmap).into_iter().map(f16::from_f32).collect()
        };

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

/// Row count and element width of a vectors file. The default is raw f32. When the sidecar
/// `<vectors.bin>.keys` exists and the file holds exactly keys × DIMS × 2 bytes, it is raw f16
/// (half the size to store and download; the engine keeps f16 in memory anyway).
pub fn detect_layout(vectors_bin: &str, file_len: usize) -> Result<(usize, bool)> {
    if let Ok(m) = std::fs::metadata(format!("{vectors_bin}.keys")) {
        let k = (m.len() / 4) as usize;
        if k > 0 && file_len == k * DIMS * 2 {
            return Ok((k, true));
        }
    }
    if !file_len.is_multiple_of(DIMS * 4) {
        bail!("vectors file size {file_len} is neither f32 ({DIMS}×4 per row) nor f16 matching the .keys sidecar");
    }
    Ok((file_len / (DIMS * 4), false))
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

/// Row-order keys for vectors.bin (`catalog::item_key`, unique across movies and TV).
/// Order of preference:
///  1. sidecar `<vectors.bin>.keys` (u32 LE keys), written by the embedding pipeline or by step 2
///  2. legacy sidecar `<vectors.bin>.ids` (u32 LE raw TMDB ids from before the movie/TV collision fix),
///     converted through catalog.db and frozen into `.keys`
///  3. derived from catalog.db (popularity DESC, the embedding pipeline order), frozen into `.keys`
pub fn resolve_ids(vectors_bin: &str, catalog_db: &str, n: usize) -> Result<Vec<u32>> {
    use crate::catalog::item_key;
    let keys_path = format!("{vectors_bin}.keys");
    let legacy_path = format!("{vectors_bin}.ids");

    if let Ok(bytes) = std::fs::read(&keys_path) {
        if bytes.len() == n * 4 {
            info!("Vector keys read from {keys_path}");
            return Ok(le_u32s(&bytes));
        }
        tracing::warn!("{keys_path} has {} keys, vectors.bin has {n} — ignoring it", bytes.len() / 4);
    }

    let conn = Connection::open(catalog_db)?;
    let has_tmdb_id = conn
        .prepare("PRAGMA table_info(catalog)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .any(|c| c == "tmdb_id");
    let id_col = if has_tmdb_id { "tmdb_id" } else { "id" };

    let keys: Vec<u32> = if let Some(raw) = std::fs::read(&legacy_path).ok().filter(|b| b.len() == n * 4) {
        // legacy raw ids: every raw id was unique in the old catalog, so the media type is found by lookup
        let mut stmt = conn.prepare(&format!("SELECT {id_col}, media_type FROM catalog"))?;
        let by_raw: HashMap<u32, u32> = stmt
            .query_map([], |r| {
                let id = r.get::<_, i64>(0)? as u32;
                let mt: String = r.get(1).unwrap_or_default();
                Ok((id, item_key(&mt, id)))
            })?
            .filter_map(|r| r.ok())
            .collect();
        let mut missing = 0usize;
        let keys: Vec<u32> = le_u32s(&raw)
            .into_iter()
            .map(|id| *by_raw.get(&id).unwrap_or_else(|| { missing += 1; &id }))
            .collect();
        if missing > 0 {
            tracing::warn!("{missing} legacy vector ids not found in catalog.db, kept as movie keys");
        }
        info!("Legacy vector ids converted to type aware keys ({legacy_path})");
        keys
    } else {
        let mut stmt = conn.prepare(&format!(
            "SELECT {id_col}, media_type FROM catalog ORDER BY popularity DESC LIMIT ?1"
        ))?;
        let keys: Vec<u32> = stmt
            .query_map([n as i64], |row| {
                let id = row.get::<_, i64>(0)? as u32;
                let mt: String = row.get(1).unwrap_or_default();
                Ok(item_key(&mt, id))
            })?
            .filter_map(|r| r.ok())
            .collect();
        if keys.len() != n {
            bail!(
                "Expected {} IDs from catalog but got {} (catalog may have fewer items than vectors)",
                n, keys.len()
            );
        }
        keys
    };

    let mut bytes = Vec::with_capacity(n * 4);
    for k in &keys {
        bytes.extend_from_slice(&k.to_le_bytes());
    }
    let tmp = format!("{keys_path}.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &keys_path)?;
    info!("Vector keys frozen to {keys_path}");
    Ok(keys)
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

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn detects_f16_f32_and_rejects_bad_sizes() {
        let dir = std::env::temp_dir().join(format!("streams_layout_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("v.bin").to_string_lossy().to_string();

        // no sidecar: f32
        assert_eq!(detect_layout(&bin, 2 * DIMS * 4).unwrap(), (2, false));
        assert!(detect_layout(&bin, 2 * DIMS * 4 + 2).is_err());

        // sidecar with 3 keys: a file of 3 × DIMS × 2 bytes is f16, 3 × DIMS × 4 is still f32
        std::fs::write(format!("{bin}.keys"), vec![0u8; 3 * 4]).unwrap();
        assert_eq!(detect_layout(&bin, 3 * DIMS * 2).unwrap(), (3, true));
        assert_eq!(detect_layout(&bin, 3 * DIMS * 4).unwrap(), (3, false));
        // f16 size that does not match the key count is rejected, not guessed
        assert!(detect_layout(&bin, 5 * DIMS * 2).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
