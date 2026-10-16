use crate::vectors::{le_f32s, le_u32s, DIMS};
use anyhow::{bail, Result};
use std::sync::Arc;
use tracing::info;

/// A single pre-discovered cluster seed from offline spherical k-means.
#[derive(Debug, Clone)]
pub struct ClusterSeed {
    /// L2-normalised centroid vector (2560d)
    pub centroid: Vec<f32>,
    /// TMDB IDs of catalog items in this cluster
    pub member_ids: Vec<u32>,
    /// Row title written for the group (from `<seeds>.names`, empty when absent)
    pub name: String,
    /// World level 2 group this seed is (-1 when unknown), from `<seeds>.ids`
    pub group: i32,
}

/// All cluster seeds loaded from `cluster_seeds.bin`.
///
/// Binary format (produced by `tools/compute_cluster_seeds.py`):
/// - 4 bytes: n_clusters (u32 LE)
/// - 4 bytes: n_dims     (u32 LE, must be the vector dimension)
///
/// For each cluster:
///   - n_dims × 4 bytes: centroid f32 LE
///   - 4 bytes: n_members (u32 LE)
///   - n_members × 4 bytes: member IDs (u32 LE)
pub struct ClusterSeeds {
    pub seeds: Vec<ClusterSeed>,
}

impl ClusterSeeds {

    /// Return an empty seed set (engine falls back to brute-force feed).
    pub fn empty() -> Arc<Self> {
        Arc::new(Self { seeds: vec![] })
    }

    /// Load the cluster seeds file (member keys from `catalog::item_key`); an absent file gives an empty set (the feed falls back to brute force).
    pub fn load(path: &str) -> Result<Arc<Self>> {
        if !std::path::Path::new(path).exists() {
            tracing::warn!("cluster seeds not found at {path} — feed will use brute-force fallback");
            return Ok(Self::empty());
        }

        let bytes = std::fs::read(path)?;
        let mut pos = 0usize;

        macro_rules! read_u32 {
            () => {{
                let v = u32::from_le_bytes(bytes[pos..pos + 4].try_into()?);
                pos += 4;
                v
            }};
        }

        let n_clusters = read_u32!() as usize;
        let n_dims = read_u32!() as usize;
        if n_dims != DIMS {
            bail!("cluster_seeds.bin dims={n_dims}, expected {DIMS}");
        }

        let mut seeds = Vec::with_capacity(n_clusters);
        for _ in 0..n_clusters {
            let centroid: Vec<f32> = le_f32s(&bytes[pos..pos + n_dims * 4]);
            pos += n_dims * 4;

            let n_members = read_u32!() as usize;
            let member_ids: Vec<u32> = le_u32s(&bytes[pos..pos + n_members * 4]);
            pos += n_members * 4;

            seeds.push(ClusterSeed { centroid, member_ids, name: String::new(), group: -1 });
        }

        // optional sidecar with one name per seed, in the order of the file (written by ai-engine/store/export_engine_files.py)
        if let Ok(text) = std::fs::read_to_string(format!("{path}.names")) {
            if let Ok(names) = serde_json::from_str::<Vec<String>>(&text) {
                if names.len() == seeds.len() {
                    for (s, n) in seeds.iter_mut().zip(names) { s.name = n; }
                    info!("Seed names loaded");
                } else {
                    tracing::warn!("{path}.names has {} names for {} seeds — ignored", names.len(), seeds.len());
                }
            }
        }
        if let Ok(text) = std::fs::read_to_string(format!("{path}.ids")) {
            if let Ok(ids) = serde_json::from_str::<Vec<i32>>(&text) {
                if ids.len() == seeds.len() { for (s, g) in seeds.iter_mut().zip(ids) { s.group = g; } }
            }
        }
        info!("ClusterSeeds loaded: {} seeds", seeds.len());
        Ok(Arc::new(Self { seeds }))
    }

}
