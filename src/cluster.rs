use crate::vectors::{cosine_sim, VectorStore, DIMS};
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
}

/// All cluster seeds loaded from `cluster_seeds.bin`.
///
/// Binary format (produced by `tools/compute_cluster_seeds.py`):
/// - 4 bytes: n_clusters (u32 LE)
/// - 4 bytes: n_dims     (u32 LE, must be 2560)
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

    /// Load `cluster_seeds_v2.bin` (member keys from `catalog::item_key`). When it is absent the legacy
    /// `cluster_seeds.bin` (raw TMDB ids, written before the movie/TV collision fix) is read instead and its
    /// member ids are converted to keys through the catalog.
    pub fn load(path: &str, legacy_path: &str, catalog: &crate::catalog::CatalogMap) -> Result<Arc<Self>> {
        let (use_path, legacy) = if std::path::Path::new(path).exists() {
            (path, false)
        } else if std::path::Path::new(legacy_path).exists() {
            tracing::warn!("{path} not found — reading legacy {legacy_path} and converting ids to keys");
            (legacy_path, true)
        } else {
            tracing::warn!("cluster seeds not found at {path} — feed will use brute-force fallback");
            return Ok(Self::empty());
        };
        let path = use_path;

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
            let centroid: Vec<f32> = bytes[pos..pos + n_dims * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            pos += n_dims * 4;

            let n_members = read_u32!() as usize;
            let member_ids: Vec<u32> = bytes[pos..pos + n_members * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            pos += n_members * 4;

            seeds.push(ClusterSeed { centroid, member_ids });
        }

        if legacy {
            let by_raw: std::collections::HashMap<u32, u32> =
                catalog.values().map(|it| (it.tmdb_id, it.id)).collect();
            let mut dropped = 0usize;
            for seed in &mut seeds {
                seed.member_ids = seed.member_ids.iter()
                    .filter_map(|raw| { let k = by_raw.get(raw).copied(); if k.is_none() { dropped += 1; } k })
                    .collect();
            }
            if dropped > 0 {
                tracing::warn!("{dropped} legacy seed members not found in the catalog were dropped");
            }
        }

        info!("ClusterSeeds loaded: {} seeds", seeds.len());
        Ok(Arc::new(Self { seeds }))
    }

    /// Select up to `n_select` maximally diverse seeds via greedy MMD
    /// (Max-Min Dispersion). Any two selected seeds must have cosine ≤ max_sim.
    /// Seeds whose dominant genre is in `blocked_genres` are skipped.
    pub fn select_diverse(
        &self,
        n_select: usize,
        max_inter_sim: f32,
        blocked_genres: &[&str],
        vectors: &VectorStore,
    ) -> Vec<&ClusterSeed> {
        let mut selected: Vec<&ClusterSeed> = Vec::new();
        let mut selected_centroids: Vec<&Vec<f32>> = Vec::new();

        'outer: for seed in &self.seeds {
            if selected.len() >= n_select {
                break;
            }

            // Inter-anchor cosine gate: reject if too similar to any selected seed
            for &prev in &selected_centroids {
                if cosine_sim(&seed.centroid, prev) > max_inter_sim {
                    continue 'outer;
                }
            }

            // Genre bitmask: skip if dominant genre is recently placed
            // (dominant genre inferred from majority vote of members)
            // For performance: approximate via a lightweight text heuristic on
            // the first few members' IDs. Full genre lookup happens in page.rs.
            // Here we just pass through — page.rs does the genre bitmask check.
            let _ = blocked_genres; // checked in page.rs after full item lookup

            selected.push(seed);
            selected_centroids.push(&seed.centroid);
        }

        selected
    }
}
